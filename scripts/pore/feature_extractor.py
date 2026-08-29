import sys
import os
import math
import json

sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

from scripts.utils.logger import get_logger
from scripts.extractors.worldcover_extractor import (
    extract_worldcover_features, strip_internal_features
)
from scripts.extractors.ghsl_extractor       import extract_ghsl_features, compute_floor_area
from scripts.extractors.solar_atlas_extractor import extract_solar_features
from scripts.extractors.osm_extractor         import extract_osm_features
from scripts.extractors.viirs_extractor       import extract_viirs_features
from scripts.extractors.climate_extractor      import extract_climate_features
from scripts.extractors.demand_extractor       import demand_from_features
from scripts.utils.geo                          import bbox as _bbox

logger = get_logger("feature_extractor")

_WC_CLASSES = [
    "water", "built_up", "tree_cover", "cropland", "bare_sparse",
    "shrubland", "grassland", "wetland", "mangrove", "moss_lichen", "snow_ice"
]


def _apply_worldcover(wc_raw, result, label=""):
    """Parse and merge a raw WorldCover result dict into result."""
    wc = strip_internal_features(wc_raw)
    if not wc:
        logger.warning(f"[{label}] ESA: no data")
        return

    classes_raw = wc.pop("wc_classes_json", None)
    if classes_raw:
        classes = json.loads(classes_raw)
        for cls in _WC_CLASSES:
            result[f"wc_{cls}_pct"] = classes.get(cls, {}).get("pct", 0.0)
    else:
        for cls in _WC_CLASSES:
            result[f"wc_{cls}_pct"] = 0.0

    if "wc_std" in wc:
        wc["wc_energy_score_std"] = wc.pop("wc_std")

    result.update(wc)
    logger.info(f"[{label}] ESA: {len(wc) + len(_WC_CLASSES)} features")


def merge_all_features(lat, lon, bbox_size_m, wc_raw, ghsl, solar, osm) -> dict:
    """
    Merge pre-fetched extractor results into a single flat feature dict.

    Called by the web app after running all extractors in parallel.
    Each argument may be an empty dict if its extractor failed -- the null
    contract (absent key = no data) is preserved throughout.

    Wind (Global Wind Atlas / Open-Meteo ERA5) removed 2026-08-27 -- see
    extract_all_features()'s docstring for why.
    """
    result = {
        "lat":         lat,
        "lon":         lon,
        "bbox_size_m": bbox_size_m,
    }

    if wc_raw:
        _apply_worldcover(wc_raw, result, label=f"{lat:.4f},{lon:.4f}")

    for data, source in [(ghsl, "GHSL"), (solar, "Solar")]:
        if data:
            result.update(data)
            logger.info(f"Merged {source}: {len(data)} features")

    if osm:
        osm_clean = {k: v for k, v in osm.items() if k not in ("lat", "lon")}
        result.update(osm_clean)
        logger.info(f"Merged OSM: {osm.get('raw_element_count', 0)} elements")

    return result


def extract_all_features(lat, lon, stratum_name="", importance_tier="",
                         strata_type="", bbox_size_m=512) -> dict:
    """
    Per-coordinate extraction, run concurrently rather than sequentially
    (changed 2026-08-27, live 10k run). All sources, plus the demand
    formula's own independent floor-area read, are mutually independent --
    none of them reads a key another one writes -- so they're fired at once
    in a thread pool instead of one after another. A thread pool, not a
    process pool, because every one of these is I/O wait (network calls or
    local file/raster reads), not CPU work, so the GIL releases during each
    one's real wait and true concurrency is possible. This was found to
    matter live: Earth Engine's climate call occasionally takes 30-95
    seconds on its own, and sequentially that got added on top of the other
    sources' time; concurrently, a coordinate's total extraction time
    becomes roughly its slowest single source, not the sum of all of them.

    Every dict write into `result` happens in the main thread only, after
    every task in the pool has finished, specifically so there is never a
    concurrent write to the shared dict from two threads at once, even
    though in practice no two sources share a key name anyway.

    Only the demand figure has a real dependency, on climate's own result
    and on floor_area, both already resolved by the time the pool exits, so
    it is computed after the pool, not inside it.

    Sources: ESA WorldCover, GHSL, Global Solar Atlas, OSM (local tile),
    VIIRS/MODIS (GEE), ERA5-Land climate (GEE).

    Global Wind Atlas / Open-Meteo ERA5 wind (speed + power density at 100m
    hub height) was removed 2026-08-27, after concurrent extraction made
    every worker's wind call land on Open-Meteo in the same instant instead
    of staggered across a sequential chain. That triggered real 429s, but a
    live isolated test (zero concurrency, single call) still got a 429 with
    the exact message "Hourly API request limit exceeded" -- proving it
    wasn't a burst/concurrency problem at all, but the free tier's hourly
    (5,000 calls) and daily (10,000 calls) caps. Each of our calls (one
    location, one variable, a full calendar year of hourly data) counts as
    ~2.15 "calls" against that budget (measured empirically: only 2,326 real
    requests fired in the hour before the block hit, well under the 5,000
    nominal cap, backing out the true per-request cost). At 10,000
    coordinates that is ~21,500 call-units -- over two full days' worth of
    the free tier's entire daily allowance for wind alone, dwarfing every
    other source, none of which carries any such cap. Moved to future scope
    rather than fixed; see thesis limitations/future-scope discussion.
    Null contract: absent key = no data. Never returns sentinel values.
    """
    from concurrent.futures import ThreadPoolExecutor

    min_lat, max_lat, min_lon, max_lon = _bbox(lat, lon, bbox_size_m)

    result = {
        "stratum_name":    stratum_name,
        "importance_tier": importance_tier,
        "strata_type":     strata_type,
        "lat":             lat,
        "lon":             lon,
        "bbox_size_m":     bbox_size_m,
    }

    tasks = {
        "worldcover": lambda: extract_worldcover_features(lat, lon, min_lat, max_lat, min_lon, max_lon),
        "ghsl":       lambda: extract_ghsl_features(lat, lon, min_lat, max_lat, min_lon, max_lon),
        "solar":      lambda: extract_solar_features(lat, lon, min_lat, max_lat, min_lon, max_lon),
        "viirs":      lambda: extract_viirs_features(lat, lon, min_lat, max_lat, min_lon, max_lon),
        "climate":    lambda: extract_climate_features(lat, lon, min_lat, max_lat, min_lon, max_lon),
        "osm":        lambda: extract_osm_features(lat, lon, min_lat, max_lat, min_lon, max_lon),
        "floor_area": lambda: compute_floor_area(min_lat, max_lat, min_lon, max_lon),
    }
    labels = {"worldcover": "ESA WorldCover", "ghsl": "GHSL", "solar": "Solar Atlas",
              "viirs": "VIIRS", "climate": "Climate",
              "osm": "OSM", "floor_area": "floor_area"}

    outputs = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for fut in futures:
            name = futures[fut]
            try:
                outputs[name] = fut.result()
            except Exception as e:
                logger.warning(f"[{stratum_name}] {labels[name]} failed: {e}")
                outputs[name] = None

    # 1. ESA WorldCover
    if outputs["worldcover"] is not None:
        _apply_worldcover(outputs["worldcover"], result, label=stratum_name)

    # 2-5. GHSL, Solar, VIIRS, Climate -- same update+log pattern each
    for name, log_label in (("ghsl", "GHSL"), ("solar", "Solar"),
                            ("viirs", "VIIRS"), ("climate", "Climate")):
        if outputs[name] is not None:
            result.update(outputs[name])
            logger.info(f"[{stratum_name}] {log_label}: {len(outputs[name])} features")

    # 6b. Heating/cooling demand -- floor_area is its own overlap-weighted, per-cell
    # built x height read across the whole bbox (compute_floor_area(), fixed 2026-08-14),
    # not derived from the mean-based ghsl_built_surface_m2/ghsl_building_height_m
    # already in `result` (those stay mean-based for the general feature record).
    # Climate values reused from `result`, no extra GEE call (see demand_extractor.py
    # docstring for the geographic-regression formula and its known limitations -- no
    # DHW term, cooling's out-of-sample accuracy is weaker than heating's).
    try:
        demand = demand_from_features(
            outputs["floor_area"],
            result.get("climate_hdd"),
            result.get("climate_cdd24"),
            lat, lon,
            result.get("climate_mean_temp_c"),
        )
        result.update(demand)
        if demand:
            logger.info(
                f"[{stratum_name}] Demand: heating={demand.get('heating_MWh')} MWh, "
                f"cooling={demand.get('cooling_MWh')} MWh"
            )
    except Exception as e:
        logger.warning(f"[{stratum_name}] Demand calc failed: {e}")

    # 7. OSM (local tile, already fetched before this function is called)
    if outputs["osm"] is not None:
        result.update(outputs["osm"])
        logger.info(f"[{stratum_name}] OSM: {outputs['osm'].get('raw_element_count', 0)} elements")  # noqa

    logger.info(
        f"[{stratum_name}] ({lat:.4f}, {lon:.4f}) -- "
        f"{len(result) - 6} feature keys populated"
    )
    return result
