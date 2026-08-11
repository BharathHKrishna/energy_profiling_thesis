import sys
import os
import math
import json

sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

from scripts.utils.logger import get_logger
from scripts.extractors.worldcover_extractor import (
    extract_worldcover_features, strip_internal_features
)
from scripts.extractors.ghsl_extractor       import extract_ghsl_features
from scripts.extractors.solar_atlas_extractor import extract_solar_features
from scripts.extractors.wind_atlas_extractor  import extract_wind_features
from scripts.extractors.osm_extractor         import extract_osm_features
from scripts.extractors.viirs_extractor       import extract_viirs_features
from scripts.extractors.climate_extractor      import extract_climate_features
from scripts.extractors.demand_extractor       import demand_from_features
from scripts.extractors.demand_score           import demand_score_for_coord
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


def merge_all_features(lat, lon, bbox_size_m, wc_raw, ghsl, solar, wind, osm) -> dict:
    """
    Merge pre-fetched extractor results into a single flat feature dict.

    Called by the web app after running all extractors in parallel.
    Each argument may be an empty dict if its extractor failed — the null
    contract (absent key = no data) is preserved throughout.
    """
    result = {
        "lat":         lat,
        "lon":         lon,
        "bbox_size_m": bbox_size_m,
    }

    if wc_raw:
        _apply_worldcover(wc_raw, result, label=f"{lat:.4f},{lon:.4f}")

    for data, source in [(ghsl, "GHSL"), (solar, "Solar"), (wind, "Wind")]:
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
    Sequential extraction pipeline for one coordinate (used by BORE/PORE batch runs).

    Sources: ESA WorldCover, GHSL, Global Solar Atlas, Global Wind Atlas, OSM Overpass.
    Null contract: absent key = no data. Never returns sentinel values.
    """
    min_lat, max_lat, min_lon, max_lon = _bbox(lat, lon, bbox_size_m)

    result = {
        "stratum_name":    stratum_name,
        "importance_tier": importance_tier,
        "strata_type":     strata_type,
        "lat":             lat,
        "lon":             lon,
        "bbox_size_m":     bbox_size_m,
    }

    # 1. ESA WorldCover
    try:
        wc = extract_worldcover_features(lat, lon, min_lat, max_lat, min_lon, max_lon)
        _apply_worldcover(wc, result, label=stratum_name)
    except Exception as e:
        logger.warning(f"[{stratum_name}] ESA WorldCover failed: {e}")

    # 2. GHSL
    try:
        ghsl = extract_ghsl_features(lat, lon, min_lat, max_lat, min_lon, max_lon)
        result.update(ghsl)
        logger.info(f"[{stratum_name}] GHSL: {len(ghsl)} features")
    except Exception as e:
        logger.warning(f"[{stratum_name}] GHSL failed: {e}")

    # 3. Global Solar Atlas
    try:
        solar = extract_solar_features(lat, lon, min_lat, max_lat, min_lon, max_lon)
        result.update(solar)
        logger.info(f"[{stratum_name}] Solar: {len(solar)} features")
    except Exception as e:
        logger.warning(f"[{stratum_name}] Solar Atlas failed: {e}")

    # 4. Global Wind Atlas
    try:
        wind = extract_wind_features(lat, lon, min_lat, max_lat, min_lon, max_lon)
        result.update(wind)
        logger.info(f"[{stratum_name}] Wind: {len(wind)} features")
    except Exception as e:
        logger.warning(f"[{stratum_name}] Wind Atlas failed: {e}")

    # 5. VIIRS Nighttime Lights (GEE)
    try:
        viirs = extract_viirs_features(lat, lon, min_lat, max_lat, min_lon, max_lon)
        result.update(viirs)
        logger.info(f"[{stratum_name}] VIIRS: {len(viirs)} features")
    except Exception as e:
        logger.warning(f"[{stratum_name}] VIIRS failed: {e}")

    # 6. Climate — HDD/CDD from ERA5-Land (GEE)
    try:
        climate = extract_climate_features(lat, lon, min_lat, max_lat, min_lon, max_lon)
        result.update(climate)
        logger.info(f"[{stratum_name}] Climate: {len(climate)} features")
    except Exception as e:
        logger.warning(f"[{stratum_name}] Climate failed: {e}")

    # 6b. Heating/cooling demand — derived from GHSL + climate values already in `result`
    # above, no extra GHSL read or GEE call (see demand_extractor.py docstring for the
    # Eurostat-fit formula and its known limitations — no DHW term, cooling is metered-use
    # not pure need).
    try:
        demand = demand_from_features(
            result.get("ghsl_built_surface_m2"),
            result.get("ghsl_building_height_m"),
            result.get("climate_hdd"),
            result.get("climate_cdd24"),
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

    # 7. OSM Overpass
    try:
        osm = extract_osm_features(lat, lon, min_lat, max_lat, min_lon, max_lon)
        result.update(osm)
        logger.info(f"[{stratum_name}] OSM: {osm.get('raw_element_count', 0)} elements")  # noqa
    except Exception as e:
        logger.warning(f"[{stratum_name}] OSM failed: {e}")

    # 7b. Energy-demand score — synthetic 0-100 tier, single-cell (see demand_score.py
    # docstring: not the notebook's validated 9-cell mean). Reuses built/height/VIIRS
    # from `result` above, but does its own separate local OSM-gate read
    # (extract_osm_use, distinct from step 7's extract_osm_features).
    try:
        score = demand_score_for_coord(
            lat, lon,
            result.get("ghsl_built_surface_m2"),
            result.get("ghsl_building_height_m"),
            result.get("viirs_ntl_nw_cm2_sr"),
        )
        result.update(score)
        if score:
            logger.info(
                f"[{stratum_name}] Demand score: {score.get('demand_score')} "
                f"({score.get('demand_tier')})"
            )
    except Exception as e:
        logger.warning(f"[{stratum_name}] Demand score failed: {e}")

    logger.info(
        f"[{stratum_name}] ({lat:.4f}, {lon:.4f}) — "
        f"{len(result) - 6} feature keys populated"
    )
    return result
