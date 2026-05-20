import sys
import os
import math
import json

sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

from scripts.utils.logger import get_logger
from scripts.extractors.worldcover_extractor import (
    extract_worldcover_features, strip_internal_features
)
from scripts.extractors.ghsl_extractor import extract_ghsl_features
from scripts.extractors.solar_atlas_extractor import extract_solar_features
from scripts.extractors.viirs_extractor import extract_viirs_features
from scripts.extractors.osm_extractor import extract_osm_features

logger = get_logger("feature_extractor")

# All 11 ESA WorldCover classes — always exploded as individual columns.
# If ESA succeeds but a class has 0 pixels, its pct = 0.0 (true value, not missing).
# If ESA fails entirely, all wc_* columns are absent.
_WC_CLASSES = [
    "water", "built_up", "tree_cover", "cropland", "bare_sparse",
    "shrubland", "grassland", "wetland", "mangrove", "moss_lichen", "snow_ice"
]


def _bbox(lat, lon, size_m=512):
    half = size_m / 2
    dlat = half / 111320
    dlon = half / (111320 * math.cos(math.radians(abs(lat) or 0.001)))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon
    # returns: min_lat, max_lat, min_lon, max_lon


def _process_worldcover(lat, lon, min_lat, max_lat, min_lon, max_lon, result, stratum_name):
    try:
        wc = extract_worldcover_features(lat, lon, min_lat, max_lat, min_lon, max_lon)
        wc = strip_internal_features(wc)

        if not wc:
            logger.warning(f"[{stratum_name}] ESA: no data returned")
            return

        # Explode wc_classes_json into 11 individual per-class % columns
        classes_raw = wc.pop("wc_classes_json", None)
        if classes_raw:
            classes = json.loads(classes_raw)
            for cls in _WC_CLASSES:
                result[f"wc_{cls}_pct"] = classes.get(cls, {}).get("pct", 0.0)
        else:
            # ESA succeeded but no class JSON — still add 0.0 for all classes
            for cls in _WC_CLASSES:
                result[f"wc_{cls}_pct"] = 0.0

        # Rename wc_std → wc_energy_score_std
        if "wc_std" in wc:
            wc["wc_energy_score_std"] = wc.pop("wc_std")

        result.update(wc)
        logger.info(f"[{stratum_name}] ESA: {len(wc) + len(_WC_CLASSES)} features")

    except Exception as e:
        logger.warning(f"[{stratum_name}] ESA WorldCover failed: {e}")


def extract_all_features(lat, lon, stratum_name, importance_tier="", strata_type="") -> dict:
    """
    Extract all PORE features for a single BORE-verified coordinate.

    Returns a flat dict with:
      - 6 metadata columns
      - 18 ESA WorldCover features (11 per-class %, 7 summary)
      - 3 GHSL features
      - 2 Solar Atlas features
      - 3 VIIRS features
      - 13 OSM features
    Sources: OSM, ESA WorldCover, VIIRS, GHSL, Global Solar Atlas.

    Null contract: absent key = no data. Never returns sentinel values.
    If an extractor fails, its keys are simply absent from the result.
    Exception: osm_building_count=0 is kept (0 buildings is valid information).
    """
    min_lat, max_lat, min_lon, max_lon = _bbox(lat, lon)

    result = {
        "stratum_name":    stratum_name,
        "importance_tier": importance_tier,
        "strata_type":     strata_type,
        "lat":             lat,
        "lon":             lon,
        "bbox_size_m":     512,
    }

    # ── 1. ESA WorldCover ────────────────────────────────────────────────────
    _process_worldcover(lat, lon, min_lat, max_lat, min_lon, max_lon, result, stratum_name)

    # ── 2. GHSL ──────────────────────────────────────────────────────────────
    try:
        ghsl = extract_ghsl_features(lat, lon, min_lat, max_lat, min_lon, max_lon)
        result.update(ghsl)
        logger.info(f"[{stratum_name}] GHSL: {len(ghsl)} features")
    except Exception as e:
        logger.warning(f"[{stratum_name}] GHSL failed: {e}")

    # ── 3. Global Solar Atlas ────────────────────────────────────────────────
    try:
        solar = extract_solar_features(lat, lon, min_lat, max_lat, min_lon, max_lon)
        result.update(solar)
        logger.info(f"[{stratum_name}] Solar: {len(solar)} features")
    except Exception as e:
        logger.warning(f"[{stratum_name}] Solar Atlas failed: {e}")

    # ── 4. VIIRS / MODIS via GEE ─────────────────────────────────────────────
    try:
        viirs = extract_viirs_features(lat, lon, min_lat, max_lat, min_lon, max_lon)
        result.update(viirs)
        logger.info(f"[{stratum_name}] VIIRS: {len(viirs)} features")
    except Exception as e:
        logger.warning(f"[{stratum_name}] VIIRS failed: {e}")

    # ── 5. OSM Overpass ───────────────────────────────────────────────────────
    try:
        osm = extract_osm_features(lat, lon, min_lat, max_lat, min_lon, max_lon)
        # OSM prepends lat, lon — they duplicate metadata, update order is harmless
        result.update(osm)
        logger.info(f"[{stratum_name}] OSM: {osm.get('raw_element_count', 0)} elements")
    except Exception as e:
        logger.warning(f"[{stratum_name}] OSM failed: {e}")

    feature_count = len(result) - 6  # subtract metadata keys
    logger.info(f"[{stratum_name}] ({lat:.4f}, {lon:.4f}) — {feature_count} feature keys populated")

    return result
