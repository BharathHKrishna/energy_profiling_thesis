import sys
import os

from google_crc32c import value
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import warnings
warnings.filterwarnings("ignore")

from scripts.utils.logger import get_logger
from scripts.utils.config_loader import load_config

logger = get_logger("viirs_extractor")
config = load_config("/srv/THESIS/energy_profiling_thesis/configs/config.yaml")

# ── GEE initialisation ────────────────────────────────────────────────────────
# Authenticated on server via:
#   earthengine authenticate --auth_mode=notebook
# Credentials at: ~/.config/earthengine/credentials
# Project: energy-thesis

try:
    import ee
    ee.Initialize(project=config.get("api", {}).get("gee_project", "energy-thesis"))
    GEE_AVAILABLE = True
    logger.info("GEE initialised successfully")
except Exception as e:
    GEE_AVAILABLE = False
    logger.warning(f"GEE initialisation failed: {e}")

# ── GEE dataset IDs ────────────────────────────────────────────────────────────
# Source: config.yaml + GEE catalogue
VIIRS_NTL_COLLECTION  = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
MODIS_NDVI_COLLECTION = "MODIS/061/MOD13A1"         # 16-day 500m NDVI
MODIS_SR_COLLECTION   = "MODIS/061/MOD09A1"         # 8-day 500m surface reflectance

# ── Date range for annual mean ─────────────────────────────────────────────────
# Use 2023 as the reference year — most recent complete year
DATE_START = "2023-01-01"
DATE_END   = "2023-12-31"

# ── Physical valid ranges ──────────────────────────────────────────────────────
# Values outside these are physically impossible — treat as nodata
VIIRS_VALID = {
    "ntl_radiance":  (0.0, 2000.0),    # nW/cm²/sr — theoretical max in brightest cities
    "ndvi":          (-1.0, 1.0),      # dimensionless — by definition
    "surface_refl":  (0.0, 1.0),       # reflectance fraction 0–1
}


# ── Core GEE sampling function ────────────────────────────────────────────────

def _sample_gee_point(image, lat, lon, band_name, scale_m=500):
    """
    Sample a single GEE image at a point.

    Returns:
        float or None — value at point, or None if GEE unavailable,
        image empty, or value outside valid range.
    """
    if not GEE_AVAILABLE:
        return None

    try:
        point  = ee.Geometry.Point([lon, lat])
        result = image.select(band_name).sample(
            region=point,
            scale=scale_m,
            numPixels=1,
            geometries=False
        ).first().getInfo()

        if result is None:
            return None

        properties = result.get("properties", {})
        value      = properties.get(band_name)

        if value is None:
            return None

        return float(value)

    except Exception as e:
        logger.warning(f"GEE sample failed for {band_name} at ({lat}, {lon}): {e}")
        return None


def _is_valid(value, feature_name):
    """Check if a value is within the physically valid range."""
    if value is None:
        return False
    valid_range = VIIRS_VALID.get(feature_name)
    if valid_range is None:
        return True
    return valid_range[0] <= value <= valid_range[1]


# ── Nighttime radiance ────────────────────────────────────────────────────────

def extract_viirs_ntl(lat, lon):
    """
    Extract VIIRS nighttime lights annual mean radiance (nW/cm²/sr).

    Uses NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG monthly composites,
    averaged over the reference year. The 'avg_rad' band gives
    average radiance — stray light corrected.

    Returns:
        float or None — annual mean radiance in nW/cm²/sr
        None when GEE unavailable, cloud-masked out, or invalid value.

    Physical interpretation:
        NTL > 100    = major city centre (Tokyo, NYC, Dubai)
        NTL 10–100   = suburban / industrial area
        NTL 1–10     = small town or sparse settlement
        NTL 0.5–1.0  = very rural, isolated light
        NTL < 0.5    = dark — uninhabited or energy-poor
    """
    if not GEE_AVAILABLE:
        return None

    try:
        collection = (
            ee.ImageCollection(VIIRS_NTL_COLLECTION)
            .filterDate(DATE_START, DATE_END)
            .select("avg_rad")
            .mean()
        )
        value = _sample_gee_point(collection, lat, lon, "avg_rad", scale_m=500)

        if not _is_valid(value, "ntl_radiance"):
            logger.info(
                f"VIIRS NTL: invalid value {value} at ({lat}, {lon}) — returning None"
            )
            return None

        logger.info(f"VIIRS NTL: {value:.3f} nW/cm²/sr at ({lat}, {lon})")
        if value is None or not _is_valid(value, "ntl_radiance"):
            return None
        return round(float(value), 3)

    except Exception as e:
        logger.warning(f"VIIRS NTL extraction failed at ({lat}, {lon}): {e}")
        return None


# ── NDVI ─────────────────────────────────────────────────────────────────────

def extract_viirs_ndvi(lat, lon):
    """
    Extract MODIS NDVI annual mean (-1 to +1).

    Uses MODIS/061/MOD13A1 16-day 500m composites.
    Band 'NDVI' is scaled by 0.0001 — we apply the scale factor.

    Returns:
        float or None — annual mean NDVI (-1 to +1)
        None when GEE unavailable or invalid.

    Physical interpretation:
        NDVI > 0.6   = dense forest (Amazon, Congo, Borneo)
        NDVI 0.4–0.6 = cropland, woodland
        NDVI 0.2–0.4 = shrubland, sparse vegetation
        NDVI 0.1–0.2 = Sahel, degraded land
        NDVI < 0.1   = bare desert, rock, water
        NDVI < 0     = water, snow, clouds
    """
    if not GEE_AVAILABLE:
        return None

    try:
        collection = (
            ee.ImageCollection(MODIS_NDVI_COLLECTION)
            .filterDate(DATE_START, DATE_END)
            .select("NDVI")
            .mean()
            # MODIS NDVI scale factor: multiply by 0.0001
            .multiply(0.0001)
        )
        value = _sample_gee_point(collection, lat, lon, "NDVI", scale_m=500)

        if not _is_valid(value, "ndvi"):
            logger.info(
                f"VIIRS NDVI: invalid value {value} at ({lat}, {lon}) — returning None"
            )
            return None

        logger.info(f"VIIRS NDVI: {value:.4f} at ({lat}, {lon})")
        if value is None or not _is_valid(value, "ndvi"):
            return None
        return round(float(value), 4)

    except Exception as e:
        logger.warning(f"VIIRS NDVI extraction failed at ({lat}, {lon}): {e}")
        return None


# ── Surface reflectance ───────────────────────────────────────────────────────

def extract_viirs_surface_reflectance(lat, lon):
    """
    Extract MODIS surface reflectance — band 1 (red, 620–670nm).

    Uses MODIS/061/MOD09A1 8-day 500m composites.
    Band 'sur_refl_b01' — red band, scale factor 0.0001.

    Returns:
        float or None — mean surface reflectance (0–1 fraction)
        None when GEE unavailable or invalid.

    Physical interpretation:
        Reflectance > 0.5  = bright desert, salt flat, snow
        Reflectance 0.2–0.5 = bare soil, urban concrete
        Reflectance 0.1–0.2 = dense vegetation (absorbs red)
        Reflectance < 0.1   = water, very dense canopy
    """
    if not GEE_AVAILABLE:
        return None

    try:
        collection = (
            ee.ImageCollection(MODIS_SR_COLLECTION)
            .filterDate(DATE_START, DATE_END)
            .select("sur_refl_b01")
            .mean()
            .multiply(0.0001)
        )
        value = _sample_gee_point(
            collection, lat, lon, "sur_refl_b01", scale_m=500
        )

        if not _is_valid(value, "surface_refl"):
            logger.info(
                f"VIIRS SR: invalid value {value} at ({lat}, {lon}) — returning None"
            )
            return None

        logger.info(f"VIIRS surface reflectance: {value:.4f} at ({lat}, {lon})")
        if value is None or not _is_valid(value, "surface_refl"):
            return None
        return round(float(value), 4)

    except Exception as e:
        logger.warning(
            f"VIIRS surface reflectance extraction failed at ({lat}, {lon}): {e}"
        )
        return None


# ── Main extraction function ───────────────────────────────────────────────────

def extract_viirs_features(lat, lon, min_lat=None, max_lat=None,
                           min_lon=None, max_lon=None):
    """
    Extract all 3 VIIRS/MODIS features for a coordinate.

    Note: GEE samples at a point (lat, lon), not the full bbox window.
    The point is the coordinate centroid. GEE's native resolution (500m)
    is larger than the 256m bbox anyway — the point sample is representative.

    Returns a flat dict with keys:
        viirs_ntl_nw_cm2_sr    — nighttime radiance (nW/cm²/sr)
        viirs_ndvi             — NDVI annual mean (-1 to +1)
        viirs_surface_refl     — surface reflectance band 1 (0–1)

    Any feature without valid data is absent from the returned dict.
    Never returns nodata sentinel values.
    """
    logger.info(f"Extracting VIIRS features for ({lat}, {lon})")

    features = {}

    ntl = extract_viirs_ntl(lat, lon)
    if ntl is not None:
        features["viirs_ntl_nw_cm2_sr"] = ntl

    ndvi = extract_viirs_ndvi(lat, lon)
    if ndvi is not None:
        features["viirs_ndvi"] = ndvi

    sr = extract_viirs_surface_reflectance(lat, lon)
    if sr is not None:
        features["viirs_surface_refl"] = sr

    populated = len(features)
    logger.info(
        f"VIIRS ({lat}, {lon}): {populated}/3 features populated — "
        f"{list(features.keys())}"
    )

    return features


# ── Main — quick test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.sampling.stratified_sampler import generate_bbox

    TEST_COORDS = [
        ("dense_urban",    52.5200,  13.4050),   # Berlin — expect moderate NTL, low NDVI
        ("forest",         -4.0000, -60.0000),   # Amazon — expect high NDVI, low NTL
        ("arid",           26.0000,   3.0000),   # Sahara — expect low NDVI, low NTL
        ("isolated_light",  61.5200,  60.5900),  # Yekaterinburg outskirts Russia
    ]

    print("\n=== VIIRS Extraction Test ===\n")
    if not GEE_AVAILABLE:
        print("WARNING: GEE not available — all results will be None")
        print("Run: earthengine authenticate --auth_mode=notebook on server first")
        print()

    for stratum, lat, lon in TEST_COORDS:
        result = extract_viirs_features(lat, lon)
        print(f"[{stratum}] ({lat}, {lon})")
        if result:
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print("  No features returned")
        print()