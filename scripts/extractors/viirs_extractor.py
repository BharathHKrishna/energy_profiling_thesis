import sys
import os

sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import warnings
warnings.filterwarnings("ignore")

from scripts.utils.logger import get_logger
from scripts.utils.config_loader import load_config

logger = get_logger("viirs_extractor")
config = load_config("/srv/THESIS/energy_profiling_thesis/configs/config.yaml")

# ── GEE initialisation ────────────────────────────────────────────────────────
try:
    import ee
    ee.Initialize(project=config.get("api", {}).get("gee_project", "energy-thesis"))
    GEE_AVAILABLE = True
    logger.info("GEE initialised successfully")
except Exception as e:
    GEE_AVAILABLE = False
    logger.warning(f"GEE initialisation failed: {e}")

# ── GEE dataset IDs ────────────────────────────────────────────────────────────
VIIRS_NTL_COLLECTION  = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
MODIS_NDVI_COLLECTION = "MODIS/061/MOD13A1"
MODIS_SR_COLLECTION   = "MODIS/061/MOD09A1"

# ── Date range for annual mean ─────────────────────────────────────────────────
DATE_START = "2023-01-01"
DATE_END   = "2023-12-31"

# ── Physical valid ranges ──────────────────────────────────────────────────────
VIIRS_VALID = {
    "ntl_radiance":  (0.0, 2000.0),
    "ndvi":          (-1.0, 1.0),
    "surface_refl":  (0.0, 1.0),
}


# ── Core GEE sampling function ────────────────────────────────────────────────

def _sample_gee_bbox(image, min_lat, max_lat, min_lon, max_lon, band_name, scale_m=500):
    """
    Reduce a GEE image band over a bbox using mean.

    Uses ee.Reducer.mean() over the bounding box rectangle.
    At VIIRS/MODIS 500m resolution and a 512m bbox, this covers
    1–4 source pixels depending on where the bbox falls on the pixel grid.
    The mean of those pixels is the correct bbox-representative value.

    Returns:
        float or None
    """
    if not GEE_AVAILABLE:
        return None

    try:
        region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
        result = (
            image.select(band_name)
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=scale_m,
                maxPixels=1e6
            )
            .getInfo()
        )
        value = result.get(band_name)
        if value is None:
            return None
        return float(value)

    except Exception as e:
        logger.warning(f"GEE bbox reduce failed for {band_name}: {e}")
        return None


def _is_valid(value, feature_name):
    if value is None:
        return False
    valid_range = VIIRS_VALID.get(feature_name)
    if valid_range is None:
        return True
    return valid_range[0] <= value <= valid_range[1]


# ── Nighttime radiance ────────────────────────────────────────────────────────

def extract_viirs_ntl(min_lat, max_lat, min_lon, max_lon):
    """
    Extract VIIRS nighttime lights annual mean radiance over a 512×512m bbox.

    Uses NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG monthly composites averaged over
    2023. Reduces the bbox via mean — at 500m native resolution this covers
    1–4 source pixels. Returns the mean radiance over the bbox.

    Returns:
        float or None — annual mean radiance in nW/cm²/sr

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
        value = _sample_gee_bbox(
            collection, min_lat, max_lat, min_lon, max_lon, "avg_rad", scale_m=500
        )

        if not _is_valid(value, "ntl_radiance"):
            logger.info(f"VIIRS NTL: invalid value {value} — returning None")
            return None

        logger.info(f"VIIRS NTL: {value:.3f} nW/cm²/sr")
        return round(float(value), 3)

    except Exception as e:
        logger.warning(f"VIIRS NTL extraction failed: {e}")
        return None


# ── NDVI ──────────────────────────────────────────────────────────────────────

def extract_viirs_ndvi(min_lat, max_lat, min_lon, max_lon):
    """
    Extract MODIS NDVI annual mean over a 512×512m bbox.

    Uses MODIS/061/MOD13A1 16-day 500m composites. Scale factor 0.0001 applied.
    Reduces the bbox via mean.

    Returns:
        float or None — annual mean NDVI (-1 to +1)

    Physical interpretation:
        NDVI > 0.6   = dense forest
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
            .multiply(0.0001)
        )
        value = _sample_gee_bbox(
            collection, min_lat, max_lat, min_lon, max_lon, "NDVI", scale_m=500
        )

        if not _is_valid(value, "ndvi"):
            logger.info(f"VIIRS NDVI: invalid value {value} — returning None")
            return None

        logger.info(f"VIIRS NDVI: {value:.4f}")
        return round(float(value), 4)

    except Exception as e:
        logger.warning(f"VIIRS NDVI extraction failed: {e}")
        return None


# ── Surface reflectance ───────────────────────────────────────────────────────

def extract_viirs_surface_reflectance(min_lat, max_lat, min_lon, max_lon):
    """
    Extract MODIS surface reflectance band 1 (red, 620–670nm) over a 512×512m bbox.

    Uses MODIS/061/MOD09A1 8-day 500m composites. Scale factor 0.0001 applied.
    Reduces the bbox via mean.

    Returns:
        float or None — mean surface reflectance (0–1 fraction)

    Physical interpretation:
        Reflectance > 0.5   = bright desert, salt flat, snow
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
        value = _sample_gee_bbox(
            collection, min_lat, max_lat, min_lon, max_lon, "sur_refl_b01", scale_m=500
        )

        if not _is_valid(value, "surface_refl"):
            logger.info(f"VIIRS SR: invalid value {value} — returning None")
            return None

        logger.info(f"VIIRS surface reflectance: {value:.4f}")
        return round(float(value), 4)

    except Exception as e:
        logger.warning(f"VIIRS surface reflectance extraction failed: {e}")
        return None


# ── Main extraction function ───────────────────────────────────────────────────

def extract_viirs_features(lat, lon, min_lat, max_lat, min_lon, max_lon):
    """
    Extract all 3 VIIRS/MODIS features over a 512×512m bounding box.

    All three features are reduced over the bbox via mean at 500m native
    resolution — typically 1–4 source pixels per bbox.

    Returns a flat dict with keys:
        viirs_ntl_nw_cm2_sr    — nighttime radiance (nW/cm²/sr)
        viirs_ndvi             — NDVI annual mean (-1 to +1)
        viirs_surface_refl     — surface reflectance band 1 (0–1)

    Any feature without valid data is absent from the returned dict.
    Never returns nodata sentinel values.
    """
    logger.info(f"Extracting VIIRS features for bbox ({min_lat:.4f},{min_lon:.4f})–({max_lat:.4f},{max_lon:.4f})")

    features = {}

    ntl = extract_viirs_ntl(min_lat, max_lat, min_lon, max_lon)
    if ntl is not None:
        features["viirs_ntl_nw_cm2_sr"] = ntl

    ndvi = extract_viirs_ndvi(min_lat, max_lat, min_lon, max_lon)
    if ndvi is not None:
        features["viirs_ndvi"] = ndvi

    sr = extract_viirs_surface_reflectance(min_lat, max_lat, min_lon, max_lon)
    if sr is not None:
        features["viirs_surface_refl"] = sr

    populated = len(features)
    logger.info(
        f"VIIRS: {populated}/3 features populated — {list(features.keys())}"
    )

    return features


# ── Main — quick test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math
    def generate_bbox(lat, lon, size_m=512):
        half = size_m / 2
        dlat = half / 111320
        dlon = half / (111320 * math.cos(math.radians(abs(lat) or 0.001)))
        return dict(min_lat=lat-dlat, max_lat=lat+dlat, min_lon=lon-dlon, max_lon=lon+dlon)

    TEST_COORDS = [
        ("dense_urban",    52.5200,  13.4050),
        ("forest",         -4.0000, -60.0000),
        ("arid",           26.0000,   3.0000),
        ("isolated_light",  61.5200,  60.5900),
    ]

    print("\n=== VIIRS Extraction Test (512m bbox) ===\n")
    if not GEE_AVAILABLE:
        print("WARNING: GEE not available — all results will be None\n")

    for stratum, lat, lon in TEST_COORDS:
        bbox = generate_bbox(lat, lon)
        result = extract_viirs_features(
            lat, lon,
            bbox["min_lat"], bbox["max_lat"],
            bbox["min_lon"], bbox["max_lon"]
        )
        print(f"[{stratum}] ({lat}, {lon})")
        if result:
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print("  No features returned")
        print()
