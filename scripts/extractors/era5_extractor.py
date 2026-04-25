import sys
import os
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import rasterio
from rasterio.windows import from_bounds as rasterio_from_bounds

from scripts.utils.logger import get_logger
from scripts.utils.config_loader import load_config

logger = get_logger("era5_extractor")
config = load_config("/srv/THESIS/energy_profiling_thesis/configs/config.yaml")

BASE = "/srv/THESIS/energy_profiling_thesis"

# ── ERA5 raster path ───────────────────────────────────────────────────────────
# Pre-downloaded as single global annual mean raster
# Variable: surface_solar_radiation_downwards (ssrd), J/m²
ERA5_SSRD_PATH = os.path.join(
    BASE,
    config.get("rasters", {}).get("copernicus", {}).get("era5", "rasters/copernicus/era5/"),
    "era5_ssrd_annual_mean.tif"
)

# ── Physical valid range ───────────────────────────────────────────────────────
# SSRD annual sum: 0 to ~10,000 MJ/m² (poles to tropics)
# As daily mean J/m²: 0 to ~30,000,000 J/m²/day
# Annual mean W/m²: 0 to ~350 W/m²
# We store J/m²/day — valid range 0 to 40,000,000
ERA5_SSRD_VALID_MIN = 0.0
ERA5_SSRD_VALID_MAX = 40_000_000.0

# ── ERA5 uses EPSG:4326 ────────────────────────────────────────────────────────
# No reprojection needed — same CRS as bbox


from math import floor, ceil

def sample_era5_ssrd(min_lat, max_lat, min_lon, max_lon):
    if not os.path.exists(ERA5_SSRD_PATH):
        logger.warning(f"ERA5: SSRD raster not found at {ERA5_SSRD_PATH}")
        return None

    try:
        with rasterio.open(ERA5_SSRD_PATH) as src:
            window = rasterio_from_bounds(
                min_lon, min_lat, max_lon, max_lat,
                src.transform
            )

            # ERA5 is 0.25° resolution — 256m bbox is sub-pixel
            # Expand to minimum 1 pixel
            window = rasterio.windows.Window( # type: ignore
                int(floor(window.col_off)),
                int(floor(window.row_off)),
                int(max(1, ceil(window.width))),
                int(max(1, ceil(window.height)))
            )

            if (window.col_off < 0 or window.row_off < 0 or
                    window.col_off >= src.width or
                    window.row_off >= src.height):
                return None

            data   = src.read(1, window=window)
            nodata = src.nodata

        if data.size == 0:
            return None

        valid_mask = np.ones(data.shape, dtype=bool)
        if nodata is not None:
            valid_mask &= (data != nodata)
        valid_mask &= (
            (data >= ERA5_SSRD_VALID_MIN) &
            (data <= ERA5_SSRD_VALID_MAX)
        )

        valid_pixels = data[valid_mask]
        if len(valid_pixels) == 0:
            logger.info("ERA5: no valid SSRD pixels — returning None")
            return None

        result = float(np.mean(valid_pixels))
        logger.info(
            f"ERA5 SSRD: {len(valid_pixels)} valid pixels, "
            f"mean = {result:.0f} J/m2/day"
        )
        return round(result, 0)

    except Exception as e:
        logger.warning(f"ERA5 SSRD: read error — {e}")
        return None


def extract_era5_features(lat, lon, min_lat, max_lat, min_lon, max_lon):
    """
    Extract ERA5 features for a 256×256m bounding box.

    Returns a flat dict with keys:
        era5_ssrd_j_m2_day — surface solar radiation downwards (J/m²/day annual mean)

    Any feature without valid data is absent from the returned dict.
    Never returns nodata sentinel values.

    Physical interpretation:
        SSRD > 20,000,000 J/m²/day = very high solar resource (Sahara, Arabia)
        SSRD 15,000,000-20,000,000  = high (Mediterranean, India, SW USA)
        SSRD 10,000,000-15,000,000  = moderate (central Europe, East Asia)
        SSRD < 10,000,000           = low (Northern Europe, high latitudes)
    """
    logger.info(f"Extracting ERA5 features for ({lat}, {lon})")

    features = {}

    ssrd = sample_era5_ssrd(min_lat, max_lat, min_lon, max_lon)
    if ssrd is not None:
        features["era5_ssrd_j_m2_day"] = ssrd

    populated = len(features)
    logger.info(
        f"ERA5 ({lat}, {lon}): {populated}/1 features populated"
    )

    return features


# ── Main — quick test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.sampling.stratified_sampler import generate_bbox

    TEST_COORDS = [
        ("arid_high_solar", 26.0000,   3.0000),   # Sahara — expect very high SSRD
        ("dense_urban",     52.5200,  13.4050),   # Berlin — expect moderate SSRD
        ("alpine",          46.4500,   6.8500),   # Swiss Alps — expect moderate
    ]

    print("\n=== ERA5 Extraction Test ===\n")
    for stratum, lat, lon in TEST_COORDS:
        bbox   = generate_bbox(lat, lon)
        result = extract_era5_features(
            lat, lon,
            bbox["min_lat"], bbox["max_lat"],
            bbox["min_lon"], bbox["max_lon"]
        )
        print(f"[{stratum}] ({lat}, {lon})")
        if result:
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print("  No features returned (ERA5 raster likely not yet downloaded)")
            print("  — Run ERA5 pre-download script in Week 3 Day 1 first")
        print()