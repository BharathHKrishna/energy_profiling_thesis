import sys
import os
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import numpy as np
import warnings
from math import floor, ceil
warnings.filterwarnings("ignore")

import rasterio
from rasterio.windows import from_bounds as rasterio_from_bounds

from scripts.utils.logger import get_logger
from scripts.utils.config_loader import load_config
from rasterio.windows import Window as RasterioWindow

logger = get_logger("solar_atlas_extractor")
config = load_config("/srv/THESIS/energy_profiling_thesis/configs/config.yaml")

# ── Raster paths ───────────────────────────────────────────────────────────────
# Solar Atlas rasters are in EPSG:4326 — no reprojection needed
BASE = "/srv/THESIS/energy_profiling_thesis"

SOLAR_PATHS = {
    "pvout": os.path.join(BASE, config["rasters"]["global_solar_atlas"]["pvout"]),
    "ghi":   os.path.join(BASE, config["rasters"]["global_solar_atlas"]["ghi"]),
}

# ── Physical valid ranges per layer ───────────────────────────────────────────
SOLAR_VALID_RANGES = {
    "pvout": (0.01, 12.0),
    "ghi":   (0.01, 12.0),
}

# ── Output feature names ───────────────────────────────────────────────────────
SOLAR_FEATURE_NAMES = {
    "pvout": "solar_pvout_kwh_kwp_day",
    "ghi":   "solar_ghi_kwh_m2_day",
}


# ── Core sampling function ────────────────────────────────────────────────────

def sample_solar_layer(layer_name, min_lat, max_lat, min_lon, max_lon):
    raster_path = SOLAR_PATHS.get(layer_name)
    if not raster_path or not os.path.exists(raster_path):
        logger.warning(f"Solar Atlas: file not found: {raster_path}")
        return None

    valid_min, valid_max = SOLAR_VALID_RANGES[layer_name]

    try:
        with rasterio.open(raster_path) as src:
            window = rasterio_from_bounds(
                min_lon, min_lat, max_lon, max_lat,
                src.transform
            )

            # ── Window too small fix ───────────────────────────────────────────
            # Solar Atlas is ~1km resolution — a 512m bbox is sub-pixel.
            # Expand window by 1 pixel in each direction to guarantee a read.
            from math import floor, ceil
            window = RasterioWindow(
                col_off=int(floor(window.col_off)),   # type: ignore
                row_off=int(floor(window.row_off)),   # type: ignore
                width=int(max(1, ceil(window.width))),   # type: ignore
                height=int(max(1, ceil(window.height)))   # type: ignore
            )

            if (window.col_off < 0 or window.row_off < 0 or
                    window.col_off >= src.width or window.row_off >= src.height):
                return None

            data   = src.read(1, window=window)
            nodata = src.nodata

        if data.size == 0:
            return None

        valid_mask = np.ones(data.shape, dtype=bool)
        if nodata is not None:
            valid_mask &= (data != nodata)
        valid_mask &= (data >= valid_min) & (data <= valid_max)

        valid_pixels = data[valid_mask]
        if len(valid_pixels) == 0:
            return None

        result = round(float(np.mean(valid_pixels)), 3)
        logger.info(f"Solar Atlas {layer_name}: {len(valid_pixels)} valid pixels, mean = {result}")
        return result

    except Exception as e:
        logger.warning(f"Solar Atlas {layer_name}: read error — {e}")
        return None

# ── Main extraction function ───────────────────────────────────────────────────

def extract_solar_features(lat, lon, min_lat, max_lat, min_lon, max_lon):
    """
    Extract all 3 Solar Atlas features for a 512×512m bounding box.

    Returns a flat dict with keys:
        solar_pvout_kwh_kwp_day  — photovoltaic power output potential
        solar_ghi_kwh_m2_day     — global horizontal irradiance

    Any feature without valid data is absent from the returned dict.
    Never returns nodata sentinel values.

    Physical interpretation:
        PVOUT > 2000 = exceptional solar potential (Atacama, Sahara)
        PVOUT 1500-2000 = very good (MENA, India, SW USA)
        PVOUT 1000-1500 = good (Mediterranean, SE Asia)
        PVOUT < 1000 = moderate or poor (Northern Europe)
    """
    logger.info(f"Extracting Solar Atlas features for ({lat}, {lon})")

    features = {}

    for layer_name, feature_name in SOLAR_FEATURE_NAMES.items():
        value = sample_solar_layer(layer_name, min_lat, max_lat, min_lon, max_lon)
        if value is not None:
            features[feature_name] = value

    populated = len(features)
    logger.info(
        f"Solar Atlas ({lat}, {lon}): {populated}/2 features populated — "
        f"{list(features.keys())}"
    )

    return features


# ── Main — quick test on 3 coordinates ────────────────────────────────────────

if __name__ == "__main__":
    import math
    def generate_bbox(lat, lon, size_m=512):
        half = size_m / 2
        dlat = half / 111320
        dlon = half / (111320 * math.cos(math.radians(abs(lat) or 0.001)))
        return dict(min_lat=lat-dlat, max_lat=lat+dlat, min_lon=lon-dlon, max_lon=lon+dlon)

    TEST_COORDS = [
        ("arid_high_solar",    26.0000,   3.0000),   # Sahara — expect very high values
        ("dense_urban",        52.5200,  13.4050),   # Berlin — expect moderate values
        ("alpine",             46.4500,   6.8500),   # Swiss Alps — expect moderate/low
    ]

    print("\n=== Solar Atlas Extraction Test ===\n")
    for stratum, lat, lon in TEST_COORDS:
        bbox   = generate_bbox(lat, lon)
        result = extract_solar_features(
            lat, lon,
            bbox["min_lat"], bbox["max_lat"],
            bbox["min_lon"], bbox["max_lon"]
        )
        print(f"[{stratum}] ({lat}, {lon})")
        if result:
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print("  No features returned (all nodata)")
        print()