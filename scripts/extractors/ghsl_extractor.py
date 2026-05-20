import sys
import os
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import rasterio
from rasterio.windows import from_bounds as rasterio_from_bounds
from pyproj import Transformer

from scripts.utils.logger import get_logger
from scripts.utils.config_loader import load_config

logger = get_logger("ghsl_extractor")
config = load_config("/srv/THESIS/energy_profiling_thesis/configs/config.yaml")

# ── Raster paths from config ───────────────────────────────────────────────────
BASE = "/srv/THESIS/energy_profiling_thesis"

GHSL_PATHS = {
    "built_surface":   os.path.join(BASE, config["rasters"]["ghsl"]["built_surface"]),
    "building_height": os.path.join(BASE, config["rasters"]["ghsl"]["building_height"]),
    "population":      os.path.join(BASE, config["rasters"]["ghsl"]["population"]),
}

# ── GHSL uses Mollweide projection ESRI:54009 ──────────────────────────────────
# We must reproject bbox from WGS84 to Mollweide before sampling
TRANSFORMER_TO_MOLL = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)

# ── Nodata values per layer ────────────────────────────────────────────────────
GHSL_NODATA = {
    "built_surface":   -200.0,
    "building_height": -200.0,
    "population":      -200.0,
}


# ── Core sampling function ────────────────────────────────────────────────────

def sample_ghsl_layer(layer_name, min_lat, max_lat, min_lon, max_lon):
    """
    Sample one GHSL raster layer within a 512×512m bounding box.

    GHSL rasters are in Mollweide projection (ESRI:54009).
    Bbox is given in WGS84 (EPSG:4326).
    We reproject the bbox corners to Mollweide, read the window,
    then compute the mean of valid pixels.

    Returns:
        float or None — mean value of valid pixels, or None if no valid data.
        None is returned when:
          - raster file not found
          - bbox falls entirely outside raster extent
          - all pixels are nodata
          - any read exception
        Never returns a nodata sentinel value.
    """
    raster_path = GHSL_PATHS.get(layer_name)
    if not raster_path:
        logger.warning(f"GHSL: unknown layer name '{layer_name}'")
        return None

    if not os.path.exists(raster_path):
        logger.warning(f"GHSL: raster file not found: {raster_path}")
        return None

    try:
        # Reproject bbox corners from WGS84 → Mollweide
        # Use all 4 corners to handle any projection distortion
        xs, ys = TRANSFORMER_TO_MOLL.transform(
            [min_lon, max_lon, min_lon, max_lon],
            [min_lat, min_lat, max_lat, max_lat]
        )
        moll_min_x = min(xs)
        moll_max_x = max(xs)
        moll_min_y = min(ys)
        moll_max_y = max(ys)

        with rasterio.open(raster_path) as src:
            window = rasterio_from_bounds(
                moll_min_x, moll_min_y,
                moll_max_x, moll_max_y,
                src.transform
            )

            # Check window is within raster extent
            if (window.width <= 0 or window.height <= 0 or
                    window.col_off < 0 or window.row_off < 0 or
                    window.col_off >= src.width or window.row_off >= src.height):
                logger.warning(
                    f"GHSL {layer_name}: bbox outside raster extent "
                    f"for window {window}"
                )
                return None

            data = src.read(1, window=window)

        if data.size == 0:
            return None

        # Remove nodata values — never return nodata sentinels
        nodata_val = GHSL_NODATA.get(layer_name, -200.0)
        valid_mask = (data != nodata_val) & (data > -100)  # extra safety guard
        valid_pixels = data[valid_mask]

        if len(valid_pixels) == 0:
            logger.info(
                f"GHSL {layer_name}: all pixels are nodata — returning None"
            )
            return None

        result = float(np.mean(valid_pixels))
        logger.info(
            f"GHSL {layer_name}: {len(valid_pixels)} valid pixels, "
            f"mean = {result:.2f}"
        )
        return round(result, 2)

    except Exception as e:
        logger.warning(f"GHSL {layer_name}: read error — {e}")
        return None


def sample_degurba(min_lat, max_lat, min_lon, max_lon):
    """
    Sample DEGURBA (urbanisation class) layer.
    Returns the most common valid class code as a string label,
    or None if no valid data.

    DEGURBA returns a categorical class, not a continuous value.
    We return the mode (most frequent class) rather than the mean.
    """
    raster_path = GHSL_PATHS["degurba"]

    if not os.path.exists(raster_path):
        logger.warning(f"GHSL: DEGURBA raster not found: {raster_path}")
        return None

    try:
        xs, ys = TRANSFORMER_TO_MOLL.transform(
            [min_lon, max_lon, min_lon, max_lon],
            [min_lat, min_lat, max_lat, max_lat]
        )
        moll_min_x = min(xs)
        moll_max_x = max(xs)
        moll_min_y = min(ys)
        moll_max_y = max(ys)

        with rasterio.open(raster_path) as src:
            window = rasterio_from_bounds(
                moll_min_x, moll_min_y,
                moll_max_x, moll_max_y,
                src.transform
            )
            if window.width <= 0 or window.height <= 0:
                return None
            data = src.read(1, window=window)

        if data.size == 0:
            return None

        # Remove nodata (255)
        valid_pixels = data[data != 255]

        if len(valid_pixels) == 0:
            return None

        # Return mode class label
        values, counts = np.unique(valid_pixels, return_counts=True)
        dominant_class = int(values[np.argmax(counts)])
        label = DEGURBA_LABELS.get(dominant_class, f"class_{dominant_class}")

        logger.info(
            f"GHSL DEGURBA: dominant class = {dominant_class} ({label})"
        )
        return label

    except Exception as e:
        logger.warning(f"GHSL DEGURBA: read error — {e}")
        return None


# ── Main extraction function ───────────────────────────────────────────────────

def extract_ghsl_features(lat, lon, min_lat, max_lat, min_lon, max_lon):
    """
    Extract GHSL features for a 512×512m bounding box.

    Returns a flat dict with keys:
        ghsl_built_surface_m2    — mean built-up surface area (m²) per 100m cell
        ghsl_building_height_m   — mean building height (metres)
        ghsl_population_per_km2  — mean population density (persons/km²)

    Any feature without valid data is simply absent from the returned dict.
    Never returns nodata sentinel values.
    """
    logger.info(f"Extracting GHSL features for ({lat}, {lon})")

    features = {}

    # ── Built surface ──────────────────────────────────────────────────────────
    built = sample_ghsl_layer("built_surface", min_lat, max_lat, min_lon, max_lon)
    if built is not None:
        features["ghsl_built_surface_m2"] = built

    # ── Building height ────────────────────────────────────────────────────────
    height = sample_ghsl_layer("building_height", min_lat, max_lat, min_lon, max_lon)
    if height is not None:
        features["ghsl_building_height_m"] = height

    # ── Population density ────────────────────────────────────────────────────
    pop = sample_ghsl_layer("population", min_lat, max_lat, min_lon, max_lon)
    if pop is not None:
        features["ghsl_population_per_km2"] = pop

    populated = len(features)
    logger.info(
        f"GHSL ({lat}, {lon}): {populated}/3 features populated — "
        f"{list(features.keys())}"
    )

    return features


# ── Main — quick test on 3 coordinates ────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")
    import math
    def generate_bbox(lat, lon, size_m=512):
        half = size_m / 2
        dlat = half / 111320
        dlon = half / (111320 * math.cos(math.radians(abs(lat) or 0.001)))
        return dict(min_lat=lat-dlat, max_lat=lat+dlat, min_lon=lon-dlon, max_lon=lon+dlon)

    # 3 test coordinates covering different GHSL scenarios
    TEST_COORDS = [
        ("dense_urban",          52.5200,  13.4050),   # Berlin — expect high pop + built
        ("industrial",           51.4880,   7.2200),   # Ruhr — expect built, moderate pop
        ("arid",                 26.0000,   3.0000),   # Sahara — expect all None
    ]

    print("\n=== GHSL Extraction Test ===\n")
    for stratum, lat, lon in TEST_COORDS:
        bbox = generate_bbox(lat, lon)
        result = extract_ghsl_features(
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