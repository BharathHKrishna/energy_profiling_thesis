import sys
import os
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import rasterio
from rasterio.windows import from_bounds as rasterio_from_bounds
from rasterio.features import shapes as rasterio_shapes
from scipy.ndimage import label as ndimage_label
from shapely.geometry import shape, box

from scripts.utils.logger import get_logger

logger = get_logger("worldcover_extractor")

# ── WorldCover class definitions ───────────────────────────────────────────────
# Source: ESA WorldCover 2021 v200 product manual
WORLDCOVER_CLASSES = {
    10:  "tree_cover",
    20:  "shrubland",
    30:  "grassland",
    40:  "cropland",
    50:  "built_up",
    60:  "bare_sparse",
    70:  "snow_ice",
    80:  "water",
    90:  "wetland",
    95:  "mangrove",
    100: "moss_lichen",
}

# Energy relevance score per class — used in M3 multi-source scoring
# Higher = more energy-relevant
# Source: original scoring logic for this pipeline
WORLDCOVER_ENERGY_SCORE = {
    10:  1.5,   # tree_cover  — biomass, hydro catchment
    20:  0.5,   # shrubland   — low relevance
    30:  0.5,   # grassland   — wind potential
    40:  1.0,   # cropland    — biofuel, agrivoltaics
    50:  3.0,   # built_up    — highest — demand + infrastructure
    60:  0.0,   # bare_sparse — desert, lowest relevance
    70:  0.0,   # snow_ice    — alpine, no direct relevance
    80:  1.5,   # water       — hydro potential
    90:  1.0,   # wetland     — moderate
    95:  1.0,   # mangrove    — coastal energy
    100: 0.0,   # moss_lichen — no relevance
    0:   0.0,   # nodata
}

# ── WorldCover tile URL builder ────────────────────────────────────────────────
def get_worldcover_tile_url(lat, lon):
    """
    Build HTTPS URL for ESA WorldCover 2021 v200 tile containing (lat, lon).
    Tiles are 3°×3°, named by SW corner.
    """
    lat_base = int(np.floor(lat / 3) * 3)
    lon_base = int(np.floor(lon / 3) * 3)

    lat_str = f"N{abs(lat_base):02d}" if lat_base >= 0 else f"S{abs(lat_base):02d}"
    lon_str = f"E{abs(lon_base):03d}" if lon_base >= 0 else f"W{abs(lon_base):03d}"

    filename = f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map.tif"
    url = (
        f"https://esa-worldcover.s3.eu-central-1.amazonaws.com"
        f"/v200/2021/map/{filename}"
    )
    return url, filename


# ── Pixel fetch ────────────────────────────────────────────────────────────────
def fetch_worldcover_pixels(min_lat, max_lat, min_lon, max_lon, lat, lon):
    """
    Fetch WorldCover pixel array for a 512×512m bounding box.
    Uses /vsicurl/ to read only the bbox window from the remote GeoTIFF.

    Returns:
        pixels    : 2D numpy array of class codes
        transform : rasterio affine transform
        success   : bool
    """
    url, _ = get_worldcover_tile_url(lat, lon)
    vsicurl_path = f"/vsicurl/{url}"

    try:
        with rasterio.open(vsicurl_path) as src:
            window    = rasterio_from_bounds(
                min_lon, min_lat, max_lon, max_lat, src.transform
            )
            pixels    = src.read(1, window=window)
            transform = src.window_transform(window)

        if pixels.size == 0:
            return None, None, False

        logger.info(
            f"WorldCover: fetched {pixels.shape} pixels for ({lat}, {lon}) "
            f"— unique classes: {np.unique(pixels).tolist()}"
        )
        return pixels, transform, True

    except Exception as e:
        logger.warning(f"WorldCover fetch failed for ({lat}, {lon}): {e}")
        return None, None, False


# ── Main extraction function ───────────────────────────────────────────────────
def extract_worldcover_features(lat, lon, min_lat, max_lat, min_lon, max_lon):
    """
    Extract comprehensive WorldCover features for a 512×512m bounding box.

    Returns a flat dict with keys:
        wc_dominant_class        — name of most frequent class
        wc_dominant_pct          — % of bbox covered by dominant class
        wc_secondary_class       — name of second most frequent class (if ≥5%)
        wc_secondary_pct         — % of bbox covered by secondary class
        wc_class_count           — number of distinct classes present
        wc_energy_score_mean     — mean energy relevance score across all pixels
        wc_std                   — standard deviation of energy scores (spatial variation)
        wc_classes_json          — JSON string of all classes with pixel counts and pcts
        wc_pixel_array           — raw pixel array (not in CSV/JSON — for M3 use only)
        wc_transform             — rasterio transform (not in CSV/JSON — for M3 use only)

    Any feature without valid data is absent from the returned dict.
    wc_pixel_array and wc_transform are internal — stripped before CSV/JSON output.
    """
    logger.info(f"Extracting WorldCover features for ({lat}, {lon})")

    pixels, transform, ok = fetch_worldcover_pixels(
        min_lat, max_lat, min_lon, max_lon, lat, lon
    )

    if not ok or pixels is None or pixels.size == 0:
        logger.warning(f"WorldCover: no data for ({lat}, {lon})")
        return {}

    # ── Strip nodata pixels (class 0) ──────────────────────────────────────────
    valid_pixels = pixels[pixels > 0]

    if len(valid_pixels) == 0:
        logger.warning(f"WorldCover: all pixels are nodata for ({lat}, {lon})")
        return {}

    total_valid = len(valid_pixels)

    # ── Class counts and percentages ──────────────────────────────────────────
    unique_classes, counts = np.unique(valid_pixels, return_counts=True)
    # Sort by count descending
    sort_idx      = np.argsort(-counts)
    unique_classes = unique_classes[sort_idx]
    counts         = counts[sort_idx]

    class_info = {}
    for cls, cnt in zip(unique_classes, counts):
        pct = round(float(cnt) / total_valid * 100, 1)
        name = WORLDCOVER_CLASSES.get(int(cls), f"class_{cls}")
        class_info[name] = {
            "class_code": int(cls),
            "pixels":     int(cnt),
            "pct":        pct,
        }

    # ── Dominant and secondary class ──────────────────────────────────────────
    dominant_name = list(class_info.keys())[0]
    dominant_pct  = class_info[dominant_name]["pct"]

    secondary_name = None
    secondary_pct  = None
    if len(unique_classes) > 1:
        secondary_name = list(class_info.keys())[1]
        secondary_pct  = class_info[secondary_name]["pct"]
        # Only report secondary if it covers at least 5% of bbox
        if secondary_pct < 5.0:
            secondary_name = None
            secondary_pct  = None

    # ── Energy score grid ─────────────────────────────────────────────────────
    energy_grid = np.vectorize(
        lambda c: WORLDCOVER_ENERGY_SCORE.get(int(c), 0.0)
    )(pixels)

    energy_mean = round(float(np.mean(energy_grid[pixels > 0])), 3)
    energy_std  = round(float(np.std(energy_grid)), 3)

    # ── Assemble features dict ────────────────────────────────────────────────
    import json
    features = {
        "wc_dominant_class":    dominant_name,
        "wc_dominant_pct":      dominant_pct,
        "wc_class_count":       len(unique_classes),
        "wc_energy_score_mean": energy_mean,
        "wc_std":               energy_std,
        "wc_classes_json":      json.dumps(class_info),
        # Internal — for M3, not for CSV/JSON output
        "_wc_pixel_array":      pixels,
        "_wc_transform":        transform,
    }

    if secondary_name is not None:
        features["wc_secondary_class"] = secondary_name
        features["wc_secondary_pct"]   = secondary_pct

    populated = len([k for k in features if not k.startswith("_")])
    logger.info(
        f"WorldCover ({lat}, {lon}): {populated} features — "
        f"dominant={dominant_name} ({dominant_pct}%), "
        f"classes={len(unique_classes)}, std={energy_std}"
    )

    return features


def strip_internal_features(features):
    """
    Remove internal keys (prefixed with _) before writing to CSV or JSON.
    Call this before any output operation.
    """
    return {k: v for k, v in features.items() if not k.startswith("_")}


# ── Main — quick test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math
    def generate_bbox(lat, lon, size_m=512):
        half = size_m / 2
        dlat = half / 111320
        dlon = half / (111320 * math.cos(math.radians(abs(lat) or 0.001)))
        return dict(min_lat=lat-dlat, max_lat=lat+dlat, min_lon=lon-dlon, max_lon=lon+dlon)

    TEST_COORDS = [
        ("dense_urban",    52.5200,  13.4050),   # Berlin — expect built_up dominant
        ("forest",         -4.0000, -60.0000),   # Amazon — expect tree_cover dominant
        ("arid",           26.0000,   3.0000),   # Sahara — expect bare_sparse dominant
        ("Rotterdam_union", 51.9494,   4.1406),  # Rotterdam — expect built_up + grassland
        ("agricultural",   47.5000,   1.8000),   # France — expect tree_cover or cropland
    ]

    print("\n=== WorldCover Extraction Test ===\n")
    for stratum, lat, lon in TEST_COORDS:
        bbox   = generate_bbox(lat, lon)
        result = extract_worldcover_features(
            lat, lon,
            bbox["min_lat"], bbox["max_lat"],
            bbox["min_lon"], bbox["max_lon"]
        )
        output = strip_internal_features(result)
        print(f"[{stratum}] ({lat}, {lon})")
        if output:
            for k, v in output.items():
                if k != "wc_classes_json":
                    print(f"  {k}: {v}")
            # Print class breakdown separately
            if "wc_classes_json" in output:
                import json
                classes = json.loads(output["wc_classes_json"])
                print("  Class breakdown:")
                for cls_name, info in classes.items():
                    print(f"    {cls_name}: {info['pct']}% ({info['pixels']} px)")
        else:
            print("  No features returned")
        print()