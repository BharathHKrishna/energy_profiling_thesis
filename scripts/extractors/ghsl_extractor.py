import sys
import os
import math
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import rasterio
from rasterio.windows import from_bounds as rasterio_from_bounds, Window
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

        # Remove nodata and invalid values per layer:
        #   building_height: exclude 0 (cell has no buildings) — averaging zeros
        #                    across mostly-empty cells drags the mean to 1-2m
        #   built_surface, population: 0 is valid (genuinely nothing there)
        GHSL_MIN = {"built_surface": 0.0, "building_height": 0.01, "population": 0.0}
        nodata_val = GHSL_NODATA.get(layer_name, -200.0)
        min_val    = GHSL_MIN.get(layer_name, 0.0)
        valid_mask = (data != nodata_val) & (data >= min_val)
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


# ── Whole-tile, overlap-weighted floor area ────────────────────────────────────
# Fixed 2026-08-14: the demand formula's own floor_area input needs a real total
# built-floor-area estimate for the whole bbox, not the mean-of-cells figures
# above (ghsl_built_surface_m2/ghsl_building_height_m stay mean-based for the
# general feature record -- see extract_ghsl_features()). An earlier version of
# the demand formula used mean built-surface x mean height directly, which
# systematically underestimates any bbox that isn't perfectly uniform, since it
# throws away the real per-cell correlation between where a tile is built AND
# how tall it is there. This reads built_surface and building_height together,
# cell by cell, and sums (not averages) each cell's own contribution, weighted
# by how much of that 100m cell geometrically overlaps the 512m bbox -- 512/100
# = 5.12, so the bbox never lines up exactly with the underlying grid, and an
# edge cell should count only its actual overlapping fraction, not the whole
# cell nor nothing.

def compute_floor_area(min_lat, max_lat, min_lon, max_lon):
    """
    floor_area = sum_i overlap_fraction_i * built_surface_i * max(height_i, 3) / 3

    Built surface is additive across cells (real m^2 of built area in that
    cell), height is read per cell rather than one shared bbox-wide average (a
    tall building on a small footprint and a sprawling low-rise development
    are not blurred into the same number), and a cell with measured built area
    but a missing or implausibly low height reading is floored at one storey,
    3 metres, rather than losing its contribution to the total.

    Returns float (m^2) or None if the bbox has no valid built_surface data at
    all (open water, raster extent miss, etc.) -- never a fabricated zero.
    """
    built_path  = GHSL_PATHS.get("built_surface")
    height_path = GHSL_PATHS.get("building_height")
    if not built_path or not height_path:
        return None
    if not (os.path.exists(built_path) and os.path.exists(height_path)):
        logger.warning("GHSL floor_area: raster file(s) not found")
        return None

    try:
        xs, ys = TRANSFORMER_TO_MOLL.transform(
            [min_lon, max_lon, min_lon, max_lon],
            [min_lat, min_lat, max_lat, max_lat]
        )
        moll_min_x, moll_max_x = min(xs), max(xs)
        moll_min_y, moll_max_y = min(ys), max(ys)

        with rasterio.open(built_path) as bsrc, rasterio.open(height_path) as hsrc:
            raw_window = rasterio_from_bounds(
                moll_min_x, moll_min_y, moll_max_x, moll_max_y, bsrc.transform
            )
            # Expand to the integer pixel range that the fractional window touches
            # at all, so a cell only partially inside the bbox is still read (its
            # overlap_fraction, computed below, is what actually discounts it).
            col_off = math.floor(raw_window.col_off)
            row_off = math.floor(raw_window.row_off)
            col_end = math.ceil(raw_window.col_off + raw_window.width)
            row_end = math.ceil(raw_window.row_off + raw_window.height)
            window = Window(col_off=col_off, row_off=row_off,
                            width=col_end - col_off, height=row_end - row_off)

            if (window.width <= 0 or window.height <= 0 or
                    window.col_off < 0 or window.row_off < 0 or
                    window.col_off >= bsrc.width or window.row_off >= bsrc.height):
                logger.warning(f"GHSL floor_area: bbox outside raster extent for window {window}")
                return None

            built_data  = bsrc.read(1, window=window)
            height_data = hsrc.read(1, window=window)

        if built_data.size == 0:
            return None

        # overlap_fraction per cell: intersection of each integer pixel cell
        # [c, c+1) x [r, r+1) (in window-local coordinates) with the real,
        # fractional bbox window -- 1.0 for a cell fully inside, 0 for a cell
        # entirely outside (shouldn't occur given the expand-to-integer step
        # above, but clamped anyway), a fraction for an edge cell.
        local_col_off = raw_window.col_off - col_off
        local_row_off = raw_window.row_off - row_off
        bbox_c0, bbox_c1 = local_col_off, local_col_off + raw_window.width
        bbox_r0, bbox_r1 = local_row_off, local_row_off + raw_window.height

        n_rows, n_cols = built_data.shape
        col_idx = np.arange(n_cols)
        row_idx = np.arange(n_rows)
        col_overlap = np.clip(np.minimum(col_idx + 1, bbox_c1) - np.maximum(col_idx, bbox_c0), 0, 1)
        row_overlap = np.clip(np.minimum(row_idx + 1, bbox_r1) - np.maximum(row_idx, bbox_r0), 0, 1)
        overlap_fraction = row_overlap[:, None] * col_overlap[None, :]

        built_nodata = GHSL_NODATA.get("built_surface", -200.0)
        height_nodata = GHSL_NODATA.get("building_height", -200.0)
        built_valid = (built_data != built_nodata) & (built_data >= 0.0)
        built_vals = np.where(built_valid, built_data, 0.0)

        # Height: use the real reading where valid, otherwise floor to 3m (one
        # storey) so a cell with real built area but a missing/implausible
        # height reading still contributes rather than being dropped entirely.
        height_valid = (height_data != height_nodata) & (height_data > 0.0)
        height_vals = np.where(height_valid, height_data, 3.0)
        height_vals = np.maximum(height_vals, 3.0)

        cell_floor_area = overlap_fraction * built_vals * (height_vals / 3.0)
        total = float(np.sum(cell_floor_area))

        if not built_valid.any():
            logger.info("GHSL floor_area: no valid built_surface cells -- returning None")
            return None

        logger.info(f"GHSL floor_area: {n_rows * n_cols} cells, "
                   f"{int(built_valid.sum())} with valid built_surface, total = {total:.1f} m^2")
        return round(total, 1)

    except Exception as e:
        logger.warning(f"GHSL floor_area: read error -- {e}")
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