import sys
import os
import math
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import rasterio
from rasterio.windows import Window
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box

# GHSL R2023A grid: 100 m cells, so 10,000 m2 of ground per cell.
CELL_AREA_M2 = 100.0 * 100.0
# Floor-to-floor height used to turn building volume into floor area.
STOREY_HEIGHT_M = 3.0
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

# ── True tile footprint in Mollweide ──────────────────────────────────────────
# Fixed 2026-09-09, in two steps.
#
# Originally every GHSL read projected the bbox's four corners into Mollweide
# and took min/max. Mollweide is equal-area but not conformal: a lat/lon
# rectangle becomes a sheared parallelogram, so the axis-aligned box around it
# is larger than the tile. Measured: 1.00x at the prime meridian, 1.12x at
# Berlin, 1.47x at Hong Kong, 1.89x in south-eastern Australia. Every overlap
# fraction was computed against that oversized rectangle.
#
# The first fix built a true 512 x 512 m square on the projected centre. That
# corrected the area but kept the wrong orientation, and checked against an
# independent grid integration it was still off by up to 16 % per tile (Nairobi
# +15.9 %), because a square is not the shape a lat/lon rectangle projects to.
#
# What is here now intersects each raster cell with the tile's real projected
# footprint, edges densified so the projection's curvature survives. Verified
# against the same independent integration: agreement within 0.3 % at Hong Kong,
# Berlin, Tokyo, Santiago, Nairobi and rural Australia. Mollweide being
# equal-area is what makes this exact rather than approximate: the polygon's
# area in projected metres is the tile's real ground area.

def tile_polygon(min_lat, max_lat, min_lon, max_lon, densify=40):
    """The bbox's real footprint in Mollweide metres, as a shapely polygon."""
    las = np.linspace(min_lat, max_lat, densify)
    los = np.linspace(min_lon, max_lon, densify)
    ring = ([(x, min_lat) for x in los] + [(max_lon, y) for y in las] +
            [(x, max_lat) for x in los[::-1]] + [(min_lon, y) for y in las[::-1]])
    xs, ys = TRANSFORMER_TO_MOLL.transform([p[0] for p in ring], [p[1] for p in ring])
    return ShapelyPolygon(zip(xs, ys))


def overlap_grid(src, poly):
    """(window, fractions) — the fraction of each raster cell inside poly.

    Returns (None, None) if the polygon falls outside the raster. A cell fully
    inside gets 1.0, a cell clipped by the tile edge gets its real share, and a
    cell outside gets 0, so a caller can sum or average without knowing anything
    about where the tile boundary fell.
    """
    x0, y0, x1, y1 = poly.bounds
    inv = ~src.transform
    c0f, r1f = inv * (x0, y0)
    c1f, r0f = inv * (x1, y1)
    c0, c1 = int(math.floor(c0f)), int(math.ceil(c1f))
    r0, r1 = int(math.floor(r0f)), int(math.ceil(r1f))

    if c1 <= c0 or r1 <= r0 or c0 < 0 or r0 < 0 or c0 >= src.width or r0 >= src.height:
        return None, None

    window = Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)
    cell_area = abs(src.transform.a * src.transform.e)
    frac = np.zeros((r1 - r0, c1 - c0), dtype=float)
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            cx0, cy0 = src.transform * (c0 + j, r0 + i + 1)
            cx1, cy1 = src.transform * (c0 + j + 1, r0 + i)
            cell = shapely_box(min(cx0, cx1), min(cy0, cy1), max(cx0, cx1), max(cy0, cy1))
            inter = poly.intersection(cell).area
            if inter > 0:
                frac[i, j] = inter / cell_area
    return window, frac


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
        poly = tile_polygon(min_lat, max_lat, min_lon, max_lon)

        with rasterio.open(raster_path) as src:
            window, frac = overlap_grid(src, poly)
            if window is None:
                logger.warning(f"GHSL {layer_name}: bbox outside raster extent")
                return None
            data = src.read(1, window=window)

        if data.size == 0 or data.shape != frac.shape:
            return None

        # Remove nodata and invalid values per layer:
        #   building_height: exclude 0 (cell has no buildings) — averaging zeros
        #                    across mostly-empty cells drags the mean to 1-2m
        #   built_surface, population: 0 is valid (genuinely nothing there)
        GHSL_MIN = {"built_surface": 0.0, "building_height": 0.01, "population": 0.0}
        nodata_val = GHSL_NODATA.get(layer_name, -200.0)
        min_val    = GHSL_MIN.get(layer_name, 0.0)
        valid_mask = (data != nodata_val) & (data >= min_val) & (frac > 0)
        weights = np.where(valid_mask, frac, 0.0)
        total_w = float(weights.sum())

        if total_w <= 0:
            logger.info(
                f"GHSL {layer_name}: all pixels are nodata — returning None"
            )
            return None

        # Area-weighted mean over the tile's real footprint: a cell only partly
        # inside counts only for the part inside, instead of counting in full or
        # not at all.
        result = float((data.astype(float) * weights).sum() / total_w)
        logger.info(
            f"GHSL {layer_name}: {int((valid_mask).sum())} valid cells, "
            f"weighted mean = {result:.2f}"
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
        poly = tile_polygon(min_lat, max_lat, min_lon, max_lon)

        with rasterio.open(built_path) as bsrc, rasterio.open(height_path) as hsrc:
            window, overlap_fraction = overlap_grid(bsrc, poly)
            if window is None:
                logger.warning("GHSL floor_area: bbox outside raster extent")
                return None
            built_data  = bsrc.read(1, window=window)
            height_data = hsrc.read(1, window=window)

        if built_data.size == 0 or built_data.shape != overlap_fraction.shape:
            return None

        n_rows, n_cols = built_data.shape

        built_nodata = GHSL_NODATA.get("built_surface", -200.0)
        height_nodata = GHSL_NODATA.get("building_height", -200.0)
        built_valid = (built_data != built_nodata) & (built_data >= 0.0)
        built_vals = np.where(built_valid, built_data, 0.0)

        # The height raster is GHS-BUILT-H AGBH, the average GROSS building
        # height, which the GHSL data package (R2023A, section 2.2.1) defines as
        # building volume divided by the WHOLE cell area:
        #
        #     AGBH = BUVOL / S          so   BUVOL = S * AGBH = 10,000 * AGBH
        #     ANBH = BUVOL / BUSURF     (the net height, over built area only)
        #     AGBH / ANBH = BUSURF / S  = the built fraction
        #
        # AGBH therefore already carries the built fraction inside it. An earlier
        # version of this function computed built_surface * AGBH / 3, which
        # applies that fraction a second time and understated floor area roughly
        # twofold. Verified against real fabric: the old form put Haussmann-era
        # central Paris at 8.3 m and FAR 0.9, where those blocks are six to seven
        # storeys at about 18 m and the arrondissement's FAR is near 3.
        #
        # Volume is the extensive quantity, so volume is what gets summed.
        height_valid = (height_data != height_nodata) & (height_data > 0.0)
        cell_volume = np.where(height_valid, height_data * CELL_AREA_M2, 0.0)

        # A cell with real built area but no usable height reading is floored at
        # one storey rather than dropped, the same intent as the old clamp but
        # applied to volume instead of to a gross height it did not fit.
        cell_volume = np.where((cell_volume <= 0.0) & built_valid,
                               built_vals * STOREY_HEIGHT_M, cell_volume)

        total_volume = float(np.sum(overlap_fraction * cell_volume))
        total = total_volume / STOREY_HEIGHT_M

        if not built_valid.any():
            logger.info("GHSL floor_area: no valid built_surface cells -- returning None")
            return None

        logger.info(f"GHSL floor_area: {n_rows * n_cols} cells, "
                   f"{int(built_valid.sum())} with valid built_surface, total = {total:.1f} m^2")
        return round(total, 1)

    except Exception as e:
        logger.warning(f"GHSL floor_area: read error -- {e}")
        return None



def compute_building_height(min_lat, max_lat, min_lon, max_lon):
    """Mean NET building height over the tile, in metres.

    The raster holds AGBH, the gross height (volume per whole cell), so reading
    it directly reports a value diluted by however much of the cell is unbuilt:
    Haussmann-era Paris reads 8.3 m that way, against a real 18.7 m. GHSL's own
    relation ANBH = BUVOL / BUSURF recovers the real height, computed here over
    the whole tile rather than per cell so a mostly-empty cell cannot swing it.

    Returns float or None when the tile has no built surface to divide by.
    """
    built_path = GHSL_PATHS.get("built_surface")
    height_path = GHSL_PATHS.get("building_height")
    if not (built_path and height_path
            and os.path.exists(built_path) and os.path.exists(height_path)):
        return None
    try:
        poly = tile_polygon(min_lat, max_lat, min_lon, max_lon)
        with rasterio.open(built_path) as bsrc, rasterio.open(height_path) as hsrc:
            window, frac = overlap_grid(bsrc, poly)
            if window is None:
                return None
            b = bsrc.read(1, window=window).astype(float)
            h = hsrc.read(1, window=window).astype(float)
        if b.shape != frac.shape:
            return None
        bn = GHSL_NODATA.get("built_surface", -200.0)
        hn = GHSL_NODATA.get("building_height", -200.0)
        b = np.where((b != bn) & (b >= 0.0), b, 0.0)
        h = np.where((h != hn) & (h > 0.0), h, 0.0)
        built_total = float((frac * b).sum())
        volume_total = float((frac * h * CELL_AREA_M2).sum())
        if built_total <= 0 or volume_total <= 0:
            return None
        return round(volume_total / built_total, 2)
    except Exception as e:
        logger.warning(f"GHSL building height: read error -- {e}")
        return None


# ── Main extraction function ───────────────────────────────────────────────────

def extract_ghsl_features(lat, lon, min_lat, max_lat, min_lon, max_lon):
    """
    Extract GHSL features for a 512×512m bounding box.

    Returns a flat dict with keys:
        ghsl_built_surface_m2    — mean built-up surface area (m²) per 100m cell
        ghsl_building_height_m   — mean building height (metres)
        ghsl_population_per_km2      — mean of GHS-POP cells (persons per 100m
                                       cell, despite the name; kept for the
                                       class gates, which read it that way)
        ghsl_population_bbox_total   — persons inside the bbox, overlap-weighted

    compute_bbox_population() also returns a per-km2 density, but that is not
    stored as a feature: the dataset's unit is the 512 m tile, and nothing reads
    a per-km2 figure. It exists only to convert published thresholds that are
    quoted per km2 (GHS-SMOD) into people-per-tile, which feature_bands.py does
    once at module level rather than per coordinate.

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
    height = compute_building_height(min_lat, max_lat, min_lon, max_lon)
    if height is not None:
        features["ghsl_building_height_m"] = height

    # ── Population ────────────────────────────────────────────────────────────
    # Mean-of-cells, kept because the class gates read it as a per-cell count
    # (see Section 3.4). Its name says per km2; the value is persons per 100m
    # cell. compute_bbox_population below is the figure that actually describes
    # the tile.
    pop = sample_ghsl_layer("population", min_lat, max_lat, min_lon, max_lon)
    if pop is not None:
        features["ghsl_population_per_km2"] = pop

    pop_total, pop_density = compute_bbox_population(min_lat, max_lat, min_lon, max_lon)
    if pop_total is not None:
        features["ghsl_population_bbox_total"] = pop_total

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

# ── Whole-tile, overlap-weighted population ───────────────────────────────────
# Added 2026-09-09. GHS-POP cells hold an absolute person count per cell, not a
# density, so the meaningful figure for a 512m bbox is the SUM of the people in
# it, not the mean of the cells it touches. The existing mean-of-cells reading
# (ghsl_population_per_km2, kept for the general feature record and for the
# class gates, which read it correctly as a per-cell count) answers a different
# question and carries a unit name it does not actually hold.
#
# Same geometry as compute_floor_area above: 512/100 = 5.12, so the bbox never
# lines up with the grid and an edge cell must count only its overlapping
# fraction. Population is assumed uniform within a cell, which is the standard
# assumption for areal interpolation of a gridded count.

def compute_bbox_population(min_lat, max_lat, min_lon, max_lon):
    """
    population = sum_i overlap_fraction_i * population_i

    Returns (total_persons, density_per_km2) or (None, None) when the bbox has
    no valid GHS-POP data at all. Density is derived from the same total and the
    bbox's real area, so the two can never disagree.
    """
    pop_path = GHSL_PATHS.get("population")
    if not pop_path or not os.path.exists(pop_path):
        logger.warning("GHSL bbox population: raster not found")
        return None, None

    try:
        poly = tile_polygon(min_lat, max_lat, min_lon, max_lon)

        with rasterio.open(pop_path) as src:
            window, overlap_fraction = overlap_grid(src, poly)
            if window is None:
                logger.warning("GHSL bbox population: bbox outside raster extent")
                return None, None
            data = src.read(1, window=window)

        if data.size == 0 or data.shape != overlap_fraction.shape:
            return None, None

        n_rows, n_cols = data.shape

        nodata_val = GHSL_NODATA.get("population", -200.0)
        valid = (data != nodata_val) & (data >= 0.0)
        if not valid.any():
            logger.info("GHSL bbox population: no valid cells -- returning None")
            return None, None

        vals = np.where(valid, data, 0.0)
        total = float(np.sum(overlap_fraction * vals))

        # Mollweide is equal-area, so the polygon's own area in projected
        # metres is the tile's real ground area (0.262144 km^2 for a 512 m tile).
        area_km2 = poly.area / 1e6
        density = total / area_km2 if area_km2 > 0 else None

        logger.info(f"GHSL bbox population: {n_rows * n_cols} cells, "
                    f"total = {total:.1f} persons over {area_km2:.4f} km^2")
        return round(total, 1), (round(density, 1) if density is not None else None)

    except Exception as e:
        logger.warning(f"GHSL bbox population: read error -- {e}")
        return None, None
