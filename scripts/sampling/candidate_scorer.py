import sys
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import json
import numpy as np
import rasterio.transform
from scipy.ndimage import label as ndimage_label
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import shape, box as shapely_box

from scripts.extractors.worldcover_extractor import (
    extract_worldcover_features,
    WORLDCOVER_ENERGY_SCORE,
)
from scripts.extractors.ghsl_extractor import extract_ghsl_features
from scripts.extractors.solar_atlas_extractor import extract_solar_features
from scripts.extractors.era5_extractor import extract_era5_features
from scripts.extractors.viirs_extractor import extract_viirs_features
from scripts.osm.osm_query import extract_osm_features, get_osm_energy_elements
from scripts.utils.config_loader import load_config
from scripts.utils.logger import get_logger

logger = get_logger("candidate_scorer")
config = load_config()

SECONDARY_THRESHOLD = config["sampling"]["secondary_class_threshold_pct"]  # 15 (%)
BBOX_SIZE_M = config["sampling"]["bbox_size_m"]  # 512

# ── M3 signal weights ────────────────────────────────────────────────────────
M3_WEIGHTS = {
    "worldcover": 0.30,
    "osm":        0.20,
    "ndvi":       0.15,
    "ntl":        0.15,
    "solar":      0.10,
    "ghsl_era5":  0.10,
}

# Modulation factors per WorldCover class code for each scalar signal
# How strongly does that signal contribute to a cell of that class?
_NDVI_MOD = {
    10: 1.0, 20: 0.7, 30: 0.7, 40: 0.9, 50: 0.2,
    60: 0.1, 70: 0.0, 80: 0.5, 90: 0.9, 95: 1.0, 100: 0.3, 0: 0.0,
}
_NTL_MOD = {
    10: 0.1, 20: 0.1, 30: 0.1, 40: 0.2, 50: 1.0,
    60: 0.3, 70: 0.0, 80: 0.8, 90: 0.2, 95: 0.2, 100: 0.0, 0: 0.0,
}
_SOLAR_MOD = {
    10: 0.3, 20: 0.7, 30: 0.8, 40: 0.7, 50: 0.6,
    60: 1.0, 70: 0.2, 80: 0.1, 90: 0.4, 95: 0.4, 100: 0.1, 0: 0.0,
}
_GHSL_ERA5_MOD = {
    10: 0.3, 20: 0.3, 30: 0.3, 40: 0.4, 50: 1.0,
    60: 0.8, 70: 0.1, 80: 0.2, 90: 0.3, 95: 0.3, 100: 0.1, 0: 0.0,
}


def _safe_normalize(arr):
    """Min-max normalize array to [0, 1]. Returns zero array if flat."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


def _build_modulated_grid(pixel_array, scalar_value, mod_table):
    """
    Build a signal grid by multiplying a scalar value by the modulation
    factor of each WorldCover pixel class.
    """
    grid = np.zeros(pixel_array.shape, dtype=float)
    for code, mod in mod_table.items():
        grid[pixel_array == code] = scalar_value * mod
    return grid


def _build_osm_gaussian(pixel_array, transform, energy_elements, sigma_cells=1.5):
    """
    Place a Gaussian kernel at each OSM energy element and sum contributions.
    Uses scipy.ndimage.gaussian_filter on a sparse impulse grid.
    Returns normalized [0, 1] grid.
    """
    from scipy.ndimage import gaussian_filter

    grid = np.zeros(pixel_array.shape, dtype=float)

    if not energy_elements:
        return grid

    n_rows, n_cols = pixel_array.shape

    for el in energy_elements:
        try:
            row, col = rasterio.transform.rowcol(transform, el["lon"], el["lat"])
            if 0 <= row < n_rows and 0 <= col < n_cols:
                grid[row, col] += el["weight"]
        except Exception:
            continue

    if grid.max() > 0:
        grid = gaussian_filter(grid, sigma=sigma_cells)

    return _safe_normalize(grid)


# ── Core M3 algorithm ────────────────────────────────────────────────────────

def compute_m3(lat, lon, bbox, wc_result=None, osm_elements=None,
               ghsl=None, solar=None, era5=None, viirs=None):
    """
    Run the M3 Feature Density scoring algorithm for a 512×512m bbox.

    Accepts pre-fetched extractor results to avoid duplicate API calls.
    If any result is None, that source is fetched here.

    Returns a dict:
        score_grid      : 2D numpy array, shape ~(51, 51)
        polygon_coords  : [[lat, lon], ...] boundary of highest-scoring zone
        polygon_geom    : Shapely polygon (for spatial ops)
        coverage_pct    : float — polygon area as % of bbox
        std             : float — std of full score grid (spatial variation)
        m3_used         : bool — True = M3 polygon, False = M1 fallback
        low_discrimination : bool — polygon covers >85% of bbox
        worldcover_only_mode : bool — OSM returned 0 elements
        pixel_array     : raw WC pixel array (for visualisation)
        transform       : rasterio transform (for visualisation)
    """
    mn_lat = bbox["min_lat"]
    mx_lat = bbox["max_lat"]
    mn_lon = bbox["min_lon"]
    mx_lon = bbox["max_lon"]

    # ── Step 1: WorldCover pixels ────────────────────────────────────────────
    if wc_result is None:
        wc_result = extract_worldcover_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)

    pixel_array = wc_result.get("_wc_pixel_array")
    transform   = wc_result.get("_wc_transform")

    if pixel_array is None or pixel_array.size == 0:
        logger.warning(f"M3 ({lat}, {lon}): no WorldCover data — returning empty result")
        return _empty_m3_result()

    # ── Step 2: Other extractors (only if not pre-supplied) ──────────────────
    if ghsl is None:
        ghsl = extract_ghsl_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)
    if solar is None:
        solar = extract_solar_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)
    if era5 is None:
        era5 = extract_era5_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)
    if viirs is None:
        viirs = extract_viirs_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)
    if osm_elements is None:
        osm_elements = get_osm_energy_elements(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)

    # ── Step 3: Build Signal A — WorldCover energy score grid ────────────────
    wc_score_grid = np.vectorize(
        lambda c: WORLDCOVER_ENERGY_SCORE.get(int(c), 0.0)
    )(pixel_array)
    sig_A = _safe_normalize(wc_score_grid)

    # ── Step 4: Signal B — OSM Gaussian density ──────────────────────────────
    sig_B = _build_osm_gaussian(pixel_array, transform, osm_elements)
    wco_mode = len(osm_elements) == 0

    # ── Step 5: Signal C — VIIRS NDVI modulated ──────────────────────────────
    ndvi = viirs.get("viirs_ndvi", 0.0) or 0.0
    ndvi_norm = (ndvi + 1.0) / 2.0   # map [-1, 1] → [0, 1]
    sig_C = _safe_normalize(
        _build_modulated_grid(pixel_array, ndvi_norm, _NDVI_MOD)
    )

    # ── Step 6: Signal D — VIIRS NTL modulated ───────────────────────────────
    ntl = viirs.get("viirs_ntl_nw_cm2_sr", 0.0) or 0.0
    ntl_norm = np.log1p(ntl) / np.log1p(2000.0)
    sig_D = _safe_normalize(
        _build_modulated_grid(pixel_array, ntl_norm, _NTL_MOD)
    )

    # ── Step 7: Signal E — Solar PVOUT modulated ─────────────────────────────
    pvout = solar.get("solar_pvout_kwh_kwp_day", 0.0) or 0.0
    pvout_norm = min(pvout / 8.0, 1.0)
    sig_E = _safe_normalize(
        _build_modulated_grid(pixel_array, pvout_norm, _SOLAR_MOD)
    )

    # ── Step 8: Signal F — GHSL + ERA5 combined modulated ────────────────────
    ghsl_norm = min((ghsl.get("ghsl_built_surface_m2", 0.0) or 0.0) / 10000.0, 1.0)
    era5_norm = min((era5.get("era5_ssrd_j_m2_day", 0.0) or 0.0) / 8_000_000.0, 1.0)
    ghsl_era5_scalar = 0.5 * ghsl_norm + 0.5 * era5_norm
    sig_F = _safe_normalize(
        _build_modulated_grid(pixel_array, ghsl_era5_scalar, _GHSL_ERA5_MOD)
    )

    # ── Step 9: Combine ───────────────────────────────────────────────────────
    score_grid = (
        M3_WEIGHTS["worldcover"] * sig_A +
        M3_WEIGHTS["osm"]        * sig_B +
        M3_WEIGHTS["ndvi"]       * sig_C +
        M3_WEIGHTS["ntl"]        * sig_D +
        M3_WEIGHTS["solar"]      * sig_E +
        M3_WEIGHTS["ghsl_era5"]  * sig_F
    )

    std = float(np.std(score_grid))

    # ── Step 10: Select top 30% cells ────────────────────────────────────────
    threshold = float(np.percentile(score_grid, 70))
    mask      = (score_grid >= threshold).astype(np.uint8)

    # ── Step 11: Largest connected component ─────────────────────────────────
    labeled, n_comp = ndimage_label(mask)
    if n_comp == 0:
        return _m1_fallback(pixel_array, wc_result, score_grid, std, wco_mode)

    comp_sizes   = np.bincount(labeled.ravel())
    comp_sizes[0] = 0   # exclude background
    largest_label = int(np.argmax(comp_sizes))
    component_mask = (labeled == largest_label).astype(np.uint8)

    # ── Step 12: Vectorise to polygon ─────────────────────────────────────────
    polygon_geom = None
    polygon_coords = []
    try:
        shapes_gen = rasterio_shapes(component_mask, transform=transform)
        for geom_dict, val in shapes_gen:
            if val == 1:
                polygon_geom = shape(geom_dict)
                break
    except Exception as e:
        logger.warning(f"M3 polygon vectorisation failed: {e}")

    # Clip to bbox
    bbox_geom = shapely_box(mn_lon, mn_lat, mx_lon, mx_lat)
    if polygon_geom is not None:
        polygon_geom = polygon_geom.intersection(bbox_geom)

    if polygon_geom is not None and not polygon_geom.is_empty:
        # Extract exterior coords as [[lat, lon], ...]
        exterior = list(polygon_geom.exterior.coords)
        polygon_coords = [[float(pt[1]), float(pt[0])] for pt in exterior]

    # ── Step 13: Coverage and cascade decision ────────────────────────────────
    bbox_area = (mx_lat - mn_lat) * (mx_lon - mn_lon)
    if polygon_geom is not None and not polygon_geom.is_empty and bbox_area > 0:
        coverage_pct = float(polygon_geom.area / bbox_area * 100)
    else:
        coverage_pct = 0.0

    low_discrimination = coverage_pct > 85.0

    # M1 fallback for genuinely uniform landscapes
    if std < 0.05 or coverage_pct > 85.0:
        return _m1_fallback(pixel_array, wc_result, score_grid, std, wco_mode)

    return {
        "score_grid":             score_grid,
        "polygon_coords":         polygon_coords,
        "polygon_geom":           polygon_geom,
        "coverage_pct":           round(coverage_pct, 2),
        "std":                    round(std, 4),
        "m3_used":                True,
        "low_discrimination":     low_discrimination,
        "worldcover_only_mode":   wco_mode,
        "pixel_array":            pixel_array,
        "transform":              transform,
    }


def _m1_fallback(pixel_array, wc_result, score_grid, std, wco_mode):
    """
    M1 fallback: polygon = bounding box of largest WorldCover class cluster.
    Used when landscape is genuinely uniform (std < 0.05 or coverage > 85%).
    """
    dominant_name = wc_result.get("wc_dominant_class", "unknown")
    logger.info(f"M1 fallback (std={std:.4f}, dominant={dominant_name})")

    # Find largest connected component of dominant class pixels
    dominant_code = None
    classes_json = wc_result.get("wc_classes_json")
    if classes_json:
        try:
            classes = json.loads(classes_json)
            if dominant_name in classes:
                dominant_code = classes[dominant_name]["class_code"]
        except Exception:
            pass

    if dominant_code is not None:
        dom_mask = (pixel_array == dominant_code).astype(np.uint8)
        labeled, n = ndimage_label(dom_mask)
        if n > 0:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            best_label = int(np.argmax(sizes))
            component_mask = (labeled == best_label).astype(np.uint8)
        else:
            component_mask = dom_mask
    else:
        component_mask = np.ones(pixel_array.shape, dtype=np.uint8)

    transform = wc_result.get("_wc_transform")
    polygon_geom = None
    polygon_coords = []

    if transform is not None:
        try:
            shapes_gen = rasterio_shapes(component_mask, transform=transform)
            for geom_dict, val in shapes_gen:
                if val == 1:
                    polygon_geom = shape(geom_dict)
                    break
        except Exception:
            pass

    if polygon_geom is not None and not polygon_geom.is_empty:
        exterior = list(polygon_geom.exterior.coords)
        polygon_coords = [[float(pt[1]), float(pt[0])] for pt in exterior]
        coverage_pct = 100.0  # dominant class always covers most of bbox in M1
    else:
        coverage_pct = 100.0

    return {
        "score_grid":             score_grid,
        "polygon_coords":         polygon_coords,
        "polygon_geom":           polygon_geom,
        "coverage_pct":           round(coverage_pct, 2),
        "std":                    round(std, 4),
        "m3_used":                False,
        "low_discrimination":     True,
        "worldcover_only_mode":   wco_mode,
        "pixel_array":            pixel_array,
        "transform":              wc_result.get("_wc_transform"),
    }


def _empty_m3_result():
    return {
        "score_grid": None, "polygon_coords": [], "polygon_geom": None,
        "coverage_pct": 0.0, "std": 0.0, "m3_used": False,
        "low_discrimination": True, "worldcover_only_mode": True,
        "pixel_array": None, "transform": None,
    }


# ── Per-union confirmation logic ─────────────────────────────────────────────

def _check_union_confirmed(confirmation_mode, primary_pct, secondary_pct,
                           threshold, viirs, ghsl):
    """
    Returns (confirmed: bool, score_A: float) using the union's confirmation mode.

    Modes:
      wc_boundary   — both WC classes must reach threshold % (default)
      ghsl_ntl      — GHSL pop >=800/km² AND VIIRS NTL <3 nW
      ntl_isolation — VIIRS NTL >=20 nW OR WC fallback (bare_sparse >=60% + primary >=5%)
      ghsl_gradient — GHSL pop in transitional range 150–8000/km²
      ghsl_height   — (height >=4m OR built_m2 >=2000) AND pop <300/km²
    """
    if confirmation_mode == "ghsl_ntl":
        # Informal urban: dense population + very low NTL (energy poverty).
        # Primary gate: population density. Fallback: high built_surface + dark area.
        # GHSL population is unreliable for informal settlements (dasymetric models
        # undercount residents in dense unplanned fabric). If pop is low but the bbox
        # has dense built fabric (built_m2 ≥ 2500) AND is dark (NTL < 30 nW), it is
        # almost certainly informal urban, not commercial or industrial.
        pop      = ghsl.get("ghsl_population_per_km2",  0.0) or 0.0
        ntl      = viirs.get("viirs_ntl_nw_cm2_sr",     0.0) or 0.0
        built_m2 = ghsl.get("ghsl_built_surface_m2",    0.0) or 0.0
        # GHSL dasymetric models systematically undercount informal residents;
        # threshold lowered from 800 to 600. Built_surface NTL ceiling raised
        # from 30 to 50 nW to capture Latin-American favelas that have moderate NTL.
        pop_confirmed   = pop >= 600.0
        built_confirmed = built_m2 >= 2500.0 and ntl < 50.0
        confirmed = pop_confirmed or built_confirmed
        pop_score   = min(pop / 2000.0, 1.0)
        built_score = min(built_m2 / 8000.0, 1.0)
        ntl_score   = max(0.0, 1.0 - ntl / 30.0)
        # Use the stronger density signal (pop or built_surface)
        density_score = max(pop_score, built_score)
        return confirmed, round(0.6 * density_score + 0.4 * ntl_score, 4)

    elif confirmation_mode == "ntl_isolation":
        # Isolated industrial facility (oil field, mine) in remote area.
        # WC classifies oil pads as bare_sparse — primary_pct IS bare_sparse fraction.
        # WC fallback: primary (bare_sparse) >=15% means facility is visible in bbox.
        # score_A priority: NTL >=20 → use NTL signal; else WC fallback → use purity.
        ntl = viirs.get("viirs_ntl_nw_cm2_sr", 0.0) or 0.0
        ntl_confirmed = ntl >= 20.0
        wc_fallback = primary_pct >= 15.0   # bare_sparse ≥15% in bbox
        confirmed = ntl_confirmed or wc_fallback
        if ntl >= 20.0:
            score_A = min(ntl / 200.0, 1.0)
        elif wc_fallback:
            # Grade by bare_sparse fraction: 60% = full score (facility core)
            score_A = min(primary_pct / 60.0, 1.0)
        else:
            score_A = 0.0
        return confirmed, round(score_A, 4)

    elif confirmation_mode == "ghsl_gradient":
        pop  = ghsl.get("ghsl_population_per_km2", 0.0) or 0.0
        ntl  = viirs.get("viirs_ntl_nw_cm2_sr",    0.0) or 0.0
        pop_gate   = 150.0 <= pop <= 8000.0
        # Fringe fallback: WC shows visible built_up AND some NTL → confirmed fringe
        # even when GHSL dasymetric model underestimates pop in fast-growing suburbs.
        wc_fringe  = secondary_pct >= 15.0 and ntl >= 2.0
        confirmed  = pop_gate or wc_fringe
        if pop_gate:
            # Trapezoidal score: 0.5 baseline, ramps to 1.0 at 1000/km², back to 0.5 at 8000.
            if pop <= 1000.0:
                score_A = 0.5 + 0.5 * (pop - 150.0) / 850.0
            else:
                score_A = max(0.5, 1.0 - 0.5 * (pop - 1000.0) / 7000.0)
        elif wc_fringe:
            score_A = min(secondary_pct / 60.0, 1.0)
        else:
            score_A = 0.0
        return confirmed, round(score_A, 4)

    elif confirmation_mode == "ghsl_height":
        # Datacenter: GHSL AGBH averages ~1-2m due to parking lot dilution.
        # Thresholds lowered: height ≥1.0m (any building present) and built ≥1500m².
        # Ashburn tested at height=1.34m, built=1790m² — both now pass.
        height   = ghsl.get("ghsl_building_height_m", 0.0) or 0.0
        pop      = ghsl.get("ghsl_population_per_km2", 0.0) or 0.0
        built_m2 = ghsl.get("ghsl_built_surface_m2", 0.0) or 0.0
        confirmed = (height >= 1.0 or built_m2 >= 1500.0) and pop < 300.0
        # height normalised to 5m (GHSL averages over 100m pixels incl. parking;
        # a 1.34m average = real building height ~8-10m diluted by ~80% parking coverage)
        height_score = min(height / 5.0, 1.0)
        built_score  = min(built_m2 / 5000.0, 1.0)
        pop_score    = max(0.0, 1.0 - pop / 300.0) if pop < 300.0 else 0.0
        signal_score = max(height_score, built_score)
        return confirmed, round(0.7 * signal_score + 0.3 * pop_score, 4)

    elif confirmation_mode == "pure_stratum":
        # Single-class dominant location: no secondary class needed.
        # Confirmed when primary WC class covers ≥85% of bbox.
        confirmed = primary_pct >= 85.0
        score_A   = min(primary_pct / 100.0, 1.0)
        return confirmed, round(score_A, 4)

    else:  # "wc_boundary" (default)
        both_present = (primary_pct >= threshold and secondary_pct >= threshold)
        if both_present:
            # Geometric mean: both classes present — reward the balance.
            geo_mean = (primary_pct * secondary_pct) ** 0.5
            score_A  = min(geo_mean / threshold, 1.0)
        else:
            # One class absent — not a boundary; score_A = 0 so Stage-2 always
            # prefers a confirmed candidate over a stronger-on-B/C/D unconfirmed one.
            score_A = 0.0
        return both_present, round(score_A, 4)


# ── Stage 1: Fast WorldCover-only score ──────────────────────────────────────

def score_fast(candidate, primary_wc_code, secondary_wc_code,
               confirmation_mode="wc_boundary"):
    """
    Stage 1 fast score using WorldCover only.
    Returns a float in [0, 1]. Higher = better candidate for full M3.

    wc_boundary   : 0.5*primary + 0.3*secondary + 0.2*std
                    Both classes matter — reward their co-presence.
    ghsl_gradient : 0.4*primary + 0.4*secondary + 0.2*std
                    Transition zone — BOTH classes equally important.
                    Pure-primary candidates (0.4 score) beat pure-secondary (0.4)
                    but tie with mixed-boundary (0.4+) — Stage-2 confirms via pop.
    signal-pattern: 0.7*primary + 0.1*secondary + 0.2*std
                    Secondary used for direction only; confirmation is non-WC.
    """
    lat  = candidate["lat"]
    lon  = candidate["lon"]
    bbox = candidate["bbox"]

    wc = extract_worldcover_features(
        lat, lon,
        bbox["min_lat"], bbox["max_lat"],
        bbox["min_lon"], bbox["max_lon"],
    )
    if not wc:
        return 0.0

    classes_json = wc.get("wc_classes_json", "{}")
    try:
        classes = json.loads(classes_json)
    except Exception:
        classes = {}

    primary_name   = _code_to_name(primary_wc_code)
    secondary_name = _code_to_name(secondary_wc_code)

    primary_pct   = classes.get(primary_name,   {}).get("pct", 0.0) / 100.0
    secondary_pct = classes.get(secondary_name, {}).get("pct", 0.0) / 100.0
    wc_std        = wc.get("wc_std", 0.0)

    if confirmation_mode == "wc_boundary":
        # Geometric mean rewards co-presence of both classes.
        # sqrt(primary × secondary) = 0 if either is absent — correctly rejects
        # pure single-class candidates in favour of true boundary locations.
        boundary_score = (primary_pct * secondary_pct) ** 0.5
        fast_score = 0.8 * boundary_score + 0.2 * wc_std
    elif confirmation_mode == "ghsl_gradient":
        # Geometric mean kills pure-cropland candidates (secondary=0 → score=0).
        # Only fringe candidates with BOTH classes present score well here.
        # Pure cropland always has zero GHSL pop → fails Stage-2 anyway.
        boundary_score = (primary_pct * secondary_pct) ** 0.5
        fast_score = 0.8 * boundary_score + 0.2 * wc_std
    elif confirmation_mode == "pure_stratum":
        # Single dominant class — maximize primary coverage, no secondary needed.
        fast_score = 0.8 * primary_pct + 0.2 * wc_std
    else:
        # ghsl_ntl, ntl_isolation, ghsl_height — secondary is direction-only
        fast_score = 0.7 * primary_pct + 0.1 * secondary_pct + 0.2 * wc_std

    # Store the wc result on the candidate for reuse in Stage 2
    candidate["_wc_result"] = wc
    candidate["_fast_score"] = fast_score
    return fast_score


def _code_to_name(code):
    """Map WorldCover class code → name string."""
    from scripts.extractors.worldcover_extractor import WORLDCOVER_CLASSES
    return WORLDCOVER_CLASSES.get(int(code), f"class_{code}")


# ── Stage 2: Full 5-criterion score ──────────────────────────────────────────

def score_full(candidate, primary_wc_code, secondary_wc_code,
               confirmation_mode="wc_boundary"):
    """
    Stage 2 full 5-criterion score. Runs all 6 extractors + M3.

    Criteria:
    A. Union confirmation  0.35  (logic depends on confirmation_mode)
    B. M3 signal strength  0.25  (std)
    C. Feature diversity   0.20
    D. OSM richness        0.12
    E. WorldCover contrast 0.08

    Returns a dict with total_score + all sub-scores + M3 polygon result.
    """
    lat  = candidate["lat"]
    lon  = candidate["lon"]
    bbox = candidate["bbox"]
    mn_lat, mx_lat = bbox["min_lat"], bbox["max_lat"]
    mn_lon, mx_lon = bbox["min_lon"], bbox["max_lon"]

    # Reuse WC result from Stage 1 if available
    wc_result = candidate.get("_wc_result") or extract_worldcover_features(
        lat, lon, mn_lat, mx_lat, mn_lon, mx_lon
    )

    # Fetch remaining sources
    ghsl = extract_ghsl_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)
    solar = extract_solar_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)
    era5  = extract_era5_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)
    viirs = extract_viirs_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)
    osm   = extract_osm_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)
    osm_elements = get_osm_energy_elements(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)

    # Run M3
    m3 = compute_m3(
        lat, lon, bbox,
        wc_result=wc_result, osm_elements=osm_elements,
        ghsl=ghsl, solar=solar, era5=era5, viirs=viirs,
    )

    # ── Criterion A: Union confirmation ──────────────────────────────────────
    classes_json = wc_result.get("wc_classes_json", "{}")
    try:
        classes = json.loads(classes_json)
    except Exception:
        classes = {}

    primary_name   = _code_to_name(primary_wc_code)
    secondary_name = _code_to_name(secondary_wc_code)
    primary_pct    = classes.get(primary_name,   {}).get("pct", 0.0)
    secondary_pct  = classes.get(secondary_name, {}).get("pct", 0.0)

    both_present, score_A = _check_union_confirmed(
        confirmation_mode, primary_pct, secondary_pct,
        SECONDARY_THRESHOLD, viirs, ghsl,
    )

    # ── Criterion B: M3 signal strength ──────────────────────────────────────
    std    = m3.get("std", 0.0)
    m3_ok  = m3.get("m3_used", False)
    if m3_ok:
        # Full M3 ran: reward spatial heterogeneity (std denominator 0.15 empirical).
        score_B = min(std / 0.15, 1.0)
    elif confirmation_mode == "pure_stratum":
        # Pure strata are intentionally uniform — M3 score_grid is flat → std=0.
        # B reinterpreted as stratum purity: how completely the class dominates.
        score_B = primary_pct / 100.0
    else:
        # M1 fallback on a non-pure landscape: some discrimination exists but weak.
        score_B = min(std / 0.15, 0.4)

    # ── Criterion C: Data source coverage ────────────────────────────────────
    # WC/GHSL/Solar/ERA5 are local rasters — extraction never fails (always 4/6).
    # VIIRS (GEE) and OSM (Overpass) are live APIs that can fail.
    # Count OSM as present only when Overpass actually responded (raw_element_count>0).
    viirs_ok = bool(viirs and len(viirs) > 0)
    osm_ok   = bool(osm and (osm.get("raw_element_count", 0) or 0) > 0)
    score_C  = (4 + int(viirs_ok) + int(osm_ok)) / 6.0

    # ── Criterion D: OSM richness ─────────────────────────────────────────────
    energy_osm_keys = [
        "osm_power_plant", "osm_power_substation", "osm_power_line",
        "osm_power_tower", "osm_generator_source", "osm_plant_source",
        "osm_pipeline", "osm_petroleum_well", "osm_storage_tank", "osm_dam",
    ]
    raw_elements = osm.get("raw_element_count", 0) or 0

    if raw_elements == 0:
        # Overpass API failure — use mode-aware proxy, capped at 0.5 to signal
        # degraded confidence vs live OSM data.
        if confirmation_mode == "ntl_isolation":
            # Oil/gas facilities are classified as bare_sparse in WorldCover.
            # primary_pct (bare_sparse) is the only available industrial signal
            # when GHSL built_surface=0 (facility too sparse for 100m GHSL pixel).
            score_D = min(primary_pct / 50.0, 0.5)
        else:
            # General fallback: GHSL built_surface proxy (port, city, datacenter)
            built_proxy = (ghsl or {}).get("ghsl_built_surface_m2", 0.0) or 0.0
            score_D = min(built_proxy / 5000.0, 0.5)
    else:
        energy_count  = sum(1 for k in energy_osm_keys if osm.get(k))
        energy_score  = min(energy_count / 4.0, 1.0)
        building_count = osm.get("osm_building_count", 0) or 0
        has_landuse   = 1 if osm.get("osm_landuse") else 0
        has_highway   = 1 if osm.get("osm_highway") else 0
        has_railway   = 1 if osm.get("osm_railway") else 0
        context_score = min(
            building_count / 30.0 + has_landuse * 0.25 + has_highway * 0.15 + has_railway * 0.10,
            1.0,
        )
        # context multiplier 0.7 (was 0.5): data-centre/urban locations have rich
        # building+landuse context but rarely have OSM power=substation tagged.
        score_D = min(max(energy_score, 0.7 * context_score), 1.0)

    # ── Weighted total ────────────────────────────────────────────────────────
    # A=0.40  union confirmation  — is this the right stratum type? (primary signal)
    # B=0.25  M3 signal strength  — spatial structure within bbox
    # C=0.15  data source coverage — how many APIs returned live data
    # D=0.20  OSM infrastructure   — energy/context richness (most energy-specific)
    # E removed — wc_contrast was correlated with A for boundaries, 0 for pure strata
    total = (0.40 * score_A + 0.25 * score_B + 0.15 * score_C + 0.20 * score_D)

    return {
        "total_score":      round(float(total), 4),
        "criteria_scores": {
            "A_union_confirmation":  round(score_A, 4),
            "B_m3_signal_strength":  round(score_B, 4),
            "C_feature_diversity":   round(score_C, 4),
            "D_osm_richness":        round(score_D, 4),
        },
        "union_confirmed":  both_present,
        "m3":               m3,
        "features": {
            "wc":   {k: v for k, v in wc_result.items() if not k.startswith("_")},
            "ghsl":  ghsl,
            "solar": solar,
            "era5":  era5,
            "viirs": viirs,
            "osm":   osm,
        },
        "primary_pct":   round(primary_pct, 1),
        "secondary_pct": round(secondary_pct, 1),
    }
