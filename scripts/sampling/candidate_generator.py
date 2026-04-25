import sys
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import numpy as np
import rasterio.transform

from scripts.extractors.worldcover_extractor import extract_worldcover_features
from scripts.utils.config_loader import load_config
from scripts.utils.logger import get_logger

logger = get_logger("candidate_generator")
config = load_config()

SEARCH_AREA_M  = config["sampling"]["search_area_m"]   # 1024
BBOX_SIZE_M    = config["sampling"]["bbox_size_m"]     # 512
N_CANDIDATES   = config["sampling"]["candidates_per_anchor"]  # 10


def generate_bbox(lat, lon, size_m=512):
    """512×512m bounding box centred on (lat, lon)."""
    delta_lat = (size_m / 2) / 111320.0
    delta_lon = (size_m / 2) / (111320.0 * np.cos(np.radians(lat)))
    return {
        "min_lat": lat - delta_lat,
        "max_lat": lat + delta_lat,
        "min_lon": lon - delta_lon,
        "max_lon": lon + delta_lon,
    }


def _offset_coord(lat, lon, dx_m, dy_m):
    """Offset a lat/lon by dx_m east and dy_m north."""
    dlat = dy_m / 111320.0
    dlon = dx_m / (111320.0 * np.cos(np.radians(lat)))
    return lat + dlat, lon + dlon


def _find_class_direction(pixel_array, wc_code):
    """
    Find the direction (unit vector in metres) from the grid centre toward
    the centroid of all pixels belonging to wc_code.

    Returns (dx_norm, dy_norm) as a unit vector, or None if class absent.
    """
    rows, cols = np.where(pixel_array == wc_code)
    if len(rows) == 0:
        return None

    n_rows, n_cols = pixel_array.shape
    centre_row = n_rows / 2.0
    centre_col = n_cols / 2.0

    # Row offset from centre (positive = south, negative = north)
    row_off = np.mean(rows) - centre_row
    col_off = np.mean(cols) - centre_col

    # Convert to metres (each cell = 10 m at WorldCover native resolution)
    # Row increases downward → positive row_off = south → dy negative
    cell_m   = SEARCH_AREA_M / n_rows   # metres per row
    dy_m     = -row_off * cell_m        # north = positive
    dx_m     = col_off * cell_m         # east  = positive

    magnitude = np.sqrt(dx_m ** 2 + dy_m ** 2)
    if magnitude < 1e-6:
        return None
    return dx_m / magnitude, dy_m / magnitude


def _fallback_directions():
    """Eight evenly-spaced cardinal + intercardinal directions as unit vectors."""
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    return [(np.cos(a), np.sin(a)) for a in angles]


def generate_candidates(anchor_lat, anchor_lon, primary_wc_code, secondary_wc_code,
                         rng=None):
    """
    Generate N_CANDIDATES bboxes around (anchor_lat, anchor_lon):
      - 30% toward primary WorldCover class zone (min 1)
      - 30% toward secondary WorldCover class zone (min 1)
      - remainder toward the boundary between them

    N_CANDIDATES is read from config (candidates_per_anchor).
    Example: N=10 → 3+3+4; N=5 → 1+1+3.

    Each candidate is a 512×512m bbox whose centre is offset 100-250 m from
    the anchor (boundary candidates: 50-150 m).

    Null contract: always returns exactly N_CANDIDATES dicts
        {"lat": float, "lon": float, "bbox": dict, "direction": str}
    Uses fallback directions if WorldCover unavailable or class absent.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Fetch WorldCover for the 1024 m search area around the anchor
    sbbox = generate_bbox(anchor_lat, anchor_lon, size_m=SEARCH_AREA_M)
    wc = extract_worldcover_features(
        anchor_lat, anchor_lon,
        sbbox["min_lat"], sbbox["max_lat"],
        sbbox["min_lon"], sbbox["max_lon"],
    )
    pixel_array = wc.get("_wc_pixel_array")

    pdir = None
    sdir = None

    if pixel_array is not None and pixel_array.size > 0:
        pdir = _find_class_direction(pixel_array, primary_wc_code)
        sdir = _find_class_direction(pixel_array, secondary_wc_code)
        logger.info(
            f"Anchor ({anchor_lat:.4f}, {anchor_lon:.4f}): "
            f"primary direction={pdir}, secondary direction={sdir}"
        )

    # Fallback: evenly-spaced directions when WC unavailable or class absent
    if pdir is None or sdir is None:
        logger.warning(
            f"Anchor ({anchor_lat:.4f}, {anchor_lon:.4f}): "
            "WC class absent — using fallback directions"
        )
        fb = _fallback_directions()
        if pdir is None:
            pdir = fb[0]
        if sdir is None:
            sdir = fb[4]

    # Boundary direction = towards the midpoint between the two class centroids
    bdx = pdir[0] + sdir[0]
    bdy = pdir[1] + sdir[1]
    bmag = np.sqrt(bdx ** 2 + bdy ** 2)
    if bmag < 1e-6:
        bdir = (pdir[0], -pdir[1])   # perpendicular as last resort
    else:
        bdir = (bdx / bmag, bdy / bmag)

    n_primary   = max(1, int(N_CANDIDATES * 0.3))
    n_secondary = max(1, int(N_CANDIDATES * 0.3))
    n_boundary  = N_CANDIDATES - n_primary - n_secondary

    candidates = []

    for _ in range(n_primary):
        offset_m = rng.uniform(100, 250)
        clat, clon = _offset_coord(anchor_lat, anchor_lon,
                                   pdir[0] * offset_m, pdir[1] * offset_m)
        candidates.append({
            "lat": clat, "lon": clon,
            "bbox": generate_bbox(clat, clon, size_m=BBOX_SIZE_M),
            "direction": "primary",
        })

    for _ in range(n_secondary):
        offset_m = rng.uniform(100, 250)
        clat, clon = _offset_coord(anchor_lat, anchor_lon,
                                   sdir[0] * offset_m, sdir[1] * offset_m)
        candidates.append({
            "lat": clat, "lon": clon,
            "bbox": generate_bbox(clat, clon, size_m=BBOX_SIZE_M),
            "direction": "secondary",
        })

    # Fan boundary candidates evenly across ±20° so they spread along the class edge
    jitter_angles = np.linspace(-np.radians(20), np.radians(20), n_boundary)
    for angle_j in jitter_angles:
        cos_j = np.cos(angle_j)
        sin_j = np.sin(angle_j)
        jdx = bdir[0] * cos_j - bdir[1] * sin_j
        jdy = bdir[0] * sin_j + bdir[1] * cos_j
        offset_m = rng.uniform(50, 150)
        clat, clon = _offset_coord(anchor_lat, anchor_lon,
                                   jdx * offset_m, jdy * offset_m)
        candidates.append({
            "lat": clat, "lon": clon,
            "bbox": generate_bbox(clat, clon, size_m=BBOX_SIZE_M),
            "direction": "boundary",
        })

    logger.info(
        f"Generated {len(candidates)} candidates for anchor "
        f"({anchor_lat:.4f}, {anchor_lon:.4f})"
    )
    return candidates
