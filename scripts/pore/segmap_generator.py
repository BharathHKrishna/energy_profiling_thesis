"""
PORE Segmentation Map Generator

Side-by-side: ESRI World Imagery | ESA overlay + OSM building polygons + OSM energy infra

Building approach:
  - OSM way["building"] polygons — georeferenced, accurate where mapped
  - Energy infra: power lines, substations, generators, pipelines, dams, railways

Usage:
    python scripts/pore/segmap_generator.py              # all 18 anchors
    python scripts/pore/segmap_generator.py --lat X --lon Y --name Z
"""
import sys, os, math, io, requests, argparse
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import mercantile
from PIL import Image
from pyproj import Transformer
from rasterio.windows import from_bounds as rasterio_from_bounds

from scripts.extractors.worldcover_extractor import fetch_worldcover_pixels, WORLDCOVER_CLASSES
from scripts.extractors.osm_extractor import (
    fetch_osm_elements_offline, circuit_breaker_wait
)
from scripts.extractors.msft_buildings_extractor import fetch_msft_buildings
from scripts.extractors.ghsl_extractor import GHSL_PATHS
from scripts.utils.logger import get_logger
from scripts.utils.geo import bbox as _bbox
from scripts.utils.naming import slug_name

logger = get_logger("segmap_generator")

BASE       = "/srv/THESIS/energy_profiling_thesis"
OUTPUT_DIR = os.path.join(BASE, "outputs/maps/segmaps")
ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TILE_ZOOM  = 17
BG         = "#0f172a"

# ── GHSL population/age raster + colormaps (shared with detection_map.py's
# render_ghsl_det via import) ───────────────────────────────────────────────────
_TO_MOLL = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
EPOCH_RASTER = os.path.join(BASE, "rasters/ghsl/built_epochs/GHS_BUILT_FIRST_EPOCH.tif")
ESA_BUILT_CODE = 50   # ESA WorldCover built-up class

CMAP_POP = mcolors.LinearSegmentedColormap.from_list(
    "pop", ["#fff5f5", "#ff9999", "#ff2222", "#880000", "#110000"])
POP_VMIN, POP_VMAX = 10, 500    # persons per 100m cell — below 10 = uninhabited, skip
NORM_POP = mcolors.LogNorm(vmin=POP_VMIN, vmax=POP_VMAX, clip=True)
POP_MIN, POP_MAX = POP_VMIN, POP_VMAX
POP_SKIP_BELOW = 10

EPOCH_MIN, EPOCH_MAX = 1975, 2020
CMAP_AGE = mcolors.LinearSegmentedColormap.from_list(
    "age", ["#660000", "#cc0000", "#ff6666", "#ffcccc", "#ffffff"])

# ── ESA WorldCover colours ────────────────────────────────────────────────────
WC_COLOURS = {
    10:  "#00ff88",   # tree_cover
    20:  "#aaff44",   # shrubland
    30:  "#ccff66",   # grassland
    40:  "#ff9900",   # cropland
    50:  "#aaaaaa",   # built_up
    60:  "#ffdd88",   # bare_sparse
    70:  "#aaeeff",   # snow_ice
    80:  "#0055ff",   # water
    90:  "#00ddff",   # wetland
    95:  "#006633",   # mangrove
    100: "#cccccc",   # moss_lichen
    0:   "#000000",   # nodata
}

# ── OSM infrastructure colours (energy only — not buildings) ─────────────────
OSM_CLR = {
    "power_line":       "#ffee00",
    "power_tower":      "#ffcc00",
    "power_pole":       "#ffe566",
    "substation":       "#ff7700",
    "power_plant":      "#ff0000",
    "wind_gen":         "#00ffff",
    "wind_turbine":     "#44ffee",   # man_made=wind_turbine
    "hydro_gen":        "#00aaff",
    "pipeline":         "#cc44ff",
    "petroleum_well":   "#ff44cc",
    "storage_tank":     "#ff66ff",
    "dam":              "#ff6600",
    "industrial_area":  "#ff8800",   # landuse=industrial polygon
    "cooling_tower":    "#ffbbaa",
    "chimney":          "#ff7755",
    "vessel":           "#00eeff",   # ship / boat over water
    "aircraft_area":    "#ffaa00",   # aeroway apron / aircraft parking
    "aeroway_line":     "#cc8800",   # runway / taxiway (line)
}

ALL_COORDS = [
    ("Industrial + Water",              39.86217,   -0.075041),
    ("Urban + Coastal",                 43.6965,     7.2663),
    ("Informal + Urban",                12.91341,   23.473813),
    ("Dense Urban",                     48.208354,  16.372504),
    ("Suburban",                       -22.248697, -45.706451),
    ("Industrial",                      51.458202,  21.982493),
    ("Data Centre + Industrial",        50.119929,   8.735434),
    ("Industrial + Arid",               23.545953,  53.311264),
    ("Agriculture / Agrivoltaics", 51.398581,  12.078941),
    ("Industrial + Forest",             58.164353,  54.986046),
    ("Hydropower Reservoir",           -26.893351, -49.060331),
    ("Agricultural + Water",            34.265156, 119.997053),
    ("Coastal + Agricultural",          10.25229,  124.770697),
    ("Mangrove + Industrial",            4.682099, 118.253752),
    ("Suburban + Agricultural",         30.700871,  76.243987),
    ("Coastal + Solar-Wind Hybrid",     58.579193,   5.865124),
]


# ── Geometry helpers ──────────────────────────────────────────────────────────
# _bbox imported from scripts.utils.geo (see top of file) — was duplicated here


def fetch_esri_img(min_lon, min_lat, max_lon, max_lat, zoom=TILE_ZOOM):
    """Fetch ESRI World Imagery tiles and stitch into a single numpy array."""
    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zoom))
    xs    = sorted(set(t.x for t in tiles))
    ys    = sorted(set(t.y for t in tiles))
    full  = Image.new("RGB", (len(xs) * 256, len(ys) * 256))

    for t in tiles:
        r   = requests.get(ESRI_URL.format(z=t.z, y=t.y, x=t.x), timeout=20)
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        full.paste(img, (xs.index(t.x) * 256, ys.index(t.y) * 256))

    # Crop to exact bbox
    t0     = mercantile.Tile(xs[0], ys[0], zoom)
    bounds = mercantile.xy_bounds(t0)
    res    = (bounds.right - bounds.left) / 256

    ulx, uly = mercantile.xy(min_lon, max_lat)
    lrx, lry = mercantile.xy(max_lon, min_lat)
    px0 = int((ulx - bounds.left) / res)
    py0 = int((bounds.top - uly)  / res)
    px1 = int((lrx - bounds.left) / res)
    py1 = int((bounds.top - lry)  / res)

    return np.array(full.crop((px0, py0, px1, py1)))


def _way_coords(el):
    """Extract (lat, lon) list from a way element using inline geometry (out geom)."""
    return [(g["lat"], g["lon"]) for g in el.get("geometry", []) if "lat" in g]


def _node_circle(lat_lon, n=16, radius_deg=0.00009):
    """Synthetic circle polygon (~10m radius) for node-only OSM features."""
    lat, lon = lat_lon
    cos_lat = math.cos(math.radians(abs(lat) or 0.001))
    return [
        (lat + radius_deg * math.sin(2 * math.pi * i / n),
         lon + radius_deg / cos_lat * math.cos(2 * math.pi * i / n))
        for i in range(n)
    ]


def _merge_building_rings(existing_rings, new_rings, iou_thr=0.3, containment_thr=0.7):
    """Merge building rings, deduplicating new_rings against existing.

    Two rejection rules:
      1. IoU >= iou_thr  → same building from two sources (doubled outline)
      2. inter/area_existing >= containment_thr → new ring is a large complex shell
         that already contains existing individual buildings; drop the shell.

    Rule 2 prevents the "big box with small boxes inside" artifact where OSM maps
    an entire facility as one polygon while MS/Overture maps each unit individually.
    """
    if not new_rings:
        return existing_rings
    if not existing_rings:
        return new_rings

    def bbox(ring):
        xs = [c[0] for c in ring]
        ys = [c[1] for c in ring]
        return min(xs), min(ys), max(xs), max(ys)

    def _inter(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    def _area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    def iou(a, b):
        inter = _inter(a, b)
        if inter == 0.0:
            return 0.0
        union = _area(a) + _area(b) - inter
        return inter / union if union > 0 else 0.0

    seen_bboxes = [bbox(r) for r in existing_rings]
    merged = list(existing_rings)

    for ring in new_rings:
        nb = bbox(ring)
        na = _area(nb)
        skip = False
        shells_to_remove = []

        for idx, sb in enumerate(seen_bboxes):
            sa = _area(sb)
            inter = _inter(nb, sb)

            # Rule 1: same building from two sources (symmetric IoU)
            union = na + sa - inter
            if union > 0 and inter / union >= iou_thr:
                skip = True; break

            if inter == 0.0:
                continue

            # Rule 2a: new ring is a large shell — contains an existing small building
            if na > 0 and inter / na >= containment_thr and na > sa:
                skip = True; break

            # Rule 2b: existing ring is a large shell — new ring is an individual inside it
            if sa > 0 and inter / sa >= containment_thr and sa > na:
                shells_to_remove.append(idx)

        if not skip:
            for idx in sorted(shells_to_remove, reverse=True):
                merged.pop(idx)
                seen_bboxes.pop(idx)
            merged.append(ring)
            seen_bboxes.append(nb)
    return merged


def _remove_shells(rings, min_contained=2):
    """Remove building polygons whose bbox contains >= min_contained other building centroids.

    These are OSM block/complex outlines drawn over individually-mapped buildings.
    Keeping them creates a 'big box with small boxes inside' visual artifact.
    """
    if len(rings) < min_contained + 1:
        return rings

    def bbox(ring):
        xs = [c[0] for c in ring]; ys = [c[1] for c in ring]
        return min(xs), min(ys), max(xs), max(ys)

    def ctr(ring):
        xs = [c[0] for c in ring]; ys = [c[1] for c in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    bboxes = [bbox(r) for r in rings]
    ctrs   = [ctr(r)  for r in rings]
    shells = set()

    for i, (bx1, by1, bx2, by2) in enumerate(bboxes):
        if i in shells:
            continue
        contained = sum(
            1 for j, (cx, cy) in enumerate(ctrs)
            if j != i and bx1 <= cx <= bx2 and by1 <= cy <= by2
        )
        if contained >= min_contained:
            shells.add(i)

    if shells:
        logger.info(f"Shell removal: dropped {len(shells)} complex outlines")
    return [r for i, r in enumerate(rings) if i not in shells]


def _filter_min_area(rings, ref_lat, min_m2=18.0):
    """Drop building rings whose bounding-box area is below min_m2.

    Removes sub-pixel noise (ML false positives, tiny courtyard artefacts)
    that clutter dense informal-settlement maps without losing real buildings
    (even the smallest informal houses are typically ≥ 3 m × 6 m = 18 m²).
    """
    cos_lat = math.cos(math.radians(abs(ref_lat) or 0.001))
    kept = []
    for ring in rings:
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        dlat_m = (max(lats) - min(lats)) * 111320
        dlon_m = (max(lons) - min(lons)) * 111320 * cos_lat
        if dlat_m * dlon_m >= min_m2:
            kept.append(ring)
    dropped = len(rings) - len(kept)
    if dropped:
        logger.info(f"Min-area filter: dropped {dropped} sub-{min_m2:.0f}m² polygons")
    return kept


def _filter_industrial_on_water(infra, pixels, min_lat, max_lat, min_lon, max_lon):
    """Remove industrial_area polygons whose centroid pixel is ESA water (80).

    Catches mis-tagged or geographically imprecise OSM landuse=industrial polygons
    that cover marina / port water areas where ESA correctly classifies as water.
    Uses centroid pixel only (faster than bbox scan, sufficient for zone polygons).
    """
    if pixels is None or not infra.get("industrial_area"):
        return infra
    h, w = pixels.shape
    filtered = []
    for poly in infra["industrial_area"]:
        if not isinstance(poly, list) or len(poly) < 3:
            filtered.append(poly)
            continue
        cy = sum(c[0] for c in poly) / len(poly)
        cx = sum(c[1] for c in poly) / len(poly)
        row = int((max_lat - cy) / (max_lat - min_lat) * h)
        col = int((cx - min_lon) / (max_lon - min_lon) * w)
        row = max(0, min(h - 1, row))
        col = max(0, min(w - 1, col))
        if pixels[row, col] != 80:   # not water — keep
            filtered.append(poly)
    dropped = len(infra["industrial_area"]) - len(filtered)
    if dropped:
        logger.info(f"Industrial water-filter: dropped {dropped} polygons on ESA water")
    infra["industrial_area"] = filtered
    return infra


def _on_water(ring, pixels, min_lat, max_lat, min_lon, max_lon, lonlat=True):
    """True if polygon centroid neighbourhood is majority ESA water (class 80).

    Samples a 3×3 pixel window around the centroid and requires ≥5/9 water pixels.
    This avoids false-positive vessel tagging for waterfront buildings whose centroid
    lands on a single coastal boundary pixel that ESA misclassifies as water.
    lonlat=True  → ring is [(lon, lat), ...] (building format)
    lonlat=False → ring is [(lat, lon), ...] (infra format)
    """
    if pixels is None or len(ring) < 3:
        return False
    h, w = pixels.shape
    if lonlat:
        cx = sum(c[0] for c in ring) / len(ring)   # lon
        cy = sum(c[1] for c in ring) / len(ring)   # lat
    else:
        cy = sum(c[0] for c in ring) / len(ring)   # lat
        cx = sum(c[1] for c in ring) / len(ring)   # lon
    # Sample the full bounding box of the ring — catches ships docked at piers
    # whose centroid is on a pier pixel but whose footprint spans open water.
    # Threshold: ≥20% of bbox pixels are water (class 80).
    if lonlat:
        lons = [c[0] for c in ring]; lats = [c[1] for c in ring]
    else:
        lats = [c[0] for c in ring]; lons = [c[1] for c in ring]
    c_min = max(0, min(w - 1, int((min(lons) - min_lon) / (max_lon - min_lon) * w)))
    c_max = max(0, min(w - 1, int((max(lons) - min_lon) / (max_lon - min_lon) * w)))
    r_min = max(0, min(h - 1, int((max_lat - max(lats)) / (max_lat - min_lat) * h)))
    r_max = max(0, min(h - 1, int((max_lat - min(lats)) / (max_lat - min_lat) * h)))
    total = (c_max - c_min + 1) * (r_max - r_min + 1)
    if total == 0:
        return False
    water = int(pixels[r_min:r_max+1, c_min:c_max+1].flatten().tolist().count(80))
    return water / total >= 0.10


def _split_buildings_water(bldg_rings, pixels, min_lat, max_lat, min_lon, max_lon):
    """Drop buildings whose centroid is on ESA water — ships/vessels are not buildings."""
    if pixels is None:
        return bldg_rings, []
    land = [r for r in bldg_rings
            if not _on_water(r, pixels, min_lat, max_lat, min_lon, max_lon, lonlat=True)]
    return land, []


def _filter_infra_water(infra, pixels, min_lat, max_lat, min_lon, max_lon):
    """Discard area-type infra polygons (lat, lon) whose centroid is on ESA water."""
    if pixels is None:
        return infra
    AREA_KEYS = {"substation", "power_plant",
                 "wind_gen", "hydro_gen", "industrial_area",
                 "storage_tank", "wind_turbine", "petroleum_well",
                 "cooling_tower", "chimney"}
    for key in AREA_KEYS:
        if key in infra:
            infra[key] = [item for item in infra[key]
                          if not (isinstance(item, list) and len(item) >= 3 and
                                  _on_water(item, pixels, min_lat, max_lat,
                                            min_lon, max_lon, lonlat=False))]
    return infra


def _point_in_ring(px, py, ring_lonlat):
    """Ray-casting point-in-polygon. px=lon, py=lat, ring is [(lon,lat),...]."""
    n = len(ring_lonlat)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring_lonlat[i]
        xj, yj = ring_lonlat[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


# Infra types that can be matched to a building footprint and coloured.
# Only types that represent actual energy structures placed inside a building shell.
_BUILDING_INFRA_TYPES = [
    "substation", "power_plant", "hydro_gen",
    "storage_tank", "cooling_tower", "chimney", "petroleum_well",
]


def _is_real_osm_polygon(item):
    """True if item looks like a real OSM way polygon, not a synthetic node circle.

    Synthetic circles are always exactly 16 points (from _node_circle n=16).
    Even if they happen to have 16 pts, their bbox diagonal is tiny (< 0.0005°).
    Real power stations / substations are much larger.
    """
    if not item or len(item) < 4:
        return False
    if len(item) == 16:   # exact synthetic circle count → likely node
        lons = [c[1] for c in item]; lats = [c[0] for c in item]
        diag = ((max(lats)-min(lats))**2 + (max(lons)-min(lons))**2) ** 0.5
        if diag < 0.0006:   # < ~60 m diagonal → synthetic circle, skip
            return False
    return True


def _classify_buildings_by_infra(bldg_rings, infra):
    """Split building rings into infra-tagged and plain.

    For each infra feature whose centroid falls inside a building ring,
    that ring is coloured with the infra type instead of plain white.

    Only real OSM way polygons trigger building colouring — synthetic node
    circles (from _node_circle) are excluded to avoid mis-tagging unrelated
    large buildings whose footprint happens to contain a generator/substation node.

    Returns (tagged: {ptype: [ring,...]}, plain: [ring,...]).
    """
    tagged = {}
    matched = set()
    for ptype in _BUILDING_INFRA_TYPES:
        for item in infra.get(ptype, []):
            if not _is_real_osm_polygon(item):
                continue
            # infra items are (lat, lon) — get centroid as lon, lat
            cx = sum(c[1] for c in item) / len(item)
            cy = sum(c[0] for c in item) / len(item)
            for idx, ring in enumerate(bldg_rings):
                if idx in matched or len(ring) < 3:
                    continue
                if _point_in_ring(cx, cy, ring):
                    tagged.setdefault(ptype, []).append(ring)
                    matched.add(idx)
                    break
    plain = [r for i, r in enumerate(bldg_rings) if i not in matched]
    if tagged:
        summary = {k: len(v) for k, v in tagged.items()}
        logger.info(f"Infra-coloured buildings: {summary}")
    return tagged, plain


def extract_osm_buildings(elements):
    """
    Extract building polygon rings from Overpass elements (out geom format).
    Returns list of (lon, lat) rings suitable for MplPolygon.
    Handles way, node (point-mapped), and relation (multipolygon) building tags.
    Includes aeroway structures (hangar, terminal) not always tagged building=*.
    """
    rings = []
    for el in elements:
        tags = el.get("tags", {})
        is_building = bool(tags.get("building"))
        aeroway = tags.get("aeroway", "")
        is_aeroway_struct = aeroway in ("hangar", "terminal", "control_tower", "gate")
        if not is_building and not is_aeroway_struct:
            continue

        etype = el.get("type")
        if etype == "way":
            coords = _way_coords(el)
            if len(coords) >= 3:
                rings.append([(c[1], c[0]) for c in coords])

        elif etype == "node" and "lat" in el:
            # Point-mapped building — synthetic small polygon
            circ = _node_circle((el["lat"], el["lon"]), radius_deg=0.00009)
            rings.append([(c[1], c[0]) for c in circ])

        elif etype == "relation":
            # Multipolygon building — extract outer member rings (inline geom)
            for member in el.get("members", []):
                if member.get("role") == "outer" and "geometry" in member:
                    coords = [(g["lat"], g["lon"])
                              for g in member["geometry"] if "lat" in g]
                    if len(coords) >= 3:
                        rings.append([(c[1], c[0]) for c in coords])
    return rings


# ── ESA overlay ───────────────────────────────────────────────────────────────

def build_esa_overlay(pixels):
    rgba = np.zeros((*pixels.shape, 4), dtype=np.float32)
    for code, hx in WC_COLOURS.items():
        r = int(hx[1:3], 16) / 255
        g = int(hx[3:5], 16) / 255
        b = int(hx[5:7], 16) / 255
        rgba[pixels == code] = [r, g, b, 0.65]
    rgba[pixels == 0, 3] = 0.0
    return rgba


# ── OSM energy infrastructure (not buildings) ─────────────────────────────────

def extract_osm_infra(elements):
    """Extract only energy infrastructure geometry from Overpass elements (out geom)."""
    infra = {k: [] for k in OSM_CLR}

    def cen(coords):
        return (sum(c[0] for c in coords) / len(coords),
                sum(c[1] for c in coords) / len(coords))

    for el in elements:
        tags    = el.get("tags", {})
        if not tags:
            continue
        is_way  = el.get("type") == "way"
        is_node = el.get("type") == "node"

        if is_way:
            coords = _way_coords(el)        # inline geometry from out geom
        elif is_node and "lat" in el:
            coords = [(el["lat"], el["lon"])]
        else:
            coords = []

        if not coords:
            continue

        pwr = tags.get("power", "")
        mm  = tags.get("man_made", "")
        ww  = tags.get("waterway", "")
        lu  = tags.get("landuse", "")
        gs  = tags.get("generator:source", "").lower()
        ps  = tags.get("plant:source", "").lower()

        def geo(feature_coords, radius_deg=0.00009):
            """Polygon ring if way with ≥3 coords, else synthetic circle for nodes."""
            if is_way and len(feature_coords) >= 3:
                return feature_coords
            return _node_circle(cen(feature_coords), radius_deg=radius_deg)

        # Power infrastructure
        if pwr == "line":
            infra["power_line"].append(coords)
        elif pwr == "tower":
            infra["power_tower"].append(_node_circle(cen(coords), radius_deg=0.00015))
        elif pwr == "pole":
            infra["power_pole"].append(_node_circle(cen(coords), radius_deg=0.00012))
        elif pwr == "substation":
            infra["substation"].append(geo(coords, radius_deg=0.00027))
        elif pwr == "plant":
            if "wind" in ps:
                infra["wind_gen"].append(geo(coords, radius_deg=0.00025))
            elif "hydro" in ps or "water" in ps:
                infra["hydro_gen"].append(geo(coords, radius_deg=0.00022))
            elif "solar" not in ps:
                infra["power_plant"].append(geo(coords, radius_deg=0.00035))
        elif pwr == "generator" or "generator:source" in tags:
            if "wind" in gs:
                infra["wind_gen"].append(geo(coords, radius_deg=0.00025))
            elif "hydro" in gs or "water" in gs:
                infra["hydro_gen"].append(geo(coords, radius_deg=0.00022))

        # Man-made features
        if mm == "pipeline":
            infra["pipeline"].append(coords)
        elif mm == "petroleum_well":
            infra["petroleum_well"].append(_node_circle(cen(coords), radius_deg=0.00009))
        elif mm == "storage_tank":
            infra["storage_tank"].append(geo(coords, radius_deg=0.00015))
        elif mm == "wind_turbine":
            infra["wind_turbine"].append(geo(coords, radius_deg=0.00025))
        elif mm == "cooling_tower":
            infra["cooling_tower"].append(geo(coords, radius_deg=0.00015))
        elif mm == "chimney":
            infra["chimney"].append(geo(coords, radius_deg=0.00010))

        # Waterway
        if ww == "dam":
            infra["dam"].append(coords)

        # Landuse areas
        if lu == "industrial" and is_way and len(coords) >= 3:
            infra["industrial_area"].append(coords)

        # Aeroway — aircraft parking areas and movement lines
        aeroway = tags.get("aeroway", "")
        if aeroway in ("apron", "aircraft_crossing") and is_way and len(coords) >= 3:
            infra["aircraft_area"].append(coords)
        elif aeroway in ("runway", "taxiway") and is_way:
            infra["aeroway_line"].append(coords)

    return infra


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_overlays(ax, esri_img, pixels, ok,
                  bldg_rings, vessel_rings, infra,
                  min_lat, max_lat, min_lon, max_lon):

    # ESA raster
    if ok and pixels is not None:
        ax.imshow(build_esa_overlay(pixels),
                  extent=[min_lon, max_lon, min_lat, max_lat],
                  origin="upper", zorder=2, aspect="auto")

    # OSM energy infra — polygon areas (drawn BEFORE buildings so outlines stay visible)
    # Large background fills (no edge outline — avoids border conflicts with building outlines)
    FILL_ONLY = {"industrial_area", "aircraft_area"}
    POLY_TYPES = [
        "substation", "power_plant",
        "wind_gen", "wind_turbine",
        "hydro_gen",
        "petroleum_well", "storage_tank",
        "cooling_tower", "chimney",
        "industrial_area",
        "power_tower", "power_pole",
        "aircraft_area",
    ]
    for ptype in POLY_TYPES:
        patches = []
        for item in infra.get(ptype, []):
            if not isinstance(item, list) or len(item) < 3:
                continue
            lons = [c[1] for c in item]
            lats = [c[0] for c in item]
            patches.append(MplPolygon(list(zip(lons, lats)), closed=True))
        if patches:
            clr = OSM_CLR[ptype]
            r, g, b = int(clr[1:3],16)/255, int(clr[3:5],16)/255, int(clr[5:7],16)/255
            fill_only = ptype in FILL_ONLY
            ax.add_collection(PatchCollection(
                patches,
                facecolor=(r, g, b, 0.18 if fill_only else 0.25),
                edgecolor="none" if fill_only else clr,
                linewidth=0 if fill_only else 1.2,
                zorder=3,
            ))

    # Classify buildings: infra-matched get the infra colour, rest are plain white
    tagged_bldgs, plain_bldgs = _classify_buildings_by_infra(bldg_rings, infra)

    # Infra-tagged buildings — filled with infra colour, outlined with infra colour
    for ptype, rings in tagged_bldgs.items():
        clr = OSM_CLR[ptype]
        r, g, b = int(clr[1:3],16)/255, int(clr[3:5],16)/255, int(clr[5:7],16)/255
        tpolys = [MplPolygon(ring, closed=True) for ring in rings if len(ring) >= 3]
        if tpolys:
            ax.add_collection(PatchCollection(
                tpolys,
                facecolor=(r, g, b, 0.55),
                edgecolor=clr,
                linewidth=1.2,
                zorder=4,
            ))

    # Plain building polygons — cyan-tinted outline so buildings are visible on
    # both dark (forest, water) and light (sand, bare ground) satellite backgrounds.
    # Adaptive density: high counts reduce linewidth/alpha to prevent clutter.
    n_bldgs = len(plain_bldgs)
    if n_bldgs > 300:
        bldg_lw, bldg_alpha = 0.8, 0.18
    elif n_bldgs > 150:
        bldg_lw, bldg_alpha = 1.0, 0.20
    elif n_bldgs > 80:
        bldg_lw, bldg_alpha = 1.2, 0.22
    else:
        bldg_lw, bldg_alpha = 1.5, 0.25
    polys = []
    for ring in plain_bldgs:
        if len(ring) >= 3:
            polys.append(MplPolygon(ring, closed=True))
    if polys:
        ax.add_collection(PatchCollection(
            polys,
            facecolor=(1, 1, 1, bldg_alpha),
            edgecolor="#aaffff",   # cyan-white: visible on sand, soil, and dark backgrounds
            linewidth=bldg_lw,
            zorder=4,
        ))

    # OSM energy infra — lines (topmost, above buildings)
    LINE_STYLE = {
        "power_line":   (1.5, "-"),
        "pipeline":     (2.0, "-"),
        "dam":          (2.5, "-"),
        "aeroway_line": (2.0, "-"),
    }
    for ltype, (lw, ls) in LINE_STYLE.items():
        for line in infra.get(ltype, []):
            lons = [c[1] for c in line]
            lats = [c[0] for c in line]
            ax.plot(lons, lats, color=OSM_CLR[ltype], lw=lw, ls=ls,
                    alpha=0.95, zorder=5)


def build_legend(pixels, ok, bldg_rings, vessel_rings, infra):
    items = []

    if ok and pixels is not None:
        for code in sorted(set(pixels.flatten()) - {0}):
            name = WORLDCOVER_CLASSES.get(code, str(code))
            items.append(mpatches.Patch(color=WC_COLOURS[code],
                         label=f"ESA: {name}", alpha=0.85))

    if bldg_rings:
        items.append(mpatches.Patch(facecolor=(1,1,1,0.3), edgecolor="#aaffff",
                     label=f"buildings ({len(bldg_rings)})"))

    OSM_LABELS = {
        "power_line":     "power line",
        "power_tower":    "tower",
        "power_pole":     "pole",
        "substation":     "substation",
        "power_plant":    "power plant",
        "wind_gen":       "wind generator",
        "wind_turbine":   "wind turbine",
        "hydro_gen":      "hydro generator",
        "pipeline":       "pipeline",
        "petroleum_well": "petroleum well",
        "storage_tank":   "storage tank",
        "dam":            "dam",
        "cooling_tower":  "cooling tower",
        "chimney":        "chimney",
        "industrial_area":"industrial area",
        "aircraft_area":  "aircraft area",
        "aeroway_line":   "runway/taxiway",
    }
    LINE_KEYS = {"power_line", "pipeline", "dam", "aeroway_line"}
    for k, label in OSM_LABELS.items():
        entries = infra.get(k, [])
        if not entries:
            continue
        if k in LINE_KEYS:
            count = len(entries)
        else:
            # Only count polygon items (lists) — nodes are not drawn
            count = sum(1 for e in entries if isinstance(e, list) and len(e) >= 3)
        if count:
            items.append(mpatches.Patch(color=OSM_CLR[k],
                         label=f"OSM: {label} ({count})"))
    return items


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_pair(ax_orig, ax_seg, esri_img,
                pixels, ok,
                min_lat, max_lat, min_lon, max_lon, name):
    BG = "#0f172a"

    for ax in (ax_orig, ax_seg):
        ax.set_facecolor(BG)
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor("#334155")

    # Left — ESRI satellite
    ax_orig.imshow(esri_img,
                   extent=[min_lon, max_lon, min_lat, max_lat],
                   origin="upper", zorder=1, aspect="auto")
    ax_orig.set_title("ESRI World Imagery", color="#94a3b8",
                      fontsize=8, fontfamily="monospace", pad=4)
    ax_orig.set_ylabel(name, color="#e2e8f0", fontsize=8,
                       fontfamily="monospace", labelpad=6)

    # Right — ESA WorldCover land cover only
    ax_seg.imshow(esri_img,
                  extent=[min_lon, max_lon, min_lat, max_lat],
                  origin="upper", zorder=1, aspect="auto")
    if ok and pixels is not None:
        ax_seg.imshow(build_esa_overlay(pixels),
                      extent=[min_lon, max_lon, min_lat, max_lat],
                      origin="upper", zorder=2, aspect="auto")

    # Legend — ESA classes present
    if ok and pixels is not None:
        import matplotlib.patches as mpatches
        items = []
        for code, hx in WC_COLOURS.items():
            if code == 0 or not np.any(pixels == code):
                continue
            label = WORLDCOVER_CLASSES.get(code, str(code))
            items.append(mpatches.Patch(color=hx, label=label))
        if items:
            ax_seg.legend(handles=items, loc="lower left", facecolor=BG,
                          labelcolor="#e2e8f0", fontsize=7, edgecolor="#334155",
                          ncol=2, framealpha=0.92, borderpad=0.6)
    ax_seg.set_title("ESA WorldCover Land Cover", color="#94a3b8",
                     fontsize=8, fontfamily="monospace", pad=4)


def fetch_osm_elements(name, min_lat, max_lat, min_lon, max_lon, lat=None, lon=None):
    """
    Fetch OSM elements for bbox — one call shared across all map generators.
    OFFLINE: reads the local planet tile (live osmium-extract fallback). Returns
    Overpass-format elements ({type, tags, geometry}) so the parsers stay unchanged.
    """
    elements = fetch_osm_elements_offline(min_lat, max_lat, min_lon, max_lon, lat=lat, lon=lon)
    logger.info(f"[{name}] OSM (offline) {len(elements)} elements")
    return elements or []


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_all(name, lat, lon, esri_img=None, wc_pixels=None):
    """esri_img/wc_pixels: pass in already-fetched values to skip the network/S3 call —
    the same bbox's imagery and WorldCover pixels get fetched independently by up to 4
    map calls per coordinate otherwise (generate_single, render_ghsl_segmap here, plus
    detection_map.py's own ESRI fetch) for identical data. Fixed 2026-08-11."""
    min_lat, max_lat, min_lon, max_lon = _bbox(lat, lon)

    if esri_img is None:
        try:
            esri_img = fetch_esri_img(min_lon, min_lat, max_lon, max_lat)
        except Exception as e:
            logger.warning(f"[{name}] ESRI fetch failed: {e}")
            esri_img = np.zeros((512, 512, 3), dtype=np.uint8)

    if wc_pixels is None:
        pixels, _, ok = fetch_worldcover_pixels(min_lat, max_lat, min_lon, max_lon, lat, lon)
    else:
        pixels, ok = wc_pixels

    return min_lat, max_lat, min_lon, max_lon, esri_img, pixels, ok


# ── GHSL sampling (population / urbanisation-epoch) ────────────────────────────
# Shared with detection_map.py's render_ghsl_det, which imports the functions and
# colormaps below rather than duplicating them.

def _moll_window(src, min_lat, max_lat, min_lon, max_lon):
    """Convert WGS84 bbox to rasterio Window in Mollweide raster coords."""
    xs, ys = _TO_MOLL.transform(
        [min_lon, max_lon, min_lon, max_lon],
        [min_lat, min_lat, max_lat, max_lat]
    )
    return rasterio_from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform)


def _read_ghsl_grid(raster_path, min_lat, max_lat, min_lon, max_lon,
                    out_shape, nodata_val=-200.0):
    """Read a GHSL Mollweide raster window and zoom to out_shape."""
    try:
        import rasterio
        from scipy.ndimage import zoom
        with rasterio.open(raster_path) as src:
            win = _moll_window(src, min_lat, max_lat, min_lon, max_lon)
            data = src.read(1, window=win).astype(np.float32)
            nd = src.nodata if src.nodata is not None else nodata_val
            data[data == nd] = 0.0
            data[data < 0] = 0.0
        if data.size == 0:
            return np.zeros(out_shape, dtype=np.float32)
        zh = out_shape[0] / max(data.shape[0], 1)
        zw = out_shape[1] / max(data.shape[1], 1)
        return zoom(data, (zh, zw), order=0).astype(np.float32)
    except Exception as e:
        logger.warning(f"GHSL grid read failed ({raster_path}): {e}")
        return np.zeros(out_shape, dtype=np.float32)


def _sample_epoch_grid(min_lat, max_lat, min_lon, max_lon, out_shape):
    return _read_ghsl_grid(EPOCH_RASTER, min_lat, max_lat, min_lon, max_lon, out_shape)


def _sample_pop_grid(min_lat, max_lat, min_lon, max_lon, out_shape):
    # Was a hardcoded literal path here — now reuses ghsl_extractor.py's own
    # config.yaml-driven path, so an epoch update in config.yaml can't silently leave
    # this map generator pointing at a different raster than the feature extractor uses.
    return _read_ghsl_grid(GHSL_PATHS["population"], min_lat, max_lat, min_lon, max_lon, out_shape)


def _ax_base(ax, esri_img, min_lon, max_lon, min_lat, max_lat, title, ylabel=""):
    ax.set_facecolor(BG)
    ax.set_xlim(min_lon, max_lon); ax.set_ylim(min_lat, max_lat)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("#334155")
    ax.imshow(esri_img, extent=[min_lon, max_lon, min_lat, max_lat],
              origin="upper", zorder=1, aspect="auto")
    ax.set_title(title, color="#94a3b8", fontsize=8, fontfamily="monospace", pad=4)
    if ylabel:
        ax.set_ylabel(ylabel, color="#e2e8f0", fontsize=8,
                      fontfamily="monospace", labelpad=6)


def _ghsl_segmap_overlay(pixels, ghsl_grid, cmap, vmin, vmax, norm=None):
    """
    Build RGBA overlay: built-up pixels coloured by GHSL value.
    Non-built-up pixels use ESA colour at lower opacity.
    """
    h, w = pixels.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    # Non-built-up: ESA colour at 0.25 opacity
    for code, hx in WC_COLOURS.items():
        if code in (0, ESA_BUILT_CODE):
            continue
        mask = pixels == code
        r = int(hx[1:3], 16) / 255
        g = int(hx[3:5], 16) / 255
        b = int(hx[5:7], 16) / 255
        rgba[mask] = [r, g, b, 0.25]

    # Built-up: GHSL colour
    # For age variant (norm=None, vmin=EPOCH_MIN): cells at 1975 (pre-satellite era)
    # shown as neutral grey — only post-1975 cells get the colour ramp.
    built_mask = pixels == ESA_BUILT_CODE
    if built_mask.any() and ghsl_grid is not None:
        g_h, g_w = ghsl_grid.shape
        if (g_h, g_w) != (h, w):
            from scipy.ndimage import zoom
            ghsl_grid = zoom(ghsl_grid, (h / g_h, w / g_w), order=0)
        n = norm if norm is not None else mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        colours = cmap(n(ghsl_grid))
        colours[..., 3] = 0.7

        # For population: skip cells below threshold (uninhabited) — leave transparent
        if norm is not None:  # pop uses LogNorm
            low_pop = (ghsl_grid < POP_SKIP_BELOW) & built_mask
            colours[low_pop, 3] = 0.0  # transparent

        # Option B: for age, grey-out pre-1975 cells (pre-existing, indeterminate)
        if norm is None and vmin == EPOCH_MIN:
            pre_existing = (ghsl_grid <= EPOCH_MIN) & built_mask
            colours[pre_existing] = [0.45, 0.45, 0.45, 0.45]

        rgba[built_mask] = colours[built_mask]

    return rgba


def render_ghsl_segmap(name, lat, lon, variant="pop", save=True, esri_img=None, wc_pixels=None):
    """
    variant: 'pop' = population density, 'age' = urbanisation epoch
    esri_img/wc_pixels: pass in already-fetched values to avoid re-fetching the same
    bbox's imagery/WorldCover pixels a 2nd time (generate_single already fetched them
    for this same coordinate) — see _fetch_all()'s docstring.
    """
    min_lat, max_lat, min_lon, max_lon = _bbox(lat, lon)
    logger.info(f"[{name}] GHSL {variant} segmap...")

    if esri_img is None:
        try:
            esri_img = fetch_esri_img(min_lon, min_lat, max_lon, max_lat)
        except Exception as e:
            logger.warning(f"[{name}] ESRI failed: {e}")
            esri_img = np.zeros((512, 512, 3), dtype=np.uint8)

    if wc_pixels is None:
        pixels, _, ok = fetch_worldcover_pixels(min_lat, max_lat, min_lon, max_lon, lat, lon)
    else:
        pixels, ok = wc_pixels

    h, w = esri_img.shape[:2]
    if variant == "pop":
        ghsl_grid = _sample_pop_grid(min_lat, max_lat, min_lon, max_lon, (h, w))
        ghsl_grid = np.where(ghsl_grid > 0, ghsl_grid, 1).astype(np.float32)
        cmap, vmin, vmax, norm = CMAP_POP, POP_MIN, POP_MAX, NORM_POP
        title = "Population Density (GHSL 2020)"
        legend_label = "Population (persons per 100m cell, log scale)"
    else:
        ghsl_grid = _sample_epoch_grid(min_lat, max_lat, min_lon, max_lon, (h, w))
        cmap, vmin, vmax, norm = CMAP_AGE, EPOCH_MIN, EPOCH_MAX, None
        title = "Urbanisation Epoch (GHSL 1975–2020)"
        legend_label = "First built-up epoch"

    fig, (ax_orig, ax_seg) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(BG)

    _ax_base(ax_orig, esri_img, min_lon, max_lon, min_lat, max_lat,
             "ESRI World Imagery", ylabel=name)
    _ax_base(ax_seg, esri_img, min_lon, max_lon, min_lat, max_lat, title)

    if ok and pixels is not None:
        overlay = _ghsl_segmap_overlay(pixels, ghsl_grid, cmap, vmin, vmax, norm=norm)
        ax_seg.imshow(overlay, extent=[min_lon, max_lon, min_lat, max_lat],
                      origin="upper", zorder=2, aspect="auto")

        # Colorbar
        cb_norm = norm if norm is not None else mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=cb_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_seg, fraction=0.03, pad=0.02)
        cbar.set_label(legend_label, color="#e2e8f0", fontsize=7)
        cbar.ax.yaxis.set_tick_params(color="#e2e8f0")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#e2e8f0", fontsize=6)

        # Non-built-up legend
        items = []
        for code, hx in WC_COLOURS.items():
            if code in (0, ESA_BUILT_CODE) or not np.any(pixels == code):
                continue
            items.append(mpatches.Patch(color=hx, alpha=0.4,
                                        label=WORLDCOVER_CLASSES.get(code, str(code))))
        if items:
            ax_seg.legend(handles=items, loc="lower left", facecolor=BG,
                          labelcolor="#e2e8f0", fontsize=6, edgecolor="#334155",
                          ncol=2, framealpha=0.9)

    fig.suptitle(f"{name}  ·  {lat:.5f}, {lon:.5f}  ·  512 m bbox",
                 color="#e2e8f0", fontsize=10, y=1.01)
    plt.tight_layout(w_pad=0.4)

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = os.path.join(OUTPUT_DIR, f"{slug_name(name)}_{variant}_segmap.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info(f"Saved: {out}")
        return out
    return fig


# ── Public API ────────────────────────────────────────────────────────────────

def generate_single(lat, lon, stratum_name="anchor", save=True, esri_img=None, wc_pixels=None):
    logger.info(f"[{stratum_name}] Processing...")
    min_lat, max_lat, min_lon, max_lon, esri_img, pixels, ok = \
        _fetch_all(stratum_name, lat, lon, esri_img=esri_img, wc_pixels=wc_pixels)

    fig, (ax_orig, ax_seg) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax_orig, ax_seg): ax.set_facecolor("#0f172a")

    render_pair(ax_orig, ax_seg, esri_img, pixels, ok,
                min_lat, max_lat, min_lon, max_lon, stratum_name)

    fig.suptitle(f"{stratum_name}  ·  {lat:.5f}, {lon:.5f}  ·  512 m bbox",
                 color="#e2e8f0", fontsize=11, y=1.01)
    plt.tight_layout(w_pad=0.4)

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out  = os.path.join(OUTPUT_DIR, f"{slug_name(stratum_name)}_segmap.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f172a")
        plt.close(fig)
        logger.info(f"Saved: {out}")
        return out
    return fig


def generate_multi(coords=None, save=True):
    if coords is None:
        coords = ALL_COORDS
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved = []
    for name, lat, lon in coords:
        out = generate_single(lat, lon, stratum_name=name, save=save)
        saved.append(out)
    logger.info(f"Done — {len(saved)} segmaps saved to {OUTPUT_DIR}")
    return saved


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lat",  type=float, default=None)
    p.add_argument("--lon",  type=float, default=None)
    p.add_argument("--name", type=str,   default=None)
    args = p.parse_args()

    if args.lat and args.lon:
        print(generate_single(args.lat, args.lon, args.name or "anchor"))
    else:
        generate_multi()
