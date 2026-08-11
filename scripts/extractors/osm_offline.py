"""Offline OSM use-intensity gate.

Returns an energy-use-intensity multiplier for a coord over the SAME 512 m tile
used by VIIRS/GHSL, from the LOCAL spatially-indexed OSM (no Overpass, no network).

It reads through `osm_extractor.read_local_osm` — the single shared source of truth
(fast GeoPackage R-tree read, with an `osmium extract` fallback) — so the gate and
the full feature extractor never drift apart on what "local OSM" means.

Method: area-weighted dominant landuse (residential/commercial/retail/industrial)
+ presence-priority functional building (office can't be outvoted by generic blocks)
+ power-generation override -> a single multiplier on the demand score.
"""
import sys
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

from collections import defaultdict, Counter
from shapely.geometry import box

from scripts.extractors.osm_extractor import generate_bbox, read_local_osm

# energy-relevant landuse zoning (everything else — grass, forest — is ignored as noise)
RELEVANT = {"industrial", "commercial", "retail", "residential"}
# functional building -> (intensity rank, canonical). highest rank present wins.
BLD = {"office": (5, "office"), "commercial": (4, "commercial"),
       "industrial": (4, "industrial"), "warehouse": (4, "warehouse"),
       "retail": (3, "retail"), "supermarket": (3, "retail"),
       "apartments": (2, "apartments"), "house": (1, "house"),
       "residential": (1, "residential"), "detached": (1, "detached")}


def _multiplier(landuse, building, generation):
    # PENALTY-LED scheme (locked): mild boost, real penalty, no-tag = neutral.
    if generation:                                              return 0.30, "power-gen(supply)"
    if landuse == "industrial" or building in ("industrial", "warehouse"):
        return 1.20, "industrial"
    if landuse == "commercial" or building == "office":         return 1.25, "office/commercial"
    if landuse == "retail" or building in ("retail", "commercial"):
        return 1.05, "retail/commercial"
    if landuse == "residential" or building in ("apartments", "house", "residential", "detached"):
        return 0.70, "residential"
    return 1.00, "none-at-pin"


def extract_osm_use(lat, lon):
    """Return dominant use + energy-intensity multiplier over the 512 m tile."""
    bb = generate_bbox(lat, lon, size_m=512)
    tile = box(bb["min_lon"], bb["min_lat"], bb["max_lon"], bb["max_lat"])
    elements = read_local_osm(bb["min_lat"], bb["max_lat"], bb["min_lon"], bb["max_lon"], lat=lat, lon=lon)

    lu_area = defaultdict(float); bcount = Counter(); generation = False; nbuild = 0
    for el in elements:
        tags = el.get("tags", {}); geom = el.get("geom")
        if geom is None or not geom.intersects(tile):
            continue
        if tags.get("landuse") and geom.geom_type in ("Polygon", "MultiPolygon"):
            lu_area[tags["landuse"]] += geom.intersection(tile).area
        b = tags.get("building")
        if b:
            nbuild += 1
        if b in BLD:
            bcount[BLD[b][1]] += 1
        if tags.get("power") in ("plant", "generator") or tags.get("generator:source"):
            generation = True

    rel = {k: v for k, v in lu_area.items() if k in RELEVANT}
    dom = (max(rel, key=rel.get) if rel
           else (max(lu_area, key=lu_area.get) if lu_area else None))
    rank = {c: r for r, c in BLD.values()}
    present = [b for b, c in bcount.items() if c >= 2] or list(bcount)
    building = max(present, key=lambda b: rank.get(b, 0)) if present else None

    # GUARD: power-gen ×0.30 only if generation DOMINATES the tile (a standalone
    # plant, few buildings) — not when a substation/plant sits inside a city.
    generation_dominant = generation and nbuild < 15
    mult, label = _multiplier(dom, building, generation_dominant)
    return {"osm_dominant_landuse": dom, "osm_key_building": building,
            "osm_use_multiplier": mult, "osm_use_label": label}


if __name__ == "__main__":
    for name, (la, lo) in {"Whitefield": (12.9856, 77.7367),
                            "Chickpet": (12.97, 77.579),
                            "Dharavi": (19.042, 72.853),
                            "Manhattan": (40.7549, -73.984)}.items():
        print(f"{name:12s}", extract_osm_use(la, lo))
