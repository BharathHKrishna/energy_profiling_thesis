"""
OpenStreetMap tiles kept as Overpass JSON, with no PBF step in between.

Why this exists. The original path fetched a tile from Overpass as JSON, wrote it
to a .osm.pbf with pyosmium, and read it back by shelling out to `osmium export
-f geojsonseq`. That last command requires its input ordered by ID (nodes by id,
then ways, then relations), which is stated in its own help text. Overpass returns
elements in query order, not ID order, and the writer preserved that order, so
`osmium export` silently discarded every way and emitted only the tagged nodes.

Buildings, land-use polygons, waterways and power lines are all ways. Measured on
the Frankfurt Altstadt tile: the stored file holds 397 buildings, and
`osmium export` reported none of them. Across a 60-tile sample of the thesis run's
own tiles, 67 percent held buildings that the extracted record shows as zero, a
mean of about 101 buildings per tile.

Sorting the file first does not rescue it; tried directly, export then returned a
single feature. The round trip is the problem, not the ordering, so this module
removes it. Overpass already returns everything needed, including the nodes a way
references (the `>` recursion in the tile query), so way geometry can be assembled
directly and the PBF serves no purpose.

Tiles are cached as gzipped JSON next to the old ones. Nothing else in the
extraction changes: the element list this produces has the same shape the previous
reader returned, `{"tags": {...}, "geom": shapely-or-None}`.
"""

import glob
import gzip
import json
import os

TILES_DIR = "/srv/THESIS/osm_planet/tiles_json"


def tile_name(lat, lon):
    """Same 5-decimal convention the PBF tiles used, so the two are comparable."""
    return f"{lat:.5f}_{lon:.5f}.json.gz"


def tile_path(lat, lon, tiles_dir=TILES_DIR):
    return os.path.join(tiles_dir, tile_name(lat, lon))


def write_tile(elements, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(elements, fh, separators=(",", ":"))
    os.replace(tmp, path)


def read_tile_raw(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def elements_from_overpass(elements):
    """Overpass JSON -> the unified element list the feature extractor expects.

    Untagged nodes are dropped from the result but used first to resolve way
    geometry; they exist in the response only because the query recurses down to
    collect them. A closed way becomes a Polygon and an open one a LineString,
    which is what the renderers and the tag readers downstream both expect.
    """
    from shapely.geometry import Point, LineString, Polygon

    coords = {}
    for e in elements:
        if e.get("type") == "node" and "lat" in e and "lon" in e:
            coords[e["id"]] = (e["lon"], e["lat"])

    out = []
    for e in elements:
        tags = e.get("tags") or {}
        if not tags:
            continue
        kind = e.get("type")
        geom = None
        try:
            if kind == "node":
                geom = Point(e["lon"], e["lat"])
            elif kind == "way":
                pts = [coords[r] for r in e.get("nodes", []) if r in coords]
                if len(pts) >= 2:
                    closed = len(pts) >= 4 and pts[0] == pts[-1]
                    geom = Polygon(pts) if closed else LineString(pts)
        except Exception:
            geom = None          # a malformed ring keeps its tags, loses its shape
        out.append({"tags": tags, "geom": geom})
    return out


def read_tile(lat, lon, tiles_dir=TILES_DIR):
    """Cached tile as an element list, or None when no tile has been fetched."""
    p = tile_path(lat, lon, tiles_dir)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    return elements_from_overpass(read_tile_raw(p))


def fetch_tile(lat, lon, tiles_dir=TILES_DIR, mirrors=None, force=False):
    """Fetch one tile from Overpass and cache it. Returns the element list.

    Returns None when every mirror fails, which the caller must treat as "not
    retrieved" rather than "nothing is there". That distinction is what the
    original tile-retrieval defect got wrong (Section 6.6): an empty response was
    written to disk as though it were a genuinely empty tile.
    """
    import random
    import requests
    from scripts.extractors.osm_overpass_extract import _build_query, _MIRRORS

    p = tile_path(lat, lon, tiles_dir)
    if os.path.exists(p) and os.path.getsize(p) > 0 and not force:
        return elements_from_overpass(read_tile_raw(p))

    urls = list(mirrors or _MIRRORS)
    random.shuffle(urls)
    query = _build_query(lat, lon)

    # An empty answer is never accepted from a single mirror. Some mirrors serve
    # only one region (overpass.osm.ch covers Switzerland) and answer "200 OK, no
    # elements" for everywhere else, which is indistinguishable from a genuinely
    # empty tile unless a second mirror is asked. Trusting the first empty answer
    # is what produced the defect in Section 6.6, so an empty result here only
    # counts once every mirror has been tried and none returned anything.
    empty_seen = 0
    tried = 0
    for url in urls:
        try:
            r = requests.post(url, data={"data": query}, timeout=180)
            if r.status_code != 200 or r.content[:1] not in (b"{", b"["):
                continue
            elements = r.json().get("elements")
            if elements is None:
                continue
            tried += 1
            if not elements:
                empty_seen += 1
                continue
            write_tile(elements, p)
            return elements_from_overpass(elements)
        except Exception:
            continue

    if tried and empty_seen == tried:
        # every mirror that answered agreed the tile holds nothing
        write_tile([], p)
        return []
    return None


def cached_count(tiles_dir=TILES_DIR):
    return len(glob.glob(os.path.join(tiles_dir, "*.json.gz")))
