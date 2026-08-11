"""
Microsoft Global ML Building Footprints extractor.

Source: https://github.com/microsoft/GlobalMLBuildingFootprints (2026-02-03 release)
Format: GeoJSON-L (one Feature per line), organised by region + quadkey.

Approach:
  1. Load the dataset index CSV (cached locally after first download).
  2. For a given bbox, compute the zoom-9 quadkey covering the centre.
  3. Download matching .csv.gz files (cached on disk per quadkey).
  4. Filter features whose polygon intersects or is contained by the bbox.
  5. Return (lon, lat) polygon rings ready for matplotlib.

Buildings are ML-derived from satellite imagery — they align with what is
visible in ESRI World Imagery far better than volunteer-traced OSM polygons.
"""

import sys, os, gzip, io, json, math, requests, csv
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import mercantile
from scripts.utils.logger import get_logger

logger = get_logger("msft_buildings")

INDEX_URL  = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
CACHE_DIR  = "/srv/THESIS/energy_profiling_thesis/cache/msft_buildings"
INDEX_PATH = os.path.join(CACHE_DIR, "dataset-links.csv")

os.makedirs(CACHE_DIR, exist_ok=True)


# ── Index ─────────────────────────────────────────────────────────────────────

def _load_index():
    """Return index as list of (location, quadkey, url) tuples. Cached to disk."""
    if not os.path.exists(INDEX_PATH):
        logger.info("Downloading Microsoft Buildings index...")
        r = requests.get(INDEX_URL, timeout=30)
        r.raise_for_status()
        with open(INDEX_PATH, "w") as f:
            f.write(r.text)
        logger.info(f"Index saved: {INDEX_PATH}")

    entries = []
    with open(INDEX_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append((row["Location"], row["QuadKey"], row["Url"]))
    return entries


_INDEX = None

def _get_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = _load_index()
    return _INDEX


# ── Tile download ─────────────────────────────────────────────────────────────

def _tile_cache_path(quadkey, location):
    safe_loc = location.replace(" ", "_").replace("/", "-")
    return os.path.join(CACHE_DIR, f"{safe_loc}_{quadkey}.jsonl")


def _download_tile(url, cache_path):
    """Download .csv.gz, decompress, save as .jsonl, return lines."""
    logger.info(f"Downloading {url.split('/')[-1]} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with gzip.open(io.BytesIO(r.content)) as f:
        data = f.read().decode("utf-8")
    with open(cache_path, "w") as f:
        f.write(data)
    return data.split("\n")


def _get_tile_lines(location, quadkey, url):
    """Return raw GeoJSON-L lines for this tile, using disk cache."""
    path = _tile_cache_path(quadkey, location)
    if os.path.exists(path):
        with open(path) as f:
            return f.read().split("\n")
    return _download_tile(url, path)


# ── Spatial filter ────────────────────────────────────────────────────────────

def _poly_in_bbox(coords, min_lon, min_lat, max_lon, max_lat):
    """True if any polygon vertex is inside or on the bbox."""
    for lon, lat in coords:
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return True
    return False


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_msft_buildings(min_lat, max_lat, min_lon, max_lon):
    """
    Fetch ML building footprints for a bbox.

    Returns list of (lon, lat) polygon rings for matplotlib.
    Falls back to empty list on any failure.
    """
    centre_lat = (min_lat + max_lat) / 2
    centre_lon = (min_lon + max_lon) / 2

    try:
        tile = mercantile.tile(centre_lon, centre_lat, 9)
        qk   = mercantile.quadkey(tile)
    except Exception as e:
        logger.warning(f"Quadkey computation failed: {e}")
        return []

    try:
        index = _get_index()
    except Exception as e:
        logger.warning(f"Index load failed: {e}")
        return []

    # All index entries whose quadkey matches (may span multiple regions)
    matching = [(loc, q, url) for loc, q, url in index if q == qk]
    if not matching:
        logger.warning(f"No Microsoft Buildings tile found for quadkey {qk}")
        return []

    # Download + filter all matching tiles
    rings = []
    for loc, q, url in matching:
        try:
            lines = _get_tile_lines(loc, q, url)
            count = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    feat   = json.loads(line)
                    coords = feat["geometry"]["coordinates"][0]  # outer ring
                    if _poly_in_bbox(coords, min_lon, min_lat, max_lon, max_lat):
                        rings.append(coords)   # already (lon, lat) per GeoJSON spec
                        count += 1
                except Exception:
                    continue
            logger.info(f"Microsoft Buildings [{loc} / {q}]: {count} buildings in bbox")
        except Exception as e:
            logger.warning(f"Tile download/parse failed [{loc} / {q}]: {e}")

    return rings
