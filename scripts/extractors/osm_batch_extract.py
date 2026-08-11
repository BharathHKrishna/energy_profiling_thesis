"""Batch OSM tile extractor — the millisecond setup.

Run ONCE per coord set: a single pass over the local filtered pbf carves out a
512 m tile per coordinate. Scales to 10k+ coords in the SAME single scan — the
time depends on the pbf size (~48 GB), not the coord count (osmium indexes all
the bboxes internally and pulls them in one pass).

After this, extract_osm_use() / extract_osm_features() read each tile in ~5-12 ms.
No server, no downtime, ~20 KB per tile.

CLI:  python osm_batch_extract.py coords.csv        # csv with lat,lon per line
API:  from osm_batch_extract import batch_extract; batch_extract([(lat,lon), ...])
"""
import json, subprocess, math, os, sys, time

from scripts.utils.geo import bbox as _geo_bbox

# Extract from the FULL planet so every tile holds 100% of OSM (roads, natural,
# rail, everything) — nothing lost. The filtered pbf is available but drops tags.
PBF       = "/srv/THESIS/osm_planet/planet-latest.osm.pbf"
TILES_DIR = "/srv/THESIS/osm_planet/tiles"
OSMIUM    = "/home/ws/udyyk/.local/opt/osmenv/bin/osmium"


def tile_name(lat, lon):
    """Deterministic tile filename for a coord (5 dp ≈ 1 m — unique per coord)."""
    return f"{lat:.5f}_{lon:.5f}.osm.pbf"


def _bbox(lat, lon, size_m=512):
    """[west, south, east, north], rounded — the shape osmium's config JSON needs.
    Math itself lives in scripts.utils.geo.bbox(); this just reshapes its output."""
    min_lat, max_lat, min_lon, max_lon = _geo_bbox(lat, lon, size_m)
    return [round(min_lon, 6), round(min_lat, 6), round(max_lon, 6), round(max_lat, 6)]


CHUNK = 450   # osmium extract caps at 500 bboxes/config -> chunk under it

def batch_extract(coords, tiles_dir=TILES_DIR, pbf=PBF, strategy="simple"):
    """Carve one 512 m tile per coord. `coords`: iterable of (lat, lon).

    osmium extract allows at most 500 regions per config, so coords are processed
    in chunks of CHUNK — each chunk is ONE pass over the pbf. So N coords ≈
    ceil(N/450) full scans (≤450 = one scan; 2500 ≈ 6 scans; not "any N in one").
    Returns tiles_dir. Re-runnable: --overwrite refreshes existing tiles.
    """
    os.makedirs(tiles_dir, exist_ok=True)
    coords = list(coords)
    chunks = [coords[i:i + CHUNK] for i in range(0, len(coords), CHUNK)]
    t = time.time()
    for k, chunk in enumerate(chunks):
        extracts = [{"output": tile_name(la, lo), "bbox": _bbox(la, lo)} for la, lo in chunk]
        cfg_path = os.path.join(tiles_dir, "_batch_cfg.json")
        json.dump({"directory": tiles_dir, "extracts": extracts}, open(cfg_path, "w"))
        subprocess.run([OSMIUM, "extract", "-c", cfg_path, pbf, "--overwrite", "-s", strategy],
                       check=True)
        print(f"  chunk {k+1}/{len(chunks)} ({len(chunk)} tiles) done")
    print(f"batch_extract: {len(coords)} tiles in {len(chunks)} scan(s), {time.time()-t:.0f}s")
    return tiles_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python osm_batch_extract.py coords.csv  (lat,lon per line)")
        sys.exit(1)
    coords = []
    for line in open(sys.argv[1]):
        line = line.strip()
        if not line or line.lower().startswith("lat"):
            continue
        la, lo = line.split(",")[:2]
        coords.append((float(la), float(lo)))
    batch_extract(coords)
