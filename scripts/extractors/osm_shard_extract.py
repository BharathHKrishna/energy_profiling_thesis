"""Shard-aware batch OSM tile extractor.

Same contract as osm_batch_extract.batch_extract() (same tile filename
convention, same output directory, same osmium `-s simple` strategy) but
reads from the pre-built regional shards in /srv/THESIS/osm_planet/shards/
instead of the full 88GB planet file. Each shard is a 10x10-degree cell
carved once (see logs/shard_build.log) — a coordinate's tile always comes
from exactly one shard since the tile bbox (512m) is minuscule next to a
10-degree cell.

Cost win: a `simple`-strategy extract's time scales with (file_size x
bboxes_per_pass), confirmed empirically (18 vs 450 bboxes: 194MB/s vs
8MB/s throughput on the full planet). Median shard is ~35MB vs 88GB
whole-planet — several orders of magnitude cheaper per pass for the
median coordinate.
"""
import json, math, os, subprocess, time
from collections import defaultdict

from scripts.extractors.osm_batch_extract import tile_name, _bbox, CHUNK, OSMIUM

SHARDS_DIR = "/srv/THESIS/osm_planet/shards"
CELLS_JSON = os.path.join(SHARDS_DIR, "_cells.json")


def _shard_for(lat, lon):
    lat_bin = int(math.floor(lat / 10.0) * 10)
    lon_bin = int(math.floor(lon / 10.0) * 10)
    return f"shard_{lat_bin}_{lon_bin}.osm.pbf", (lat_bin, lon_bin)


def shard_batch_extract(coords, tiles_dir, strategy="simple"):
    """Carve one 512m tile per coord, reading from the smallest shard that
    contains it instead of the full planet. `coords`: iterable of (lat, lon).
    Returns (tiles_dir, missing) — missing is a list of (lat, lon, shard_name)
    for coords whose shard file doesn't exist or is empty (falls outside all
    206 occupied cells — should not happen for real BORE-discovered coords,
    but checked rather than assumed)."""
    os.makedirs(tiles_dir, exist_ok=True)
    coords = list(coords)

    by_shard = defaultdict(list)
    missing = []
    for la, lo in coords:
        shard_name, cell = _shard_for(la, lo)
        shard_path = os.path.join(SHARDS_DIR, shard_name)
        if not os.path.exists(shard_path) or os.path.getsize(shard_path) == 0:
            missing.append((la, lo, shard_name))
            continue
        by_shard[shard_path].append((la, lo))

    t = time.time()
    n_passes = 0
    for shard_path, shard_coords in by_shard.items():
        chunks = [shard_coords[i:i + CHUNK] for i in range(0, len(shard_coords), CHUNK)]
        for chunk in chunks:
            extracts = [{"output": tile_name(la, lo), "bbox": _bbox(la, lo)} for la, lo in chunk]
            cfg_path = os.path.join(tiles_dir, "_shard_batch_cfg.json")
            json.dump({"directory": tiles_dir, "extracts": extracts}, open(cfg_path, "w"))
            subprocess.run([OSMIUM, "extract", "-c", cfg_path, shard_path, "--overwrite", "-s", strategy],
                           check=True)
            n_passes += 1

    print(f"shard_batch_extract: {len(coords)} coord(s), {len(by_shard)} shard(s), "
          f"{n_passes} pass(es), {len(missing)} missing, {time.time()-t:.1f}s")
    return tiles_dir, missing
