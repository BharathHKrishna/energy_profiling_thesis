"""
Run PORE over a city grid, writing to the case study's own output directory.

This is the second half of the city case study. The first half (city_grid.py)
replaced BORE and SELECT with an exhaustive 512 m sweep; this half changes
nothing at all. It calls the same per-coordinate worker the thesis pipeline uses,
so extraction, the demand formula, the four map renders and the caption all
behave exactly as they do for the 10,000-coordinate dataset. That is the point:
if the outputs differ, the difference is in the sampling, not the measurement.

Only the destinations differ. Features, captions and maps go under
outputs/casestudy/<slug>/, and the two map modules take their directory from an
environment variable for that reason, so the thesis dataset in outputs/csv and
outputs/maps is never touched by a run of this script.

Resume-safe: a coordinate already present in features.csv is skipped, so an
interrupted run continues rather than restarting.

Usage:
    python -m scripts.casestudy.run_city --slug frankfurt --limit 50   # pilot
    python -m scripts.casestudy.run_city --slug frankfurt              # full
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

BASE = "/srv/THESIS/energy_profiling_thesis"
sys.path.insert(0, BASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N tiles only (pilot)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--caption-workers", type=int, default=10)
    ap.add_argument("--no-captions", action="store_true")
    args = ap.parse_args()

    outdir = os.path.join(BASE, "outputs/casestudy", args.slug)
    grid_csv = os.path.join(outdir, "grid.csv")
    feat_csv = os.path.join(outdir, "features.csv")
    caps_json = os.path.join(outdir, "captions.json")

    # Point the map renderers at this run's own folders before any worker imports
    # them; children inherit the environment, so this has to happen up here.
    os.environ["PORE_SEGMAP_DIR"] = os.path.join(outdir, "maps/segmaps")
    os.environ["PORE_DETECTION_DIR"] = os.path.join(outdir, "maps/detection")
    for d in (os.environ["PORE_SEGMAP_DIR"], os.environ["PORE_DETECTION_DIR"]):
        os.makedirs(d, exist_ok=True)

    from run_pipeline import _extract_and_maps_worker, _coord_key

    rows = list(csv.DictReader(open(grid_csv)))
    if args.limit:
        rows = rows[:args.limit]

    done = set()
    if os.path.exists(feat_csv):
        with open(feat_csv) as fh:
            done = {_coord_key(r["lat"], r["lon"]) for r in csv.DictReader(fh)}
    todo = [r for r in rows if _coord_key(r["lat"], r["lon"]) not in done]

    captions = json.load(open(caps_json)) if os.path.exists(caps_json) else {}

    print(f"city grid   : {grid_csv}")
    print(f"tiles       : {len(rows)} ({len(done)} already done, {len(todo)} to do)")
    if not todo:
        print("nothing to do")
        return

    from scripts.captions.kit_caption import generate_caption

    results, failures = [], []
    t0 = time.time()
    cap_pool = ThreadPoolExecutor(max_workers=args.caption_workers)
    cap_futs = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_extract_and_maps_worker, r): r for r in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            row = futs[fut]
            try:
                feats, maps_ok, err = fut.result()
            except Exception as e:
                feats, maps_ok, err = None, False, repr(e)
            if feats is None:
                failures.append((row["stratum_name"], err))
            else:
                results.append(feats)
                if not args.no_captions:
                    cap_futs[cap_pool.submit(generate_caption, feats, 512)] = \
                        _coord_key(feats["lat"], feats["lon"])
            if n % 25 == 0:
                rate = n / (time.time() - t0)
                left = (len(todo) - n) / rate if rate else 0
                print(f"  {n}/{len(todo)}  {rate*60:.1f}/min  ~{left/60:.0f} min left  "
                      f"({len(failures)} failed)", flush=True)

    for fut in as_completed(cap_futs):
        key = cap_futs[fut]
        try:
            captions[key] = fut.result()
        except Exception as e:
            failures.append((key, f"caption: {e!r}"))
    cap_pool.shutdown()

    # features: union of every key seen, so a sparse row keeps the null contract
    # (absent means absent) rather than being padded with zeros
    existing = list(csv.DictReader(open(feat_csv))) if os.path.exists(feat_csv) else []
    allrows = existing + results
    cols = []
    for r in allrows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(feat_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(allrows)

    if not args.no_captions:
        json.dump(captions, open(caps_json, "w"), ensure_ascii=False, indent=1)

    mins = (time.time() - t0) / 60
    print(f"\nCOMPLETE  {len(results)} extracted, {len(captions)} captions, "
          f"{len(failures)} failures, {mins:.1f} min "
          f"({mins*60/max(len(todo),1):.2f} s/tile)")
    for name, err in failures[:10]:
        print(f"  FAIL {name}: {str(err)[:110]}")


if __name__ == "__main__":
    main()
