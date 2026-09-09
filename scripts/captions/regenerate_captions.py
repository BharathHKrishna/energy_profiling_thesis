"""
Regenerate every caption from the existing feature table, without re-running
extraction or map rendering.

Written 2026-09-09 for the categorical-caption change. The captions produced by
the full run quote figures directly from the CSV row and include Global Solar
Atlas values; both are being replaced (see caption_prompt.py). Re-running the
whole pipeline to get new captions would mean re-hitting GEE, Overpass and the
GHSL rasters for 10,000 coordinates, roughly ten hours of work to change text
that depends on nothing but the CSV that already exists.

Reads outputs/csv/pore_features.csv, writes outputs/captions/pore_captions.json,
keyed exactly as run_pipeline._coord_key does so the two stay interchangeable.
The previous captions file is copied aside before anything is written.

Usage:
    python -m scripts.captions.regenerate_captions               # all 10,000
    python -m scripts.captions.regenerate_captions --limit 20    # smoke test
    python -m scripts.captions.regenerate_captions --resume      # keep existing
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

from scripts.captions.kit_caption import generate_caption
from scripts.utils.logger import get_logger

logger = get_logger("regenerate_captions")

BASE = "/srv/THESIS/energy_profiling_thesis"
FEATURES_CSV = os.path.join(BASE, "outputs/csv/pore_features.csv")
CAPTIONS_JSON = os.path.join(BASE, "outputs/captions/pore_captions.json")
BBOX_M = 512
WORKERS = 10          # same thread-pool size the live pipeline uses for captions
SAVE_EVERY = 50


def coord_key(lat, lon):
    """Must match run_pipeline._coord_key exactly."""
    return f"{round(float(lat), 6)},{round(float(lon), 6)}"


def load_rows():
    """CSV row -> feature dict, empty strings dropped so the null contract holds
    (an absent key means no data; it must never arrive as '' or NaN)."""
    rows = []
    with open(FEATURES_CSV) as fh:
        for raw in csv.DictReader(fh):
            feat = {}
            for k, v in raw.items():
                if v is None or v == "" or v == "None":
                    continue
                try:
                    f = float(v)
                    if math.isnan(f):
                        continue
                    feat[k] = f
                except (TypeError, ValueError):
                    feat[k] = v
            if "lat" in feat and "lon" in feat:
                rows.append(feat)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only caption the first N coordinates (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="keep captions already present instead of regenerating them")
    ap.add_argument("--out", default=CAPTIONS_JSON)
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    rows = load_rows()
    if args.limit:
        rows = rows[:args.limit]
    logger.info(f"Loaded {len(rows)} coordinates from {FEATURES_CSV}")

    captions = {}
    if os.path.exists(args.out):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{args.out}.pre_categorical_{stamp}.bak"
        shutil.copy2(args.out, backup)
        logger.info(f"Existing captions backed up to {backup}")
        if args.resume:
            with open(args.out) as fh:
                captions = json.load(fh)
            logger.info(f"Resuming: {len(captions)} captions already present")

    todo = [r for r in rows if coord_key(r["lat"], r["lon"]) not in captions]
    logger.info(f"To generate: {len(todo)}")
    if not todo:
        logger.info("Nothing to do.")
        return

    lock = threading.Lock()
    done = {"n": 0, "fail": 0}
    t0 = time.time()

    def save():
        tmp = args.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(captions, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, args.out)

    def work(row):
        key = coord_key(row["lat"], row["lon"])
        return key, generate_caption(row, BBOX_M)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, r): r for r in todo}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                key, caption = fut.result()
                with lock:
                    captions[key] = caption
                    done["n"] += 1
            except Exception as e:
                with lock:
                    done["fail"] += 1
                logger.error(f"FAILED {coord_key(row['lat'], row['lon'])}: {e}")
            with lock:
                n = done["n"] + done["fail"]
                if n % SAVE_EVERY == 0:
                    save()
                    rate = n / max(time.time() - t0, 1e-9)
                    left = (len(todo) - n) / rate if rate else 0
                    logger.info(f"{n}/{len(todo)} done ({done['fail']} failed), "
                                f"{rate*60:.1f}/min, ~{left/60:.0f} min left")

    save()
    logger.info(f"COMPLETE: {done['n']} captions written, {done['fail']} failed, "
                f"{len(captions)} total in {args.out}, "
                f"elapsed {(time.time()-t0)/60:.1f} min")
    if done["fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
