"""
Backfill the whole-tile population columns onto an existing feature table.

Added 2026-09-09. GHS-POP cells hold an absolute person count per cell, so the
figure that actually describes a 512 m tile is the overlap-weighted SUM of the
cells it covers, not the mean of them. The existing ghsl_population_per_km2
column is that mean, which is a per-cell count wearing a per-km2 name; the class
gates read it correctly as a per-cell count, so the values were never wrong for
their own purpose, only mislabelled and answering a different question.

This adds two columns and changes none:
    ghsl_population_bbox_total   persons inside the 512 m tile
    ghsl_population_density_km2  the same total over the tile's real area

Reads local rasters only, no network. The original CSV is copied aside first.

Usage:
    python -m scripts.extractors.backfill_bbox_population
    python -m scripts.extractors.backfill_bbox_population --limit 200 --dry-run
"""

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

from scripts.extractors.ghsl_extractor import compute_bbox_population
from scripts.utils.geo import bbox
from scripts.utils.logger import get_logger

logger = get_logger("backfill_bbox_population")

BASE = "/srv/THESIS/energy_profiling_thesis"
FEATURES_CSV = os.path.join(BASE, "outputs/csv/pore_features.csv")
NEW_COLS = ["ghsl_population_bbox_total", "ghsl_population_density_km2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=FEATURES_CSV)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.csv) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if args.limit:
        rows = rows[:args.limit]
    logger.info(f"Loaded {len(rows)} rows from {args.csv}")

    done = missing = 0
    for i, row in enumerate(rows, 1):
        try:
            lat, lon = float(row["lat"]), float(row["lon"])
            size = float(row.get("bbox_size_m") or 512)
        except (TypeError, ValueError):
            row.update({c: "" for c in NEW_COLS})
            missing += 1
            continue

        mn, mx, ln, lx = bbox(lat, lon, size)
        total, density = compute_bbox_population(mn, mx, ln, lx)
        if total is None:
            row.update({c: "" for c in NEW_COLS})
            missing += 1
        else:
            row["ghsl_population_bbox_total"] = total
            row["ghsl_population_density_km2"] = density
            done += 1

        if i % 1000 == 0:
            logger.info(f"{i}/{len(rows)} processed ({done} with data, {missing} without)")

    logger.info(f"COMPLETE: {done} rows with population, {missing} without")

    if args.dry_run:
        logger.info("Dry run, nothing written.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{args.csv}.pre_bbox_pop_{stamp}.bak"
    shutil.copy2(args.csv, backup)
    logger.info(f"Original backed up to {backup}")

    for c in NEW_COLS:
        if c not in fieldnames:
            fieldnames.append(c)

    tmp = args.csv + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, args.csv)
    logger.info(f"Wrote {len(rows)} rows with {len(fieldnames)} columns to {args.csv}")


if __name__ == "__main__":
    main()
