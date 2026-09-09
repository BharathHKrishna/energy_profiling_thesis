"""
Recompute every GHSL-derived and demand column after the Mollweide geometry fix.

Added 2026-09-09. Until that fix, each GHSL read projected the bbox's four
corners into Mollweide and took min/max. Mollweide shears a lat/lon rectangle,
so the axis-aligned box around it covered up to ~1.9x the intended ground area,
worst far from the prime meridian. Everything read through that window was
therefore taken over too large an area: the mean-of-cells features, the
overlap-weighted floor area, and, because demand = floor area x intensity, the
heating and cooling demand as well.

Rewrites in place (originals backed up first):
    ghsl_built_surface_m2        mean built surface per cell
    ghsl_building_height_m       mean building height
    ghsl_population_per_km2      mean of GHS-POP cells (per-cell count)
    ghsl_population_bbox_total   persons inside the tile
    ghsl_population_density_km2  that total over the tile's real area
    demand_floor_m2              overlap-weighted whole-tile floor area
    demand_regime                unchanged in logic, recomputed for consistency
    heating_MWh / cooling_MWh    floor area x intensity, one or the other

Climate columns are read from the CSV, not re-fetched, so this runs entirely on
local rasters with no network calls.

Usage:
    python -m scripts.extractors.recompute_ghsl_demand --limit 200 --dry-run
    python -m scripts.extractors.recompute_ghsl_demand
"""

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

from scripts.extractors.ghsl_extractor import extract_ghsl_features, compute_floor_area
from scripts.extractors.demand_extractor import demand_from_features
from scripts.utils.geo import bbox
from scripts.utils.logger import get_logger

logger = get_logger("recompute_ghsl_demand")

BASE = "/srv/THESIS/energy_profiling_thesis"
FEATURES_CSV = os.path.join(BASE, "outputs/csv/pore_features.csv")

GHSL_COLS = ["ghsl_built_surface_m2", "ghsl_building_height_m",
             "ghsl_population_per_km2", "ghsl_population_bbox_total"]
DEMAND_COLS = ["demand_floor_m2", "demand_regime", "heating_MWh", "cooling_MWh"]


def fnum(row, col):
    v = row.get(col, "")
    if v is None or v == "" or v == "None":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


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
    logger.info(f"Loaded {len(rows)} rows")

    for c in GHSL_COLS + DEMAND_COLS:
        if c not in fieldnames:
            fieldnames.append(c)

    stats = {"ghsl": 0, "floor": 0, "demand": 0, "no_floor": 0}
    floor_ratio = []

    for i, row in enumerate(rows, 1):
        lat, lon = fnum(row, "lat"), fnum(row, "lon")
        if lat is None or lon is None:
            continue
        size = fnum(row, "bbox_size_m") or 512.0
        mn, mx, ln, lx = bbox(lat, lon, size)

        # ── GHSL features ────────────────────────────────────────────────────
        old_floor = fnum(row, "demand_floor_m2")
        feats = extract_ghsl_features(lat, lon, mn, mx, ln, lx)
        for c in GHSL_COLS:
            row[c] = feats.get(c, "")
        if feats:
            stats["ghsl"] += 1

        # ── Floor area and demand ────────────────────────────────────────────
        floor = compute_floor_area(mn, mx, ln, lx)
        for c in DEMAND_COLS:
            row[c] = ""
        if floor is None:
            stats["no_floor"] += 1
        else:
            stats["floor"] += 1
            if old_floor and old_floor > 0:
                floor_ratio.append(floor / old_floor)
            dem = demand_from_features(
                floor,
                fnum(row, "climate_hdd"),
                fnum(row, "climate_cdd24"),
                lat, lon,
                fnum(row, "climate_mean_temp_c"),
            )
            for k, v in dem.items():
                row[k] = v
            if dem:
                stats["demand"] += 1

        if i % 500 == 0:
            logger.info(f"{i}/{len(rows)} processed")

    if floor_ratio:
        floor_ratio.sort()
        q = lambda f: floor_ratio[int(f * (len(floor_ratio) - 1))]
        logger.info(f"floor area new/old: p10={q(.1):.3f} median={q(.5):.3f} "
                    f"p90={q(.9):.3f} (n={len(floor_ratio)})")
    logger.info(f"GHSL populated {stats['ghsl']}, floor area {stats['floor']}, "
                f"demand {stats['demand']}, no floor area {stats['no_floor']}")

    if args.dry_run:
        logger.info("Dry run, nothing written.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{args.csv}.pre_mollweide_fix_{stamp}.bak"
    shutil.copy2(args.csv, backup)
    logger.info(f"Backed up to {backup}")

    tmp = args.csv + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, args.csv)
    logger.info(f"Wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
