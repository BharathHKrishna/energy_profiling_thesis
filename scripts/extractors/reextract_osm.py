"""
Re-extract only the OpenStreetMap columns of an existing feature table.

Nothing else is touched. The land-cover, GHSL, VIIRS, MODIS, climate and demand
columns are copied through byte for byte; only raw_element_count and the osm_*
fields are recomputed. That keeps this run independent of every other correction
made to the dataset, and means it can be judged on its own.

The reason it is needed is set out in osm_json_tiles.py: tiles were written to
PBF and read back with `osmium export`, which requires ID-ordered input and
silently drops every way when it does not get it. Buildings, land use, waterways,
power lines and roof attributes are all ways. This path never touches a PBF; it
reads the Overpass JSON directly.

A tile that cannot be retrieved leaves its row's OSM columns exactly as they were
rather than blanking them, so a failed fetch never destroys data. A tile every
mirror agrees is empty is recorded as genuinely empty.

Usage:
    python -m scripts.extractors.reextract_osm --limit 100 --sample --dry-run
    python -m scripts.extractors.reextract_osm --workers 8
"""

import argparse
import csv
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

BASE = "/srv/THESIS/energy_profiling_thesis"
sys.path.insert(0, BASE)

FEATURES_CSV = os.path.join(BASE, "outputs/csv/pore_features.csv")

OSM_COLS = ["osm_building_count", "osm_building_type", "osm_building_start_date",
            "osm_roof_shape", "osm_roof_material", "osm_roof_colour",
            "osm_power_plant", "osm_power_substation", "osm_power_line",
            "osm_power_tower", "osm_generator_source", "osm_plant_source",
            "osm_landuse", "osm_waterway", "osm_amenity"]


def work(row):
    """One coordinate. Returns (key, osm_dict, n_elements) or (key, None, None)."""
    from scripts.extractors.osm_json_tiles import fetch_tile
    from scripts.extractors.osm_extractor import parse_osm_response
    lat, lon = float(row["lat"]), float(row["lon"])
    key = (row["lat"], row["lon"])
    try:
        els = fetch_tile(lat, lon)
    except Exception:
        els = None
    if els is None:
        return key, None, None          # not retrieved: leave the row alone
    feats = parse_osm_response(els)
    feats = {k: v for k, v in feats.items()
             if v is not None or k == "osm_building_count"}
    return key, feats, len(els)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=FEATURES_CSV)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", action="store_true",
                    help="draw the limit at random across the whole table, not the first N")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import logging
    logging.disable(logging.INFO)

    with open(args.csv) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        fieldnames = list(rd.fieldnames)

    targets = rows
    if args.limit:
        targets = random.Random(0).sample(rows, args.limit) if args.sample else rows[:args.limit]

    print(f"table   : {len(rows)} rows")
    print(f"targets : {len(targets)}")

    before_b = sum(1 for r in targets if (r.get("osm_building_count") or "0") not in ("", "0"))
    updates, failed, empty = {}, 0, 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, r) for r in targets]
        for n, fut in enumerate(as_completed(futs), 1):
            key, feats, nel = fut.result()
            if feats is None:
                failed += 1
            else:
                updates[key] = (feats, nel)
                if nel == 0:
                    empty += 1
            if n % 20 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(targets)}  {rate:.2f}/s  "
                      f"{failed} unretrieved  {empty} empty", flush=True)

    gained = sum(1 for k, (f, _) in updates.items() if f.get("osm_building_count", 0) > 0)
    tot_b = sum(f.get("osm_building_count", 0) for f, _ in updates.values())
    secs = time.time() - t0

    print(f"\nretrieved      : {len(updates)}/{len(targets)}")
    print(f"unretrieved    : {failed}")
    print(f"genuinely empty: {empty}")
    print(f"with buildings : {gained}  (was {before_b} before)")
    print(f"buildings total: {tot_b}")
    print(f"rate           : {secs/max(len(targets),1):.2f} s/tile"
          f"   -> {len(rows)*secs/max(len(targets),1)/3600:.1f} h for all {len(rows)}")

    if args.dry_run:
        print("\nDry run, nothing written.")
        return

    for c in OSM_COLS + ["raw_element_count"]:
        if c not in fieldnames:
            fieldnames.append(c)

    for r in rows:
        got = updates.get((r["lat"], r["lon"]))
        if not got:
            continue
        feats, nel = got
        for c in OSM_COLS:
            r[c] = feats.get(c, "")
        r["raw_element_count"] = nel

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(args.csv, f"{args.csv}.pre_osm_refix_{stamp}.bak")
    tmp = args.csv + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, args.csv)
    print(f"\nwrote {len(rows)} rows, {len(updates)} OSM columns updated")


if __name__ == "__main__":
    main()
