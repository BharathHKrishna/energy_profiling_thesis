"""
Final check on a full map re-render: every image must be newer than the run that
produced it, and every coordinate must have all four.

The reason the timestamp matters is the race described in rerender_maps.py. An
earlier pass wrote all four images under the stratum name alone, so two workers
handling different coordinates of the same class raced for one filename and one
tile's image was renamed under the other's coordinate. Those files are still on
disk until the corrected run overwrites them. A file older than the corrected
run's start time is therefore a file that may still be mislabelled, and this
reports it rather than assuming the overwrite happened.

Usage:
    python -m scripts.pore.verify_maps --since "2026-09-10 18:34"
"""

import argparse
import csv
import datetime as dt
import os
import re

BASE = "/srv/THESIS/energy_profiling_thesis"
MAPS = os.path.join(BASE, "outputs/maps")
PAT = re.compile(r"_(-?\d+\.\d{5})_(-?\d+\.\d{5})_(segmap|pop_segmap|detection|pop_det)\.png$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(BASE, "outputs/csv/pore_features.csv"))
    ap.add_argument("--since", required=True, help='e.g. "2026-09-10 18:34"')
    args = ap.parse_args()

    cutoff = dt.datetime.strptime(args.since, "%Y-%m-%d %H:%M").timestamp()
    want = {"%.5f_%.5f" % (float(r["lat"]), float(r["lon"]))
            for r in csv.DictReader(open(args.csv))}

    seen, stale, orphan = {}, [], []
    for sub in ("segmaps", "detection"):
        d = os.path.join(MAPS, sub)
        for f in os.listdir(d):
            m = PAT.search(f)
            if not m:
                orphan.append(f)
                continue
            key = f"{m.group(1)}_{m.group(2)}"
            if key not in want:
                orphan.append(f)
                continue
            seen.setdefault(key, set()).add(m.group(3))
            if os.path.getmtime(os.path.join(d, f)) < cutoff:
                stale.append(f)

    complete = sum(1 for v in seen.values() if len(v) == 4)
    print(f"coordinates in table      : {len(want)}")
    print(f"with all four maps        : {complete}")
    print(f"with an incomplete set    : {sum(1 for v in seen.values() if len(v) != 4)}")
    print(f"with no maps at all       : {len(want - set(seen))}")
    print(f"files not in the table    : {len(orphan)}")
    print(f"files older than the run  : {len(stale)}")
    for f in stale[:10]:
        print("   stale:", f)
    ok = (complete == len(want) and not orphan and not stale)
    print("\nRESULT:", "all maps current and complete" if ok else "NEEDS ATTENTION")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
