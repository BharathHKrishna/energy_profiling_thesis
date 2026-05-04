#!/usr/bin/env python3
"""
n25_full_validation.py — Run ALL 18 strata at N=25, pick best anchor per stratum.

Writes raw pool to: outputs/csv/n25_full_validation.csv
Writes best-per-stratum to: outputs/csv/filtered_strata_sample.csv (overwrites)

"Best" = highest primary_pct among the 25 accepted rows.
"""

import csv
import random
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from scripts.sampling.real_anchor_finder import find_anchors, RATE_DELAY, RNG_SEED
from scripts.sampling.coordinate_filter import (
    parse_esa_classes, parse_importance_tiers, build_strata_config,
    FEATURES_HTML, BREAKDOWN_HTML,
)

N_PER_STRATUM = 25
RAW_CSV       = BASE / "outputs" / "csv" / "n25_full_validation.csv"
BEST_CSV      = BASE / "outputs" / "csv" / "filtered_strata_sample.csv"

FIELDNAMES = [
    "stratum_name", "importance_tier", "strata_type",
    "location_name", "lat", "lon",
    "primary_class", "primary_pct",
    "secondary_class", "secondary_pct",
    "bbox_m", "remark",
]


def main() -> None:
    esa_classes   = parse_esa_classes(FEATURES_HTML)
    imp_map       = parse_importance_tiers(BREAKDOWN_HTML)
    imp_lower     = {k.lower(): v for k, v in imp_map.items()}
    strata_config = build_strata_config(esa_classes)

    print(f"Full N={N_PER_STRATUM} validation — {len(strata_config)} strata\n")

    all_rows: list[dict] = []
    pass_rates: list[tuple] = []

    for idx, sc in enumerate(strata_config):
        tier = imp_lower.get(sc["name"].lower(), "MID")
        stratum_rng = random.Random(RNG_SEED + idx)
        found = find_anchors(sc, tier, stratum_rng, N_PER_STRATUM, [])
        all_rows.extend(found)

        last = found[-1]["remark"] if found else "tries=0"
        tried = int(last.split("tries=")[-1]) if "tries=" in last else 0
        pr = len(found) / tried * 100 if tried else 0
        pass_rates.append((sc["name"], len(found), tried, pr))
        if len(found) < N_PER_STRATUM:
            print(f"  [WARN] {sc['name']}: only {len(found)}/{N_PER_STRATUM}")
        time.sleep(RATE_DELAY)

    # Write raw pool
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[RAW] {len(all_rows)} rows → {RAW_CSV}")

    # Per-stratum anchor selection rules:
    # SECONDARY_BEST: select by highest secondary_pct
    #   Industrial+Forest:        tree_cover primary → pick highest built_up (sawmill dominant)
    #   Agrivoltaics:             cropland primary → pick highest built_up (solar array visible)
    #   Mangrove+Industrial:      mangrove primary → pick highest built_up (industrial visible)
    #   Industrial+Arid:          bare_sparse primary → pick highest built_up (infra visible,
    #                             not pure desert where 1.6% built_up is invisible at zoom 15)
    #   Utility-scale Solar Farm: bare_sparse primary → pick highest built_up (panel coverage)
    #   Coastal+Agricultural:     cropland primary → pick highest water (open sea visible,
    #                             not inland ponds — 74% water = clear coastal character)
    #   Coastal+Solar-Wind:       water primary → pick highest grassland (coastal land visible,
    #                             75% grassland = tidal flat / meadow with turbines clearly seen)
    # MINIMUM_PRIMARY: select by LOWEST primary_pct
    #   Hydropower Reservoir: primary=water → lowest water% = dam wall prominent, not pure reservoir
    SECONDARY_BEST = {
        "Industrial + Forest",
        "Agrivoltaics (Solar + Farmland)",
        "Mangrove + Industrial",
        "Industrial + Arid",           # highest built_up% — infra visible in desert
        "Utility-scale Solar Farm",    # highest built_up% — most panel coverage
        "Coastal + Agricultural",      # highest water% — open sea not inland pond
        "Coastal + Solar-Wind Hybrid", # highest grassland% — coastal land strip visible
    }
    MINIMUM_PRIMARY = {"Hydropower Reservoir"}

    best: dict[str, dict] = {}
    for row in all_rows:
        name = row["stratum_name"]
        try:
            if name in SECONDARY_BEST:
                pct = float(row.get("secondary_pct") or 0)
            else:
                pct = float(row["primary_pct"])
        except (ValueError, TypeError):
            pct = 0.0

        if name not in best:
            best[name] = row
        elif name in MINIMUM_PRIMARY:
            cur = float(best[name]["primary_pct"])
            if pct < cur:
                best[name] = row
        elif name in SECONDARY_BEST:
            cur = float(best[name].get("secondary_pct") or 0)
            if pct > cur:
                best[name] = row
        else:
            if pct > float(best[name]["primary_pct"]):
                best[name] = row

    best_rows = [best[sc["name"]] for sc in strata_config if sc["name"] in best]
    with open(BEST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(best_rows)
    print(f"[BEST] {len(best_rows)} rows → {BEST_CSV}")

    print("\n── Final pass-rate + best primary_pct ─────────────────────────────────────")
    print(f"  {'Stratum':<40} {'Got':>4}  {'Tried':>5}  {'Pass%':>6}  {'BestPct':>8}")
    print(f"  {'-'*40}  {'-'*4}  {'-'*5}  {'-'*6}  {'-'*8}")
    for sc in strata_config:
        name = sc["name"]
        row_best = best.get(name)
        best_pct = float(row_best["primary_pct"]) if row_best else 0.0
        stats = next((x for x in pass_rates if x[0] == name), None)
        if stats:
            _, got, tried, pr = stats
            print(f"  {name:<40} {got:>4}  {tried:>5}  {pr:>5.1f}%  {best_pct:>7.1f}%")


if __name__ == "__main__":
    main()
