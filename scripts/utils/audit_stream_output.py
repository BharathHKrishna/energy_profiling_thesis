"""Post-stream completeness audit.

Checks every coordinate `select` targeted (outputs/csv/filtered_strata_sample_selected.csv)
against what `stream` actually produced: a row in pore_features.csv, all 4 map files, a
caption in pore_captions.json. Prints a per-coordinate gap list instead of relying on
scrolling through run logs to notice a silent failure — built after the 18-coordinate smoke
test silently lost all 4 maps for 2 strata with nothing but a one-line log warning.

Also flags features rows with empty/NaN values in key fields (a row existing is not the same
as a row being complete — added after a NaN heating_MWh/cooling_MWh went unnoticed since
"has_features" only checked the row existed, not what was in it). This is reported
separately from the pass/fail completeness tally above, not folded into it: an empty field
can be a legitimate null-contract result (e.g. no ERA5-Land climate coverage over water), not
necessarily a bug — the point is to make it visible so a human can judge, not to auto-fail it.

Read-only — makes no changes, safe to run at any time, doesn't require the pipeline to be
mid-run or finished.

CLI: python scripts/utils/audit_stream_output.py
"""
import os, sys, csv, json

# Fields worth flagging when empty — the ones downstream consumers (captions, notebooks,
# demand analysis) actually depend on. Not every one of the 50+ feature columns is checked:
# many are legitimately sparse/optional per null-contract convention and flagging all of them
# would just be noise.
KEY_FIELDS = [
    "heating_MWh", "cooling_MWh", "demand_regime", "demand_score", "demand_tier",
    "ghsl_built_surface_m2", "ghsl_building_height_m", "ghsl_population_per_km2",
    "viirs_ntl_nw_cm2_sr", "raw_element_count",
]

BASE = "/srv/THESIS/energy_profiling_thesis"
SELECTED_CSV  = f"{BASE}/outputs/csv/filtered_strata_sample_selected.csv"
FEATURES_CSV  = f"{BASE}/outputs/csv/pore_features.csv"
CAPTIONS_JSON = f"{BASE}/outputs/captions/pore_captions.json"
SEGMAPS_DIR   = f"{BASE}/outputs/maps/segmaps"
DETECTION_DIR = f"{BASE}/outputs/maps/detection"

MAP_FILES = {
    "segmap":     (SEGMAPS_DIR,   "{s}_segmap.png"),
    "pop_segmap": (SEGMAPS_DIR,   "{s}_pop_segmap.png"),
    "detection":  (DETECTION_DIR, "{s}_detection.png"),
    "pop_det":    (DETECTION_DIR, "{s}_pop_det.png"),
}


def slug(name: str) -> str:
    return (name.replace(" ", "_").replace("/", "-")
                .replace("+", "plus").replace("(", "").replace(")", ""))


def audit():
    if not os.path.exists(SELECTED_CSV):
        print(f"No selected coordinates yet: {SELECTED_CSV}")
        return []

    with open(SELECTED_CSV, newline="") as f:
        selected = list(csv.DictReader(f))

    feat_by_stratum = {}
    if os.path.exists(FEATURES_CSV):
        with open(FEATURES_CSV, newline="") as f:
            for row in csv.DictReader(f):
                feat_by_stratum[row["stratum_name"]] = row

    caps = {}
    if os.path.exists(CAPTIONS_JSON):
        with open(CAPTIONS_JSON) as f:
            caps = json.load(f)

    rows = []
    for r in selected:
        name = r["stratum_name"]
        s = slug(name)
        missing_maps = [
            label for label, (d, tmpl) in MAP_FILES.items()
            if not os.path.exists(os.path.join(d, tmpl.format(s=s)))
        ]
        feat_row = feat_by_stratum.get(name)
        empty_fields = [
            f for f in KEY_FIELDS
            if feat_row is not None and (feat_row.get(f, "") or "").strip() == ""
        ]
        rows.append({
            "stratum_name": name,
            "has_features": feat_row is not None,
            "missing_maps": missing_maps,
            "has_caption": name in caps and len(caps[name]) > 20,
            "empty_fields": empty_fields,
        })
    return rows


def report(rows):
    n = len(rows)
    if n == 0:
        return
    n_feat = sum(r["has_features"] for r in rows)
    n_maps = sum(not r["missing_maps"] for r in rows)
    n_cap  = sum(r["has_caption"] for r in rows)

    print(f"{n} coordinate(s) targeted by select\n")
    print(f"  features : {n_feat}/{n}")
    print(f"  maps     : {n_maps}/{n}  (all 4 present)")
    print(f"  captions : {n_cap}/{n}")

    stragglers = [r for r in rows
                  if not (r["has_features"] and not r["missing_maps"] and r["has_caption"])]
    if not stragglers:
        print("\nAll complete.")
    else:
        print(f"\n{len(stragglers)} incomplete:")
        for r in stragglers:
            gaps = []
            if not r["has_features"]:
                gaps.append("features")
            if r["missing_maps"]:
                gaps.append(f"maps({','.join(r['missing_maps'])})")
            if not r["has_caption"]:
                gaps.append("caption")
            print(f"  {r['stratum_name']:<35} missing: {', '.join(gaps)}")

    # Separate from the pass/fail tally above — an empty key field can be a legitimate
    # null-contract result (e.g. no ERA5-Land coverage), not necessarily a bug. Surfaced
    # so a human can judge, not auto-failed.
    flagged = [r for r in rows if r["empty_fields"]]
    if flagged:
        print(f"\n{len(flagged)} row(s) with empty key field(s) (may be legitimate — check):")
        for r in flagged:
            print(f"  {r['stratum_name']:<35} empty: {', '.join(r['empty_fields'])}")


if __name__ == "__main__":
    report(audit())
