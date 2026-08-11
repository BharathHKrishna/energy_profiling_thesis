"""Energy-demand score — synthetic 0-100 tiering index, single-cell.

    base  = sqrt(built_m2) * viirs^0.2 * max(height_m, 4)^0.3 / 5
    score = softceil(base * osm_gate, knee=72, span=28, tau=37)
    tier  = HIGH >=65, MID-HIGH >=45, MID >=25, LOW <25

Ported from notebooks/energy_demand_formula.ipynb. Deliberately single-cell, NOT the
notebook's validated 9-cell (3x3 neighborhood) mean — see
docs/demand_work_summary_2026-07-31.md §7 for why (BORE's candidate-pool cost model
doesn't apply here since this runs once per already-verified coordinate, but a true
9-mean would still mean 8 extra neighboring-cell GHSL/VIIRS/OSM-gate fetches per
coordinate; single-cell was chosen to avoid that for the first pass). This makes
demand_score a different, unvalidated number from the notebook's own 15-city table.

osm_gate comes from scripts/extractors/osm_offline.py's extract_osm_use() — real
landuse-dominance + power-generation-override logic, not the hardcoded gate values
the notebook uses internally (it doesn't import project code).

Not statistically validated against real data the way heat/cool are (see
demand_extractor.py) — no agency publishes an "energy-demand score" to fit a
synthetic 0-100 index against. The notebook's closest real-world check is a
GDP-per-capita vs electricity-per-capita World Bank panel (R2=0.74), a proxy for
"does urban/economic intensity predict energy use", not a validation of this
literal built/VIIRS/height formula.
"""
import sys, math
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

from scripts.extractors.osm_offline import extract_osm_use

KNEE, SPAN, TAU = 72.0, 28.0, 37.0
MIN_HEIGHT_M = 4.0   # floor for the height term only — distinct from
                     # demand_extractor.py's STOREY_H=3.0, a separately calibrated formula


def _softceil(x, knee=KNEE, span=SPAN, tau=TAU):
    return x if x <= knee else knee + span * (1 - math.exp(-(x - knee) / tau))


def _base_score(built_m2, viirs, height_m):
    if built_m2 <= 0 or height_m <= 0:
        return 0.0
    return (built_m2 ** 0.5) * (viirs ** 0.2) * (max(height_m, MIN_HEIGHT_M) ** 0.3) / 5.0


def _tier(score):
    if score >= 65:
        return "HIGH"
    if score >= 45:
        return "MID-HIGH"
    if score >= 25:
        return "MID"
    return "LOW"


def demand_score_from_features(built_m2, height_m, viirs, osm_gate):
    """Score + tier from already-extracted values and an already-computed OSM gate
    multiplier — no OSM read here. For the OSM-read version, see demand_score_for_coord.

    Null contract: needs built_m2 (GHSL) and viirs both present to mean anything. A
    real reading of built=0/viirs~0 legitimately scores near 0 (matches the notebook's
    own Sahara/Amazon rows) — this returns {} only when the underlying data is
    genuinely missing, not when it's a real low-activity area.
    """
    if built_m2 is None or viirs is None:
        return {}
    height_m = height_m or MIN_HEIGHT_M
    osm_gate = osm_gate if osm_gate is not None else 1.0
    score = round(_softceil(_base_score(built_m2, viirs, height_m) * osm_gate), 1)
    return dict(demand_score=score, demand_tier=_tier(score))


def demand_score_for_coord(lat, lon, built_m2, height_m, viirs):
    """Pipeline entry point — takes already-fetched built/height/viirs (from
    feature_extractor.py's GHSL/VIIRS steps) but does its own OSM-gate lookup via
    extract_osm_use() (a local offline OSM tile read — a second, separate OSM read
    for this coordinate on top of extract_osm_features()'s earlier one in the same
    pass; cheap now that OSM is fully offline, per project decision).
    """
    if built_m2 is None or viirs is None:
        return {}
    gate_info = extract_osm_use(lat, lon)
    return demand_score_from_features(built_m2, height_m, viirs, gate_info.get("osm_use_multiplier"))


if __name__ == "__main__":
    # pure-math check, no OSM/network — reproduces the notebook's own single-pin
    # Chicago figure (gate=1.0 "none"): expect demand_score=91.5, tier=HIGH
    print("Chicago (gate=1.0):", demand_score_from_features(5076.9, 26.93, 254.6, 1.0))
    # Singapore demo coordinate — notebook's own bad-pin example (nature reserve,
    # zero built, near-zero VIIRS): expect a real near-zero score, not {}
    print("Singapore nature reserve (gate=1.0):", demand_score_from_features(0.0, 0.0, 5.6, 1.0))
    print("missing GHSL:", demand_score_from_features(None, 8.0, 50.0, 1.0))
