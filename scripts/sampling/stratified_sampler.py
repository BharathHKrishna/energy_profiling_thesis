import sys
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import os
import json
import time
import numpy as np
import pandas as pd

from scripts.utils.logger import get_logger
from scripts.utils.config_loader import load_config

logger = get_logger("stratified_sampler")
config = load_config()

BBOX_SIZE_M   = config["sampling"]["bbox_size_m"]   # 512
OSM_DELAY     = config["api"]["overpass_request_delay"]  # 1.5 s


# ── Bounding box helper ───────────────────────────────────────────────────────
# Kept for import compatibility with existing notebooks and extractors.

def generate_bbox(lat, lon, size_m=512):
    """
    Generate a square bounding box centred on (lat, lon).
    Default size is 512 m × 512 m (the pipeline standard).
    """
    delta_lat = (size_m / 2) / 111320.0
    delta_lon = (size_m / 2) / (111320.0 * np.cos(np.radians(lat)))
    return {
        "min_lat": lat - delta_lat,
        "max_lat": lat + delta_lat,
        "min_lon": lon - delta_lon,
        "max_lon": lon + delta_lon,
    }


# ── Core sampling logic ───────────────────────────────────────────────────────

def _process_anchor(anchor, union_name, union_config, rng):
    """
    Run the full two-stage screening for one anchor location:
      Stage 1 — WorldCover-only fast score on all 10 candidates
      Stage 2 — Full M3 + 5-criterion score on top 3
      Returns the winning candidate result dict, or None on failure.
    """
    from scripts.sampling.candidate_generator import generate_candidates
    from scripts.sampling.candidate_scorer import score_fast, score_full

    anchor_lat = anchor["lat"]
    anchor_lon = anchor["lon"]
    primary_wc_code   = union_config["primary_wc_code"]
    secondary_wc_code = union_config["secondary_wc_code"]
    confirmation_mode = union_config.get("confirmation_mode", "wc_boundary")

    logger.info(f"  Anchor: {anchor['name']} ({anchor_lat:.4f}, {anchor_lon:.4f})")
    logger.info(f"  Confirmation mode: {confirmation_mode}")

    # ── Stage 1: generate candidates, WorldCover-only score ──────────────────
    candidates = generate_candidates(
        anchor_lat, anchor_lon,
        primary_wc_code, secondary_wc_code,
        rng=rng,
    )

    fast_scores = []
    for c in candidates:
        try:
            fs = score_fast(c, primary_wc_code, secondary_wc_code,
                            confirmation_mode=confirmation_mode)
        except Exception as e:
            logger.warning(f"  Fast score failed: {e}")
            fs = 0.0
        fast_scores.append(fs)
        time.sleep(0.1)  # light throttle between WorldCover tile fetches

    top3_idx = sorted(range(len(fast_scores)),
                      key=lambda i: fast_scores[i], reverse=True)[:3]
    top3 = [candidates[i] for i in top3_idx]

    logger.info(
        f"  Stage 1 top-3 fast scores: "
        f"{[round(fast_scores[i], 3) for i in top3_idx]}"
    )

    # ── Stage 2: full M3 + 5-criterion score on top 3 ────────────────────────
    best_result      = None
    best_total       = -1.0
    confirmed_found  = False   # once a confirmed candidate is found, non-confirmed can't win

    for c in top3:
        try:
            result = score_full(c, primary_wc_code, secondary_wc_code,
                                confirmation_mode=confirmation_mode)
            time.sleep(OSM_DELAY)   # OSM rate limit
        except Exception as e:
            logger.warning(f"  Full score failed: {e}")
            continue

        new_confirmed = result["union_confirmed"]
        new_total     = result["total_score"]

        # A confirmed candidate always beats a non-confirmed one regardless of
        # absolute score. This prevents candidates with high B/C/D but missing
        # union confirmation from stealing the winner slot.
        is_better = (
            (new_confirmed and not confirmed_found) or
            (new_confirmed == confirmed_found and new_total > best_total)
        )
        if is_better:
            best_total      = new_total
            best_result     = result
            confirmed_found = new_confirmed
            best_result["_winner_lat"]  = c["lat"]
            best_result["_winner_lon"]  = c["lon"]
            best_result["_winner_bbox"] = c["bbox"]
            best_result["_direction"]   = c["direction"]

    if best_result is None:
        logger.warning(f"  All Stage-2 scores failed for {anchor['name']}")
        return None

    logger.info(
        f"  Winner: ({best_result['_winner_lat']:.5f}, "
        f"{best_result['_winner_lon']:.5f}) | "
        f"score={best_result['total_score']:.4f} | "
        f"std={best_result['m3']['std']:.4f} | "
        f"direction={best_result['_direction']}"
    )
    return best_result


def run_sampling(output_csv=None, output_json=None, seed=None, anchors=None):
    """
    Main orchestrator. Processes all anchors, runs two-stage screening per
    anchor, and saves results to CSV + JSON.

    Pass anchors= to override the default ANCHORS dict (e.g. test runs).
    Returns list of result dicts.
    """
    if anchors is None:
        from scripts.sampling.union_anchor_regions import ANCHORS
        anchors = ANCHORS

    rng = np.random.default_rng(seed)

    all_results = []
    coord_id    = 1

    for union_name, union_config in anchors.items():
        importance = union_config["importance"]
        logger.info(
            f"\n── Union: {union_name} "
            f"[{importance.upper()}] ──────────────────────────────────"
        )

        for anchor in union_config["anchor_locations"]:
            result = _process_anchor(anchor, union_name, union_config, rng)
            if result is None:
                continue

            m3 = result["m3"]
            row = {
                "id":               coord_id,
                "union":            union_name,
                "importance":       importance,
                "anchor_name":      anchor["name"],
                "lat":              round(result["_winner_lat"], 6),
                "lon":              round(result["_winner_lon"], 6),
                "stratum_primary":  _code_to_name(union_config["primary_wc_code"]),
                "stratum_secondary":_code_to_name(union_config["secondary_wc_code"]),
                "stratum_type":     union_config.get("stratum_type", "union"),
                "direction":        result["_direction"],
                "total_score":      result["total_score"],
                "union_confirmed":  result["union_confirmed"],
                "primary_pct":      result["primary_pct"],
                "secondary_pct":    result["secondary_pct"],
                # M3 polygon
                "polygon_method":   "feature_density" if m3["m3_used"] else "worldcover_dominant",
                "polygon_std":      m3["std"],
                "polygon_coverage_pct": m3["coverage_pct"],
                "low_discrimination":   m3["low_discrimination"],
                "worldcover_only_mode": m3["worldcover_only_mode"],
                # Scoring breakdown
                "score_A_union":    result["criteria_scores"]["A_union_confirmation"],
                "score_B_m3std":    result["criteria_scores"]["B_m3_signal_strength"],
                "score_C_diversity":result["criteria_scores"]["C_feature_diversity"],
                "score_D_osm":      result["criteria_scores"]["D_osm_richness"],
            }

            # Flatten bbox
            bbox = result["_winner_bbox"]
            row.update({
                "bbox_min_lat": round(bbox["min_lat"], 6),
                "bbox_max_lat": round(bbox["max_lat"], 6),
                "bbox_min_lon": round(bbox["min_lon"], 6),
                "bbox_max_lon": round(bbox["max_lon"], 6),
            })

            # Flatten all feature values
            for source, feats in result["features"].items():
                for k, v in feats.items():
                    if not k.startswith("_"):
                        row[k] = v

            # Keep M3 internals for notebook visualisation (not written to CSV/JSON)
            row["_m3_internal"] = result["m3"]

            all_results.append(row)
            coord_id += 1

    logger.info(f"\nSampling complete: {len(all_results)} coordinates generated")

    if output_csv:
        _save_csv(all_results, output_csv)
    if output_json:
        _save_json(all_results, output_json)

    return all_results


def _code_to_name(code):
    from scripts.extractors.worldcover_extractor import WORLDCOVER_CLASSES
    return WORLDCOVER_CLASSES.get(int(code), f"class_{code}")


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    logger.info(f"CSV saved: {path} ({len(df)} rows)")
    return df


def _save_json(results, path):
    """Save as nested JSON with sub-polygon coordinates."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    output = []
    for row in results:
        obj = {
            "coordinate_id":    row["id"],
            "lat":              row["lat"],
            "lon":              row["lon"],
            "union":            row["union"],
            "importance":       row["importance"],
            "anchor_name":      row["anchor_name"],
            "stratum_primary":  row["stratum_primary"],
            "stratum_secondary":row["stratum_secondary"],
            "stratum_type":     row["stratum_type"],
            "bbox": {
                "min_lat": row["bbox_min_lat"],
                "max_lat": row["bbox_max_lat"],
                "min_lon": row["bbox_min_lon"],
                "max_lon": row["bbox_max_lon"],
            },
            "sub_polygon": {
                "method":            row["polygon_method"],
                "std":               row["polygon_std"],
                "coverage_pct":      row["polygon_coverage_pct"],
                "low_discrimination":row["low_discrimination"],
                "worldcover_only_mode": row["worldcover_only_mode"],
                # polygon_coords stored separately below
            },
            "scoring": {
                "total_score":     row["total_score"],
                "union_confirmed": row["union_confirmed"],
                "direction":       row["direction"],
                "criteria":        {
                    k: row[k] for k in [
                        "score_A_union", "score_B_m3std",
                        "score_C_diversity", "score_D_osm", "score_E_contrast",
                    ]
                },
            },
        }
        output.append(obj)

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"JSON saved: {path} ({len(output)} records)")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = "/srv/THESIS/energy_profiling_thesis"
    results = run_sampling(
        output_csv  = f"{BASE}/outputs/csv/sampler_demo_6coords.csv",
        output_json = f"{BASE}/outputs/csv/sampler_demo_6coords.json",
        seed        = 42,
    )
    print(f"\n=== Sampling complete: {len(results)} coordinates ===")
    for r in results:
        print(
            f"  [{r['id']:02d}] {r['union']:35s} | "
            f"({r['lat']:.4f}, {r['lon']:.4f}) | "
            f"score={r['total_score']:.3f} | "
            f"std={r['polygon_std']:.4f}"
        )
