"""
DEPRECATED (2026-08-10) — superseded by run_pipeline.py's `stream` stage.

Do not run this directly: it reads straight from filtered_strata_sample.csv (BORE's full
raw pool), bypassing the `select` stage entirely (no tier-balanced subsampling) and the
`osm_carve` stage (no pre-carved tiles, every OSM read falls back to a live ~15-20 min
full-planet scan per coordinate). It also does extraction only — no maps, no captions,
no process-pool parallelism. Kept only as a reference for what per-coordinate extraction
looked like before the 4-stage bore/select/osm_carve/stream architecture existed.

Use `python run_pipeline.py --stages stream` instead.

── Original docstring ──
PORE runner — Per-coordinate feature extraction for BORE-verified anchors.

Input:  outputs/csv/filtered_strata_sample.csv
Output: outputs/csv/pore_features.csv

Resume-safe: skips coordinates already present in output CSV (checked by lat+lon).
"""
import sys
import os
import time

sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import pandas as pd
from scripts.utils.logger import get_logger
from scripts.pore.feature_extractor import extract_all_features, _bbox
from scripts.extractors.osm_extractor import circuit_breaker_wait, extract_osm_features

logger = get_logger("run_pore")

BASE        = "/srv/THESIS/energy_profiling_thesis"
INPUT_CSV   = os.path.join(BASE, "outputs/csv/filtered_strata_sample.csv")
OUTPUT_CSV  = os.path.join(BASE, "outputs/csv/pore_features.csv")
LOG_EVERY   = 5
OSM_DELAY_S = 1.5  # polite delay between coordinates (OSM rate limit)


def _already_done(output_csv):
    if not os.path.exists(output_csv):
        return set()
    try:
        df = pd.read_csv(output_csv)
        return set(zip(df["lat"].round(6), df["lon"].round(6)))
    except Exception:
        return set()


def run(input_csv=INPUT_CSV):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    df_in = pd.read_csv(input_csv)
    logger.info(f"PORE: {len(df_in)} coordinates from {input_csv}")

    done      = _already_done(OUTPUT_CSV)
    total     = len(df_in)
    processed = 0
    skipped   = 0

    # Load already-done rows so we can merge them into the final write
    existing_rows = []
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_rows = pd.read_csv(OUTPUT_CSV).to_dict("records")
        except Exception:
            existing_rows = []

    new_rows = []

    for i, row in df_in.iterrows():
        lat = round(float(row["lat"]), 6)
        lon = round(float(row["lon"]), 6)
        key = (lat, lon)

        if key in done:
            logger.info(f"[{i+1}/{total}] Skip (done): {row['stratum_name']}")
            skipped += 1
            continue

        logger.info(f"\n[{i+1}/{total}] ── {row['stratum_name']} ({lat}, {lon})")

        # Wait if OSM circuit breaker is still open from a previous failure
        circuit_breaker_wait()

        features = extract_all_features(
            lat=lat,
            lon=lon,
            stratum_name=str(row["stratum_name"]),
            importance_tier=str(row.get("importance_tier", "")),
            strata_type=str(row.get("strata_type", "")),
        )

        new_rows.append(features)
        done.add(key)
        processed += 1

        if processed % LOG_EVERY == 0:
            logger.info(f"── Progress: {processed} processed, {skipped} skipped ──")

        time.sleep(OSM_DELAY_S)

    # Write all rows together — pandas aligns columns, fills missing with NaN
    all_rows = existing_rows + new_rows
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        df_out.to_csv(OUTPUT_CSV, index=False)

    logger.info(
        f"\nPORE complete: {processed} processed, {skipped} skipped → {OUTPUT_CSV}"
    )

    # ── OSM retry pass ────────────────────────────────────────────────────────
    # Re-run OSM only for coords where raw_element_count=0 due to circuit breaker.
    # Waits for circuit breaker expiry before each retry.
    _retry_osm_misses(OUTPUT_CSV)

    return OUTPUT_CSV


def _retry_osm_misses(output_csv):
    if not os.path.exists(output_csv):
        return

    df = pd.read_csv(output_csv)
    osm_cols = [c for c in df.columns if c.startswith('osm_')]
    missed   = df[df['raw_element_count'] == 0]

    if missed.empty:
        logger.info("OSM retry pass: no misses found.")
        return

    logger.info(f"\nOSM retry pass: {len(missed)} coords with raw_element_count=0")
    updated = False

    for idx, row in missed.iterrows():
        lat = float(row['lat'])
        lon = float(row['lon'])
        logger.info(f"  Retrying OSM for {row['stratum_name']} ({lat}, {lon})")

        circuit_breaker_wait()
        time.sleep(OSM_DELAY_S)

        min_lat, max_lat, min_lon, max_lon = _bbox(lat, lon)
        try:
            osm = extract_osm_features(lat, lon, min_lat, max_lat, min_lon, max_lon)
            if osm.get('raw_element_count', 0) > 0:
                for k, v in osm.items():
                    df.at[idx, k] = v
                logger.info(f"  -> {osm['raw_element_count']} elements retrieved")
                updated = True
            else:
                logger.info(f"  -> Still 0 elements (genuinely sparse or server still slow)")
        except Exception as e:
            logger.warning(f"  -> OSM retry failed: {e}")

    if updated:
        df.to_csv(output_csv, index=False)
        logger.info("OSM retry pass: CSV updated.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=INPUT_CSV)
    args = p.parse_args()
    run(input_csv=args.input)
