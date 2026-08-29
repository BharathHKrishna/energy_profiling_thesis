"""
Energy Profiling Pipeline — unified runner

════════════════════════════════════════
  THE ONLY THING YOU EVER CHANGE:
  N_PER_STRATUM below — one number per stratum.
  Then run:  python run_pipeline.py
  All outputs are overwritten fresh each run.
════════════════════════════════════════

Stages (run in order):
  1. bore      → BORE coordinate sampling                → outputs/csv/filtered_strata_sample.csv
                 N_PER_STRATUM below is a SEARCH quota fed to real_anchor_finder.py.
  2. select    → subsample BORE's discovered set down to exactly N_PER_STRATUM coordinates
                 per stratum (the SAME dial as stage 1) → outputs/csv/filtered_strata_sample_selected.csv.
                 Seeded random sample per stratum (RNG_SEED, same convention as
                 real_anchor_finder.py) — NOT first-N-in-file-order, since discovery order
                 clusters by phase/region. If a stratum has fewer discovered anchors than its
                 target, takes all available and logs a warning rather than failing.
                 Set N_PER_STRATUM to 1 for a quick smoke test through the full pipeline, or
                 to the real target plan for a production run — same code path either way.
  3. stream    → per-coordinate pipeline, N_WORKERS coords in flight at once:
                   OSM tile fetch (live Overpass) + extract (6 sources + demand) + 4 maps
                                                                                    [process pool]
                   → caption                                            [thread pool]
                 A coordinate's caption fires as soon as ITS OWN extract+maps finish —
                 it does NOT wait for the other coordinates in the batch. This replaces
                 the old features → maps → captions stage barriers. osm_carve no longer
                 exists as its own upfront batch stage (merged 2026-08-24) -- each worker
                 fetches its own coordinate's OSM tile via the live Overpass API (multiple
                 public mirrors, tried in randomized order) as the first step of its own
                 per-coordinate work, instead of reading a tile a separate stage
                 pre-carved from a local planet file.

Advanced usage:
    python run_pipeline.py --stages bore select         # run specific stages only
    python run_pipeline.py --skip bore                  # skip a stage already done
    python run_pipeline.py --status                     # check what's already done
    python run_pipeline.py --workers 16                 # override process-pool size
    python run_pipeline.py --caption-workers 10          # override caption thread-pool size
"""
import sys, os, argparse, time, math, threading, csv as _csv, json as _json
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

from scripts.utils.logger import get_logger
logger = get_logger("run_pipeline")

# ════════════════════════════════════════════════════════════════════════════
# ▼▼▼  CHANGE THESE NUMBERS — nothing else  ▼▼▼
# This ONE dial drives both `bore` (search quota) and `select` (how many of the found
# anchors per stratum get used downstream) — same numbers, same code path, whether it's
# 1 (smoke test) or the real target plan. Resume-safe: if filtered_strata_sample.csv already
# has more than a given stratum's quota here, BORE skips that stratum's search entirely.
# Real production target plan (16 strata, 10,000 total) -- HIGH tier 1600 each (4 classes),
# MID tier 500 each (7 classes), LOW tier 20 each (5 classes). "Agriculture / Agrivoltaics"
# is the final class name (renamed 2026-08-27 from "Agrivoltaics (Solar + Farmland)" after
# a visual reliability audit found the class is genuinely mostly real farmland, but the
# "solar" half of the old name was badly overclaiming what's actually visible).
N_PER_STRATUM: dict = {
    "Suburban":                        1600,
    "Dense Urban":                     1600,
    "Informal + Urban":                1600,
    "Industrial":                      1600,
    "Suburban + Agricultural":         500,
    "Industrial + Water":              500,
    "Hydropower Reservoir":            500,
    "Agriculture / Agrivoltaics":      500,
    "Industrial + Arid":               500,
    "Industrial + Forest":             500,
    "Urban + Coastal":                 500,
    "Agricultural + Water":            20,
    "Mangrove + Industrial":           20,
    "Data Centre + Industrial":        20,
    "Coastal + Solar-Wind Hybrid":     20,
    "Coastal + Agricultural":          20,
}
# ▲▲▲  CHANGE THESE NUMBERS — nothing else  ▲▲▲
# ════════════════════════════════════════════════════════════════════════════

N_WORKERS       = 16   # process-pool size for extract+maps (CPU/IO heavy: rasterio, GEE,
                        # matplotlib, OSM). Raised 4 -> 8 -> 16 after live evidence the
                        # machine sat mostly idle at lower worker counts (each coordinate
                        # is dominated by network wait, not CPU), matching this machine's
                        # 16 physical cores.
CAPTION_WORKERS = 10   # thread-pool size for captions (network I/O only — KIT)
                        # bumped from 6 -> 10 same day; kit_caption.py backs off 10s/20s/...
                        # specifically on detected 429s instead of a flat 3s retry, so higher
                        # concurrency degrades gracefully into longer waits instead of losing captions
RNG_SEED        = 809  # select-stage sampling seed, changed again 2026-08-14 (42 -> 777 ->
                        # 555 -> 314 -> 271 -> 809), this time to force a genuinely fresh draw
                        # so this run actually exercises the new compute_floor_area() formula

BASE          = "/srv/THESIS/energy_profiling_thesis"
COORDS_CSV    = os.path.join(BASE, "outputs/csv/filtered_strata_sample.csv")
SELECTED_CSV  = os.path.join(BASE, "outputs/csv/filtered_strata_sample_selected.csv")
FEATURES_CSV  = os.path.join(BASE, "outputs/csv/pore_features.csv")
SEGMAPS_DIR   = os.path.join(BASE, "outputs/maps/segmaps")
DETECTION_DIR = os.path.join(BASE, "outputs/maps/detection")
CAPTIONS_JSON = os.path.join(BASE, "outputs/captions/pore_captions.json")

ALL_STAGES = ["bore", "select", "stream"]


def _clean_row(row: dict) -> dict:
    """NaN -> None. Needed only for rows loaded back from pore_features.csv via
    pandas (empty cells become float('nan'), not the null-contract's absent key) —
    freshly-extracted dicts from extract_all_features() don't have this problem."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


def _coord_key(lat, lon):
    """Coordinate-based key for captions/resume lookups, not stratum name -- a real
    bug (fixed) had captions/maps keyed by class name instead of coordinate, so at
    N>1-per-class scale only the first coordinate to finish in each class ever
    actually got captioned; every other coordinate in that class was silently
    skipped since the class name was already "done"."""
    return f"{round(float(lat), 6)},{round(float(lon), 6)}"


# ── Module-level worker function (must be top-level for multiprocessing) ──────

def _extract_and_maps_worker(row):
    """One coordinate: fetch its own OSM tile via live Overpass (osm_carve no
    longer exists as a separate stage, see module docstring), extract all
    features (6 sources + demand) and render all 4 maps, sharing one OSM fetch.
    Runs in a worker PROCESS -- everything here is CPU/IO heavy (rasterio, GEE,
    matplotlib, an OSM fetch), unlike captions (network-only), which stays in
    the caller's thread pool instead.

    Returns (features_dict_or_None, maps_ok, error_or_None).
    """
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

    try:
        from scripts.pore.feature_extractor import extract_all_features
        from scripts.extractors.osm_extractor import circuit_breaker_wait
        from scripts.pore.segmap_generator import (
            generate_single, fetch_osm_elements, _bbox, render_ghsl_segmap, fetch_esri_img
        )
        from scripts.pore.detection_map import render_detection_panels, render_ghsl_det
        from scripts.extractors.worldcover_extractor import fetch_worldcover_pixels
        from scripts.extractors.msft_buildings_extractor import fetch_msft_buildings
        from scripts.extractors.osm_overpass_extract import fetch_tile_any_mirror, TILES_DIR as _TILES_DIR
        from scripts.extractors.osm_batch_extract import tile_name as _tile_name
        import numpy as _np
        import time as _time

        name = str(row.get("stratum_name", ""))
        lat  = float(row["lat"])
        lon  = float(row["lon"])

        # OSM tile fetch -- osm_carve used to be a separate upfront batch step between
        # select and stream; merged 2026-08-24 now that OSM comes from Overpass (cheap
        # per-coordinate) rather than the old local-planet batch scan (which needed
        # batching to amortize one huge fixed per-pass file-scan cost; that reason for
        # a separate stage no longer applies). Lets carve and the other extraction
        # sources (GEE etc.) overlap in wall-clock time instead of running as two fully
        # sequential all-or-nothing stages.
        _tile_path = os.path.join(_TILES_DIR, _tile_name(lat, lon))
        # size check, not just existence -- osmium's own batch config used to create every
        # output file as a 0-byte placeholder immediately at start, before any real
        # data is written; a killed/interrupted earlier carve run leaves these stale
        # empty files behind, and treating "exists" as "already carved" silently
        # read them as genuinely-empty tiles (found live 2026-08-24, N=250 test --
        # 885 stale 0-byte tiles left over from earlier tonight's killed local carve).
        if not (os.path.exists(_tile_path) and os.path.getsize(_tile_path) > 0):
            if not fetch_tile_any_mirror(lat, lon):
                logger.warning(f"[{name}] OSM carve failed on every mirror for ({lat},{lon}) "
                               "-- proceeding, features/maps will show sparse/no OSM data")

        circuit_breaker_wait()
        features = extract_all_features(
            lat=lat, lon=lon, stratum_name=name,
            importance_tier=str(row.get("importance_tier", "")),
            strata_type=str(row.get("strata_type", "")),
        )

        maps_ok, maps_err = True, None
        try:
            min_lat, max_lat, min_lon, max_lon = _bbox(lat, lon)
            elements = fetch_osm_elements(name, min_lat, max_lat, min_lon, max_lon, lat=lat, lon=lon)
            # Always attempt all 4 — an empty OSM fetch is a valid (if sparse) result,
            # not a failure: generate_single()/render_ghsl_segmap() don't use `elements`
            # at all, and render_detection_panels()/render_ghsl_det() handle elements=[]
            # fine (zero OSM overlay, still draw Microsoft Buildings independently).

            # Fetch the shared, bbox-scoped inputs ONCE and pass them to all 4 maps —
            # each of these used to get re-fetched independently by 2-4 of the map calls
            # below for identical data (same ESRI tile, same WorldCover pixels, same MS
            # Buildings query).
            try:
                esri_img = fetch_esri_img(min_lon, min_lat, max_lon, max_lat)
            except Exception as e:
                logger.warning(f"[{name}] ESRI fetch failed: {e}")
                esri_img = _np.zeros((512, 512, 3), dtype=_np.uint8)
            wc_pixels_raw, _, wc_ok = fetch_worldcover_pixels(min_lat, max_lat, min_lon, max_lon, lat, lon)
            wc_pixels = (wc_pixels_raw, wc_ok)
            try:
                ms_rings = fetch_msft_buildings(min_lat, max_lat, min_lon, max_lon)
            except Exception as e:
                logger.warning(f"[{name}] MS buildings fetch failed: {e}")
                ms_rings = []

            generate_single(lat, lon, stratum_name=name, esri_img=esri_img, wc_pixels=wc_pixels)
            render_ghsl_segmap(name, lat, lon, variant="pop", esri_img=esri_img, wc_pixels=wc_pixels)
            render_detection_panels(name, lat, lon, elements=elements, esri_img=esri_img, ms_rings=ms_rings)
            render_ghsl_det(name, lat, lon, variant="pop", elements=elements, esri_img=esri_img, ms_rings=ms_rings)
            if not elements:
                logger.warning(f"[{name}] OSM fetch returned empty — maps rendered "
                               "without OSM overlay (base imagery / MS Buildings still used)")
        except Exception as e:
            maps_ok, maps_err = False, str(e)

        return (features, maps_ok, maps_err)
    except Exception as e:
        return (None, False, str(e))


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_bore():
    logger.info("=" * 60)
    logger.info("STAGE 1 — BORE: coordinate sampling")
    logger.info("=" * 60)
    import scripts.bore.real_anchor_finder as raf
    raf.N_PER_STRATUM = N_PER_STRATUM
    logger.info(f"N_PER_STRATUM = {N_PER_STRATUM}  (total: {sum(N_PER_STRATUM.values())})")
    from scripts.bore.real_anchor_finder import main as bore_main
    bore_main()
    logger.info(f"BORE done → {COORDS_CSV}")


def run_select():
    """Subsample BORE's discovered set down to exactly N_PER_STRATUM coordinates per
    stratum — the same dial used for the bore stage above, so bumping N_PER_STRATUM later
    (e.g. smoke test → real target plan) drives both stages identically with no separate
    config to keep in sync.

    Per stratum: a seeded random sample (RNG_SEED) of exactly the target count — NOT the
    first N rows in file order, since BORE's discovery order clusters by phase (OSM chunk →
    densify grid → GHSL-polygon seed) and by region within each phase, so file order isn't
    geographically representative. If a stratum has fewer discovered anchors than its
    target, takes all available and logs a warning rather than failing.
    """
    logger.info("=" * 60)
    logger.info("STAGE — SELECT: subsample to N_PER_STRATUM coordinates per stratum")
    logger.info("=" * 60)
    if not os.path.exists(COORDS_CSV):
        raise FileNotFoundError(f"Coordinates CSV not found: {COORDS_CSV}\n"
                                "Run BORE first:  python run_pipeline.py --stages bore")

    import pandas as pd, random

    targets = dict(N_PER_STRATUM)
    total_target = sum(targets.values())
    logger.info(f"Target plan: {len(targets)} strata, {total_target} coordinate(s) total (N_PER_STRATUM)")

    df = pd.read_csv(COORDS_CSV)
    rng = random.Random(RNG_SEED)

    picked = []
    for stratum, target in targets.items():
        pool = df[df["stratum_name"] == stratum]
        available = len(pool)
        if available == 0:
            logger.warning(f"[select] {stratum}: 0 available in {COORDS_CSV} — skipping")
            continue
        if available < target:
            logger.warning(f"[select] {stratum}: fewer discovered than target — taking all available")
            picked.append(pool)
        else:
            idx = sorted(rng.sample(range(available), target))
            picked.append(pool.iloc[idx])
        logger.info(f"[select] {stratum}: {min(available, target)}/{target} selected")

    if not picked:
        raise RuntimeError("Select produced zero rows — check stratum names in "
                           f"{COORDS_CSV} match N_PER_STRATUM exactly")

    out = pd.concat(picked, ignore_index=True)
    os.makedirs(os.path.dirname(SELECTED_CSV), exist_ok=True)
    out.to_csv(SELECTED_CSV, index=False)
    logger.info(f"Select done — {len(out)}/{total_target} coordinates → {SELECTED_CSV}")


def run_stream(n_workers=N_WORKERS, caption_workers=CAPTION_WORKERS):
    """extract+maps (process pool) → caption (thread pool), per coordinate.

    A coordinate's caption is submitted the moment ITS OWN extract+maps future
    completes — via as_completed() inside the loop below, not after the whole
    batch finishes. Coordinates whose features already exist from a previous run
    (resume) still get checked for a missing caption, submitted up front so they
    don't wait on this run's new extractions either.
    """
    logger.info("=" * 60)
    logger.info(f"STAGE — PORE streaming: extract+maps ({n_workers} processes) "
               f"→ captions ({caption_workers} threads)")
    logger.info("=" * 60)
    if not os.path.exists(SELECTED_CSV):
        raise FileNotFoundError(f"Selected coordinates CSV not found: {SELECTED_CSV}\n"
                                "Run select first:  python run_pipeline.py --stages bore select")

    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    from dotenv import load_dotenv
    # Switched 2026-08-24 from Groq to KIT's KI-Toolbox (see kit_caption.py docstring)
    # -- Groq's account-level daily token cap (200,000/day) makes 10k captions take
    # ~38 days regardless of in-process pacing; KIT has no such documented hard cap.
    # Groq removed entirely 2026-08-27 (groq_caption.py deleted; its one still-needed
    # piece, build_prompt(), lives in caption_prompt.py now).
    from scripts.captions.kit_caption import generate_caption

    load_dotenv(os.path.join(BASE, ".env"))

    df_in = pd.read_csv(SELECTED_CSV)
    logger.info(f"Loaded {len(df_in)} coordinates")

    # ── prune stale entries from a PREVIOUS select draw ─────────────────────────
    selected_keys = set(zip(df_in["lat"].round(6), df_in["lon"].round(6)))
    stale_strata = set()
    if os.path.exists(FEATURES_CSV):
        try:
            df_old = pd.read_csv(FEATURES_CSV)
            is_stale = ~df_old.apply(
                lambda r: (round(float(r["lat"]), 6), round(float(r["lon"]), 6)) in selected_keys,
                axis=1)
            if is_stale.any():
                stale_strata = set(df_old.loc[is_stale, "stratum_name"])
                df_old[~is_stale].to_csv(FEATURES_CSV, index=False)
                logger.info(f"Pruned {int(is_stale.sum())} stale feature row(s) no longer in "
                           f"the selected set: {sorted(stale_strata)}")
        except Exception as e:
            logger.warning(f"Stale-row prune skipped (features CSV unreadable): {e}")
    if os.path.exists(CAPTIONS_JSON) and os.path.exists(FEATURES_CSV):
        # Captions are keyed by coordinate now, not stratum name, so pruning them
        # against the still-current features file (by coordinate) rather than the
        # stale-stratum-name list above is the correct, coordinate-accurate check.
        try:
            with open(CAPTIONS_JSON) as f:
                _caps = _json.load(f)
            with open(FEATURES_CSV) as f:
                current_keys = {_coord_key(r["lat"], r["lon"]) for r in _csv.DictReader(f)}
            removed = [k for k in list(_caps) if k not in current_keys]
            if removed:
                for k in removed:
                    _caps.pop(k, None)
                with open(CAPTIONS_JSON, "w") as f:
                    _json.dump(_caps, f, indent=2, ensure_ascii=False)
                logger.info(f"Pruned {len(removed)} stale caption(s) no longer in the selected set")
        except Exception as e:
            logger.warning(f"Stale-caption prune skipped: {e}")

    existing_rows = []
    done_features = set()
    if os.path.exists(FEATURES_CSV):
        try:
            df_done = pd.read_csv(FEATURES_CSV)
            done_features = set(zip(df_done["lat"].round(6), df_done["lon"].round(6)))
            existing_rows = [_clean_row(r) for r in df_done.to_dict("records")]
        except Exception:
            pass

    tasks = [row.to_dict() for _, row in df_in.iterrows()
             if (round(float(row["lat"]), 6), round(float(row["lon"]), 6)) not in done_features]
    logger.info(f"{len(tasks)} coord(s) need extraction+maps, "
               f"{len(existing_rows)} already done (will still check for a missing caption)")

    os.makedirs(SEGMAPS_DIR, exist_ok=True)
    os.makedirs(DETECTION_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(CAPTIONS_JSON), exist_ok=True)

    # ── incremental CSV write (row-at-a-time, resume-safe) ─────────────────────
    csv_lock = threading.Lock()
    csv_header_written = os.path.exists(FEATURES_CSV) and os.path.getsize(FEATURES_CSV) > 0
    feature_fieldnames = None
    if csv_header_written:
        with open(FEATURES_CSV) as f:
            feature_fieldnames = next(_csv.reader(f))

    def _append_feature_row(feat: dict):
        nonlocal csv_header_written, feature_fieldnames
        with csv_lock:
            if feature_fieldnames is None:
                feature_fieldnames = list(feat.keys())
            else:
                for k in feat:
                    if k not in feature_fieldnames:
                        feature_fieldnames.append(k)
            write_header = not csv_header_written
            with open(FEATURES_CSV, "a", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=feature_fieldnames, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                    csv_header_written = True
                w.writerow(feat)

    # ── incremental captions JSON (rewritten whole file each time — small, cheap) ─
    json_lock = threading.Lock()
    captions = {}
    if os.path.exists(CAPTIONS_JSON):
        try:
            with open(CAPTIONS_JSON) as f:
                captions = _json.load(f)
        except Exception:
            captions = {}

    def _save_caption(coord_key: str, caption: str):
        with json_lock:
            captions[coord_key] = caption
            with open(CAPTIONS_JSON, "w") as f:
                _json.dump(captions, f, indent=2, ensure_ascii=False)

    caption_pool = ThreadPoolExecutor(max_workers=caption_workers)
    caption_futures = {}
    n_cap_done = [0]  # mutable cell, safe to read/increment from callbacks under json_lock

    def _on_caption_done(fut):
        # Saves the instant THIS ONE caption finishes, not batched until every
        # coordinate's extraction is done (fixed 2026-08-27) -- at 10k-coordinate
        # scale, the run can take hours, and a crash/kill at any point before the
        # old end-of-run save loop used to mean every caption generated so far,
        # real, already-paid-for KIT calls, was lost, even though features were
        # already safe on disk (those write incrementally). Now a caption is only
        # ever at risk for the few seconds between its own completion and this
        # callback running.
        coord_key, stratum = caption_futures[fut]
        try:
            caption = fut.result()
            _save_caption(coord_key, caption)
            with json_lock:
                n_cap_done[0] += 1
                done_now = n_cap_done[0]
            logger.info(f"[captions] {done_now}/{len(caption_futures)} {stratum} ({coord_key}) — done")
        except Exception as e:
            logger.warning(f"[captions] {stratum} ({coord_key}) failed: {e}")

    def _submit_caption(feat: dict):
        coord_key = _coord_key(feat["lat"], feat["lon"])
        if coord_key in captions:
            return
        bbox_m = int(float(feat.get("bbox_size_m", 512) or 512))
        fut = caption_pool.submit(generate_caption, feat, bbox_m)
        caption_futures[fut] = (coord_key, feat.get("stratum_name", coord_key))
        fut.add_done_callback(_on_caption_done)

    # coords that already had features from a previous run — check for a missing
    # caption up front, don't make them wait on this run's new extractions
    for feat in existing_rows:
        _submit_caption(feat)

    # extract+maps for new coords — each one streams straight to its caption the
    # moment IT finishes, not after the whole `tasks` batch is done
    n_done = n_failed = 0
    if tasks:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_extract_and_maps_worker, t): t for t in tasks}
            for future in as_completed(futures):
                row = futures[future]
                features, maps_ok, maps_err = future.result()
                if features is not None:
                    _append_feature_row(features)
                    n_done += 1
                    tail = "" if maps_ok else f"  [maps failed: {maps_err}]"
                    logger.info(f"[extract+maps] ✓ {features.get('stratum_name')} "
                               f"({n_done}/{len(tasks)}){tail}")
                    _submit_caption(features)
                else:
                    n_failed += 1
                    logger.warning(f"[extract+maps] ✗ {row.get('stratum_name')}: {maps_err}")

    logger.info(f"Extract+maps done — {n_done} new, {n_failed} failed, "
               f"{len(existing_rows)} pre-existing")
    logger.info(f"Waiting on {len(caption_futures)} caption job(s)...")

    # _on_caption_done (attached via add_done_callback in _submit_caption) already
    # saved and logged each caption the instant IT finished — this just blocks
    # until every outstanding one has, so the function doesn't return early.
    for _ in as_completed(list(caption_futures)):
        pass
    caption_pool.shutdown(wait=True)

    # ── normalize FEATURES_CSV header ───────────────────────────────────────────
    # _append_feature_row above grows feature_fieldnames as later coordinates
    # introduce keys earlier ones didn't have (expected — the null-contract
    # convention means different strata legitimately populate different optional
    # keys), but the on-disk header is only written ONCE, from the first row's
    # narrower key set. Later rows silently outgrow it — pandas then fails with
    # "Expected N fields, saw M" (found 2026-08-10 while reviewing this run's
    # output). Each row's width is always a prefix of feature_fieldnames in
    # write order (it only ever grows by appending), so this is a safe rewrite:
    # re-pad every row to the final column count and replace the stale header.
    if os.path.exists(FEATURES_CSV) and feature_fieldnames:
        with open(FEATURES_CSV, newline="") as f:
            rows = list(_csv.reader(f))
        if rows and len(rows[0]) < len(feature_fieldnames):
            old_width, body = len(rows[0]), rows[1:]
            with open(FEATURES_CSV, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(feature_fieldnames)
                for row in body:
                    row = row[:len(feature_fieldnames)] + [""] * (len(feature_fieldnames) - len(row))
                    w.writerow(row)
            logger.info(f"Normalized {FEATURES_CSV} header ({old_width} → {len(feature_fieldnames)} columns)")

    n_seg = len([f for f in os.listdir(SEGMAPS_DIR) if f.endswith(".png")]) if os.path.isdir(SEGMAPS_DIR) else 0
    n_det = len([f for f in os.listdir(DETECTION_DIR) if f.endswith(".png")]) if os.path.isdir(DETECTION_DIR) else 0
    logger.info(f"Stream stage complete — {n_done + len(existing_rows)} featured, "
               f"{n_seg} segmaps, {n_det} detection maps, {n_cap_done[0]} new captions")


# ── Output status check ───────────────────────────────────────────────────────

def _status():
    checks = [
        ("bore",      COORDS_CSV,    "outputs/csv/filtered_strata_sample.csv",          "rows"),
        ("select",    SELECTED_CSV,  "outputs/csv/filtered_strata_sample_selected.csv", "rows"),
        ("features",  FEATURES_CSV,  "outputs/csv/pore_features.csv",          "rows"),
        ("maps/seg",  SEGMAPS_DIR,   "outputs/maps/segmaps/",                  "segmaps"),
        ("maps/det",  DETECTION_DIR, "outputs/maps/detection/",                "maps"),
        ("captions",  CAPTIONS_JSON, "outputs/captions/pore_captions.json",    ""),
    ]
    logger.info("Pipeline output status:")
    for stage, path, label, kind in checks:
        exists = os.path.exists(path)
        mark = "✓" if exists else "✗"
        extra = ""
        if stage == "bore":
            pass  # discovered-pool size intentionally not surfaced here
        elif exists and path.endswith(".csv"):
            with open(path) as f:
                n = sum(1 for _ in _csv.DictReader(f))
            extra = f"  ({n} {kind})"
        elif exists and os.path.isdir(path):
            n = len([x for x in os.listdir(path) if x.endswith(".png")])
            extra = f"  ({n} {kind})"
        logger.info(f"  {mark} {stage:<10} {label}{extra}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Run the energy profiling pipeline end-to-end or stage by stage."
    )
    p.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=ALL_STAGES,
                   metavar="STAGE",
                   help=f"Stages to run. Choices: {ALL_STAGES}. Default: all.")
    p.add_argument("--skip", nargs="+", choices=ALL_STAGES, default=[],
                   metavar="STAGE",
                   help="Stages to skip even if listed in --stages.")
    p.add_argument("--workers", type=int, default=N_WORKERS,
                   help=f"Process-pool size for extract+maps (default: {N_WORKERS})")
    p.add_argument("--caption-workers", type=int, default=CAPTION_WORKERS,
                   help=f"Thread-pool size for captions (default: {CAPTION_WORKERS})")
    p.add_argument("--status", action="store_true",
                   help="Print current output status and exit.")
    args = p.parse_args()

    if args.status:
        _status()
        return

    stages = [s for s in args.stages if s not in args.skip]
    if not stages:
        logger.warning("No stages to run after applying --skip. Exiting.")
        return

    logger.info(f"Running stages: {stages}  "
               f"(workers={args.workers}, caption_workers={args.caption_workers})")
    t0_total = time.time()
    runners = {
        "bore":      run_bore,
        "select":    run_select,
        "stream":    lambda: run_stream(args.workers, args.caption_workers),
    }

    for stage in stages:
        t0 = time.time()
        try:
            runners[stage]()
            elapsed = (time.time() - t0) / 60
            logger.info(f"Stage '{stage}' completed in {elapsed:.1f} min")
        except Exception as e:
            logger.error(f"Stage '{stage}' FAILED: {e}")
            logger.error("Aborting pipeline.")
            raise

    total = (time.time() - t0_total) / 60
    logger.info("=" * 60)
    logger.info(f"Pipeline complete — {len(stages)} stage(s) in {total:.1f} min")
    _status()


if __name__ == "__main__":
    main()
