"""
Energy Profiling Pipeline — unified runner

════════════════════════════════════════
  THE ONLY THING YOU EVER CHANGE:
  N_PER_STRATUM below — one number per stratum.
  Then run:  python run_pipeline.py
  All outputs are overwritten fresh each run.
════════════════════════════════════════

Stages (run in order):
  1. bore      → BORE coordinate sampling  → outputs/csv/filtered_strata_sample.csv
  2. features  → PORE feature extraction   → outputs/csv/pore_features.csv
  3. segmaps   → PORE segmentation maps    → outputs/maps/*.png
  4. captions  → Groq LLM captions        → outputs/captions/pore_captions.json

Full 2500-run targets (set N_PER_STRATUM to these):
  HIGH_union strata (Industrial+Water, Urban+Coastal, Informal+Urban):  250 each
  HIGH_pure  strata (Dense Urban, Suburban, Industrial):                300 each
  MID        strata (7 strata):                                         100 each
  LOW        strata (5 strata):                                          30 each

Advanced usage:
    python run_pipeline.py --stages bore features   # run specific stages only
    python run_pipeline.py --skip bore              # skip stages already done
    python run_pipeline.py --status                 # check what's already done
"""
import sys, os, argparse, time
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

from scripts.utils.logger import get_logger
logger = get_logger("run_pipeline")

# ════════════════════════════════════════════════════════════════════════════
# ▼▼▼  CHANGE THESE NUMBERS — nothing else  ▼▼▼
N_PER_STRATUM: dict = {
    "Industrial + Water":              1,   # full run: 250
    "Urban + Coastal":                 1,   # full run: 250
    "Informal + Urban":                1,   # full run: 250
    "Dense Urban":                     1,   # full run: 300
    "Suburban":                        1,   # full run: 300
    "Industrial":                      1,   # full run: 300
    "Data Centre + Industrial":        1,   # full run: 100
    "Industrial + Arid":               1,   # full run: 100
    "Agrivoltaics (Solar + Farmland)": 1,   # full run: 100
    "Industrial + Forest":             1,   # full run: 100
    "Hydropower Reservoir":            1,   # full run: 100
    "Utility-scale Solar Farm":        1,   # full run: 100
    "Airport / Aviation":              1,   # full run: 100
    "Agricultural + Water":            1,   # full run:  30
    "Coastal + Agricultural":          1,   # full run:  30
    "Mangrove + Industrial":           1,   # full run:  30
    "Suburban + Agricultural":         1,   # full run:  30
    "Coastal + Solar-Wind Hybrid":     1,   # full run:  30
}
# ▲▲▲  CHANGE THESE NUMBERS — nothing else  ▲▲▲
# ════════════════════════════════════════════════════════════════════════════

BASE        = "/srv/THESIS/energy_profiling_thesis"
COORDS_CSV  = os.path.join(BASE, "outputs/csv/filtered_strata_sample.csv")
FEATURES_CSV = os.path.join(BASE, "outputs/csv/pore_features.csv")
SEGMAPS_DIR  = os.path.join(BASE, "outputs/maps")
CAPTIONS_JSON = os.path.join(BASE, "outputs/captions/pore_captions.json")

ALL_STAGES = ["bore", "features", "segmaps", "captions"]


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_bore():
    """BORE: sample coordinates for all 18 strata using N_PER_STRATUM above."""
    logger.info("=" * 60)
    logger.info("STAGE 1 — BORE: coordinate sampling")
    logger.info("=" * 60)
    import scripts.bore.real_anchor_finder as raf
    raf.N_PER_STRATUM = N_PER_STRATUM
    logger.info(f"N_PER_STRATUM = {N_PER_STRATUM}  (total: {sum(N_PER_STRATUM.values())})")
    from scripts.bore.real_anchor_finder import main as bore_main
    bore_main()
    logger.info(f"BORE done → {COORDS_CSV}")


def run_features():
    """PORE: extract features per coordinate from 5 sources."""
    logger.info("=" * 60)
    logger.info("STAGE 2 — PORE: feature extraction")
    logger.info("=" * 60)
    if not os.path.exists(COORDS_CSV):
        raise FileNotFoundError(
            f"Coordinates CSV not found: {COORDS_CSV}\n"
            "Run BORE first:  python run_pipeline.py --stages bore"
        )
    from scripts.pore.run_pore import run as pore_run
    pore_run(input_csv=COORDS_CSV)
    logger.info(f"Features done → {FEATURES_CSV}")


def run_segmaps():
    """PORE: generate side-by-side segmentation maps for all coordinates."""
    logger.info("=" * 60)
    logger.info("STAGE 3 — PORE: segmentation maps")
    logger.info("=" * 60)
    if not os.path.exists(COORDS_CSV):
        raise FileNotFoundError(
            f"Coordinates CSV not found: {COORDS_CSV}\n"
            "Run BORE first:  python run_pipeline.py --stages bore"
        )
    # Load coords from CSV so the segmaps always match the current BORE output
    import csv
    coords = []
    with open(COORDS_CSV) as f:
        for row in csv.DictReader(f):
            coords.append((row["stratum_name"], float(row["lat"]), float(row["lon"])))
    logger.info(f"Loaded {len(coords)} coordinates from {COORDS_CSV}")

    from scripts.pore.segmap_generator import generate_multi
    generate_multi(coords=coords)
    logger.info(f"Segmaps done → {SEGMAPS_DIR}/")


def run_captions():
    """Groq: generate 3-5 line LLM captions from pore_features.csv."""
    logger.info("=" * 60)
    logger.info("STAGE 4 — Captions: Groq LLM")
    logger.info("=" * 60)
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(
            f"Features CSV not found: {FEATURES_CSV}\n"
            "Run features first:  python run_pipeline.py --stages features"
        )
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE, ".env"))
    from scripts.captions.groq_caption import main as caption_main
    caption_main()
    logger.info(f"Captions done → {CAPTIONS_JSON}")


# ── Output status check ───────────────────────────────────────────────────────

def _status():
    checks = [
        ("bore",     COORDS_CSV,    "outputs/csv/filtered_strata_sample.csv"),
        ("features", FEATURES_CSV,  "outputs/csv/pore_features.csv"),
        ("segmaps",  SEGMAPS_DIR,   "outputs/maps/"),
        ("captions", CAPTIONS_JSON, "outputs/captions/pore_captions.json"),
    ]
    logger.info("Pipeline output status:")
    for stage, path, label in checks:
        exists = os.path.exists(path)
        mark = "✓" if exists else "✗"
        extra = ""
        if exists and path.endswith(".csv"):
            import csv
            with open(path) as f:
                n = sum(1 for _ in csv.DictReader(f))
            extra = f"  ({n} rows)"
        elif exists and os.path.isdir(path):
            n = len([x for x in os.listdir(path) if x.endswith("_segmap.png")])
            extra = f"  ({n} segmaps)"
        logger.info(f"  {mark} {stage:<10} {label}{extra}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Run the energy profiling pipeline end-to-end or stage by stage."
    )
    p.add_argument(
        "--stages", nargs="+", choices=ALL_STAGES, default=ALL_STAGES,
        metavar="STAGE",
        help=f"Stages to run. Choices: {ALL_STAGES}. Default: all.",
    )
    p.add_argument(
        "--skip", nargs="+", choices=ALL_STAGES, default=[],
        metavar="STAGE",
        help="Stages to skip even if listed in --stages.",
    )
    p.add_argument(
        "--status", action="store_true",
        help="Print current output status and exit.",
    )
    args = p.parse_args()

    if args.status:
        _status()
        return

    stages = [s for s in args.stages if s not in args.skip]
    if not stages:
        logger.warning("No stages to run after applying --skip. Exiting.")
        return

    logger.info(f"Running stages: {stages}")
    t0_total = time.time()
    runners = {
        "bore":     run_bore,
        "features": run_features,
        "segmaps":  run_segmaps,
        "captions": run_captions,
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
