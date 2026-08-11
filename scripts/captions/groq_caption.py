"""
Groq Caption Generator

Accepts a pre-built feature dict and returns a factual 3-5 sentence
energy-focused description using llama-3.3-70b-versatile.

Primary usage (web app):
    from scripts.captions.groq_caption import generate_caption
    caption = generate_caption(features, bbox_size_m=512)

CLI usage (batch, reads from pore_features.csv):
    python scripts/captions/groq_caption.py
"""
import os, sys, json, csv, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")
load_dotenv("/srv/THESIS/energy_profiling_thesis/.env")

from groq import Groq
from scripts.utils.logger import get_logger

logger = get_logger("groq_caption")

MODEL    = "llama-3.3-70b-versatile"
BASE_DIR = Path("/srv/THESIS/energy_profiling_thesis")
CSV_PATH = BASE_DIR / "outputs/csv/pore_features.csv"
OUT_PATH = BASE_DIR / "outputs/captions/pore_captions.json"
CAPTION_WORKERS = 6   # threads, not processes — this is all network I/O wait on the
                      # Groq API, no CPU work; each generate_caption() call already
                      # retries 3x with backoff on its own, so worker failures are cheap


def _fmt(val, decimals=1, unit="", missing="unknown"):
    if val is None:
        return missing
    try:
        return f"{float(val):.{decimals}f}{unit}"
    except (TypeError, ValueError):
        return str(val)


def build_prompt(features: dict, bbox_size_m: int = 512) -> str:
    lat = features.get("lat", 0)
    lon = features.get("lon", 0)

    dominant     = features.get("wc_dominant_class", "unknown")
    dominant_pct = _fmt(features.get("wc_dominant_pct"), unit="%")
    built_pct    = _fmt(features.get("wc_built_up_pct"), unit="%")
    tree_pct     = _fmt(features.get("wc_tree_cover_pct"), unit="%")
    water_pct    = _fmt(features.get("wc_water_pct"), unit="%")
    crop_pct     = _fmt(features.get("wc_cropland_pct"), unit="%")
    bare_pct     = _fmt(features.get("wc_bare_sparse_pct"), unit="%")

    pop    = _fmt(features.get("ghsl_population_per_km2"), decimals=0, unit=" /km²")
    bld_ht = _fmt(features.get("ghsl_building_height_m"), unit=" m")

    pvout = _fmt(features.get("solar_pvout_kwh_kwp_day"), unit=" kWh/kWp/day")
    ghi   = _fmt(features.get("solar_ghi_kwh_m2_day"),   unit=" kWh/m²/day")

    wind_speed = _fmt(features.get("wind_speed_100m_ms"),           unit=" m/s")
    wind_power = _fmt(features.get("wind_power_density_100m_wm2"), decimals=0, unit=" W/m²")

    ntl       = _fmt(features.get("viirs_ntl_nw_cm2_sr"), decimals=2, unit=" nW/cm²/sr")
    hdd       = _fmt(features.get("climate_hdd"),         decimals=0, unit=" °C·days")
    cdd       = _fmt(features.get("climate_cdd"),         decimals=0, unit=" °C·days")
    mean_temp = _fmt(features.get("climate_mean_temp_c"),             unit=" °C")

    heating_mwh   = _fmt(features.get("heating_MWh"), unit=" MWh/yr")
    cooling_mwh   = _fmt(features.get("cooling_MWh"), unit=" MWh/yr")
    demand_regime = features.get("demand_regime") or "unknown"
    demand_score  = _fmt(features.get("demand_score"), unit="/100")
    demand_tier   = features.get("demand_tier") or "unknown"

    infra_parts = []
    if features.get("osm_power_plant"):
        src = features.get("osm_plant_source", "")
        infra_parts.append(f"power plant ({src})" if src else "power plant")
    if features.get("osm_power_substation"):
        infra_parts.append("substation")
    if features.get("osm_power_line"):
        infra_parts.append("power lines")
    if features.get("osm_power_tower"):
        infra_parts.append("power towers")
    if features.get("osm_generator_source"):
        infra_parts.append(f"generator ({features['osm_generator_source']})")
    if features.get("osm_waterway"):
        infra_parts.append(f"waterway ({features['osm_waterway']})")
    if features.get("osm_landuse"):
        infra_parts.append(f"land-use: {features['osm_landuse']}")
    infra_str = ", ".join(infra_parts) if infra_parts else "none detected"

    bld_count = features.get("osm_building_count", 0)

    return f"""You are a geospatial energy analyst. A user has selected a {bbox_size_m}m \xd7 {bbox_size_m}m area at {lat:.4f}\xb0, {lon:.4f}\xb0. Write a concise 3-5 sentence factual description of this location from an energy perspective.

Land cover (ESA WorldCover): dominant={dominant} ({dominant_pct}), built-up={built_pct}, tree cover={tree_pct}, water={water_pct}, cropland={crop_pct}, bare/sparse={bare_pct}
Solar potential: PVOUT={pvout}, GHI={ghi}
Wind potential (100m hub height): speed={wind_speed}, power density={wind_power}
Nighttime lights (VIIRS): {ntl}
Climate (annual, base 18°C): heating degree days={hdd}, cooling degree days={cdd}, mean temp={mean_temp}
Estimated building energy demand (Eurostat-fit model, per 512m cell): heating={heating_mwh}, cooling={cooling_mwh} (regime: {demand_regime})
Energy-demand score (synthetic 0-100 index): {demand_score} ({demand_tier})
Population density: {pop}
Mean building height: {bld_ht}
OSM buildings in area: {bld_count}
Energy infrastructure (OSM): {infra_str}

Describe what the land cover, energy potential figures, and infrastructure reveal about this location’s energy profile. Be specific and factual. Write only the description — no headings, no bullet points. Do not use em dashes (—); use commas, periods, or "and" instead."""


def generate_caption(features: dict, bbox_size_m: int = 512) -> str:
    """
    Generate a single energy caption for one coordinate's feature dict.
    Returns the caption string. Raises on failure — never returns an
    "[Error: ...]" string that could pass as a real caption downstream (a missing
    key or an exhausted retry used to return one; `audit_stream_output.py`'s
    `len(caption) > 20` completeness check couldn't tell that apart from a real
    caption, so a missing GROQ_API_KEY silently looked like 100% success. Fixed
    2026-08-11 — callers must handle the exception (run_pipeline.py's stream
    already does; see its caption-harvesting loop).
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)
    prompt = build_prompt(features, bbox_size_m)

    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            caption = resp.choices[0].message.content.strip()
            logger.info(f"Caption generated ({len(caption)} chars)")
            return caption
        except Exception as e:
            last_err = e
            logger.warning(f"Groq attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(3)

    raise RuntimeError(f"caption generation failed after 3 attempts: {last_err}")


# ── Batch CLI (reads pore_features.csv, writes pore_captions.json) ────────────

_NUMERIC_KEYS = ("lat", "lon", "wc_dominant_pct", "wc_built_up_pct",
                 "wc_tree_cover_pct", "wc_water_pct", "wc_cropland_pct",
                 "wc_bare_sparse_pct", "ghsl_population_per_km2",
                 "ghsl_building_height_m", "solar_pvout_kwh_kwp_day",
                 "solar_ghi_kwh_m2_day", "wind_speed_100m_ms",
                 "wind_power_density_100m_wm2", "osm_building_count",
                 "viirs_ntl_nw_cm2_sr", "climate_hdd", "climate_cdd",
                 "climate_cdd24", "climate_mean_temp_c",
                 "heating_MWh", "cooling_MWh", "demand_score")


def _caption_task(i, row):
    """One row's worth of work — runs in a worker thread. Returns (stratum, caption)."""
    stratum = row.get("stratum_name", f"row_{i}")
    bbox_m  = int(float(row.get("bbox_size_m", 512)))

    features = {k: (None if v == "" else v) for k, v in row.items()}
    for num_key in _NUMERIC_KEYS:
        if features.get(num_key) not in (None, ""):
            try:
                features[num_key] = float(features[num_key])
            except (ValueError, TypeError):
                pass

    try:
        caption = generate_caption(features, bbox_size_m=bbox_m)
    except Exception as e:
        logger.warning(f"{stratum}: caption failed: {e}")
        caption = None
    return stratum, caption


def main():
    """Batch caption generation for BORE/PORE pipeline output — parallel over rows."""
    load_dotenv(BASE_DIR / ".env")

    rows = []
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    logger.info(f"Loaded {len(rows)} rows from {CSV_PATH} — {CAPTION_WORKERS} parallel workers")

    results  = {}
    done     = 0
    lock     = threading.Lock()

    with ThreadPoolExecutor(max_workers=CAPTION_WORKERS) as pool:
        futures = {pool.submit(_caption_task, i, row): row for i, row in enumerate(rows)}
        for future in as_completed(futures):
            stratum, caption = future.result()
            if caption is not None:
                results[stratum] = caption
            with lock:
                done += 1
                status = "done" if caption is not None else "FAILED"
                logger.info(f"[{done}/{len(rows)}] {stratum} — {status}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(results)} captions → {OUT_PATH}")

    for stratum, cap in results.items():
        print(f"\n--- {stratum} ---\n{cap}")


if __name__ == "__main__":
    main()
