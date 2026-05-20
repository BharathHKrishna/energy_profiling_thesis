"""
Groq Caption Generator — pore_features.csv → pore_captions.json

Reads the 18-row feature CSV and generates a 3-5 line descriptive caption
per stratum using llama-3.3-70b-versatile.

Run:
    python scripts/pore/groq_caption.py
"""
import os, sys, json, csv, time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")
load_dotenv("/srv/THESIS/energy_profiling_thesis/.env")

from groq import Groq
from scripts.utils.logger import get_logger

logger = get_logger("groq_caption")

BASE_DIR   = Path("/srv/THESIS/energy_profiling_thesis")
CSV_PATH   = BASE_DIR / "outputs/csv/pore_features.csv"
OUT_PATH   = BASE_DIR / "outputs/captions/pore_captions.json"
MODEL      = "llama-3.3-70b-versatile"


def _fmt(val, unit="", missing="n/a"):
    try:
        f = float(val)
        return f"{f:.2f}{unit}" if unit else f"{f:.2f}"
    except (ValueError, TypeError):
        return str(val) if val not in ("", None) else missing


def build_prompt(row: dict) -> str:
    stratum    = row["stratum_name"]
    lat, lon   = _fmt(row["lat"]), _fmt(row["lon"])
    dominant   = row.get("wc_dominant_class", "n/a")
    dom_pct    = _fmt(row.get("wc_dominant_pct"), "%")
    built_pct  = _fmt(row.get("wc_built_up_pct"), "%")
    water_pct  = _fmt(row.get("wc_water_pct"), "%")
    tree_pct   = _fmt(row.get("wc_tree_cover_pct"), "%")
    crop_pct   = _fmt(row.get("wc_cropland_pct"), "%")
    bld_ht     = _fmt(row.get("ghsl_building_height_m"), " m")
    pop        = _fmt(row.get("ghsl_population_per_km2"), "/km²")
    pvout      = _fmt(row.get("solar_pvout_kwh_kwp_day"), " kWh/kWp/day")
    ntl        = _fmt(row.get("viirs_ntl_nw_cm2_sr"))
    bld_cnt    = row.get("osm_building_count", "n/a")
    has_plant  = row.get("osm_power_plant", "")
    plant_src  = row.get("osm_plant_source", "")
    landuse    = row.get("osm_landuse", "")
    waterway   = row.get("osm_waterway", "")
    has_sub    = row.get("osm_power_substation", "")

    infra_bits = []
    if str(has_plant).strip().lower() == "true":
        infra_bits.append(f"power plant ({plant_src})" if plant_src else "power plant")
    if waterway:
        infra_bits.append(f"waterway ({waterway})")
    if str(has_sub).strip().lower() == "true":
        infra_bits.append("power substation")
    infra_str = ", ".join(infra_bits) if infra_bits else "none detected"
    if landuse:
        infra_str += f"; land-use tagged as '{landuse}'"

    prompt = f"""You are a geospatial energy analyst. Write a concise 3-5 line description of the following 512m×512m tile sampled for an energy infrastructure study. Focus on what makes this location energy-relevant.

Stratum: {stratum}
Coordinates: {lat}, {lon}
Land cover (ESA WorldCover): dominant={dominant} {dom_pct}, built-up={built_pct}, water={water_pct}, tree cover={tree_pct}, cropland={crop_pct}
Building stock: count={bld_cnt}, mean height={bld_ht}
Population density: {pop}
Solar resource: {pvout}
VIIRS night-time lights: {ntl} nW/cm²/sr
Energy infrastructure (OSM): {infra_str}

Write only the description. No headings, no bullet points."""
    return prompt


def generate_captions(dry_run=False) -> dict:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    rows = []
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(rows)} rows from {CSV_PATH}")

    results = {}
    for i, row in enumerate(rows):
        stratum = row["stratum_name"]
        prompt  = build_prompt(row)

        if dry_run:
            results[stratum] = "[dry-run]"
            logger.info(f"[{i+1}/{len(rows)}] {stratum} — skipped (dry-run)")
            continue

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.4,
                )
                caption = resp.choices[0].message.content.strip()
                results[stratum] = caption
                logger.info(f"[{i+1}/{len(rows)}] {stratum} — OK ({len(caption)} chars)")
                break
            except Exception as e:
                logger.warning(f"[{i+1}/{len(rows)}] {stratum} attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(5)
                else:
                    results[stratum] = f"[ERROR: {e}]"

        time.sleep(0.5)  # stay well under Groq rate limit

    return results


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    captions = generate_captions()

    with open(OUT_PATH, "w") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(captions)} captions → {OUT_PATH}")

    for stratum, cap in captions.items():
        print(f"\n--- {stratum} ---")
        print(cap)


if __name__ == "__main__":
    main()
