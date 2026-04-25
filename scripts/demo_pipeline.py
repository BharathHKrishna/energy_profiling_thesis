"""
demo_pipeline.py

Full end-to-end pipeline for 10 globally important energy-relevant locations:
  1. Generate 512m bboxes
  2. Extract all 6 data sources (WC, GHSL, Solar, ERA5, VIIRS, OSM)
  3. Save raw dataset to CSV + JSON
  4. Use Groq (Llama-3.3-70b) to generate an energy profile caption per location
  5. Save final labelled dataset with Google Maps links

Run from project root:
    source venv/bin/activate
    python scripts/demo_pipeline.py
"""

import sys
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import os
import json
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from scripts.extractors.worldcover_extractor import extract_worldcover_features
from scripts.extractors.ghsl_extractor import extract_ghsl_features
from scripts.extractors.solar_atlas_extractor import extract_solar_features
from scripts.extractors.era5_extractor import extract_era5_features
from scripts.extractors.viirs_extractor import extract_viirs_features
from scripts.osm.osm_query import extract_osm_features
from scripts.utils.logger import get_logger

load_dotenv("/srv/THESIS/energy_profiling_thesis/.env")
logger = get_logger("demo_pipeline")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"

OUTPUT_CSV  = "/srv/THESIS/energy_profiling_thesis/outputs/csv/demo_10locations.csv"
OUTPUT_JSON = "/srv/THESIS/energy_profiling_thesis/outputs/csv/demo_10locations.json"
OUTPUT_CAPTIONED = "/srv/THESIS/energy_profiling_thesis/outputs/csv/demo_10locations_captioned.json"

# ── 10 globally important locations ──────────────────────────────────────────
# Chosen to cover diverse energy-access contexts across all continents.

LOCATIONS = [
    {
        "id": 1,
        "name": "Rotterdam Maasvlakte Port",
        "country": "Netherlands",
        "lat": 51.9700, "lon": 4.0183,
        "union_type": "industrial_water",
        "context": "Europe's largest port — container terminal meets North Sea; major energy import hub",
    },
    {
        "id": 2,
        "name": "Kibera Informal Settlement",
        "country": "Kenya",
        "lat": -1.3066, "lon": 36.7907,
        "union_type": "informal_urban",
        "context": "One of Africa's largest informal settlements; extreme energy poverty, near-zero NTL",
    },
    {
        "id": 3,
        "name": "Tengiz Oil Processing Facility",
        "country": "Kazakhstan",
        "lat": 45.5050, "lon": 53.2340,
        "union_type": "isolated_industrial_darkness",
        "context": "One of the world's largest oil fields; isolated in Kazakh steppe, very high NTL",
    },
    {
        "id": 4,
        "name": "Mumbai Nariman Point Seawall",
        "country": "India",
        "lat": 18.9540, "lon": 72.8067,
        "union_type": "urban_coastal",
        "context": "Reclaimed land at Arabian Sea — dense urban fabric meets coastline in single pixel row",
    },
    {
        "id": 5,
        "name": "Noida Megacity Fringe",
        "country": "India",
        "lat": 28.6270, "lon": 77.3720,
        "union_type": "megacity_energy_periphery",
        "context": "Outer Delhi megacity boundary; GHSL pop ~8000/km² transitions to farmland within 500m",
    },
    {
        "id": 6,
        "name": "Ashburn Data Center Cluster",
        "country": "USA",
        "lat": 39.0352, "lon": -77.4812,
        "union_type": "datacenter_industrial",
        "context": "World's densest data center corridor (Route 28); hyperscale facilities, very low residential pop",
    },
    {
        "id": 7,
        "name": "Congo Basin Rainforest Interior",
        "country": "DRC",
        "lat": -0.5800, "lon": 23.5200,
        "union_type": "pure_tree_cover",
        "context": "World's second largest tropical forest; near-zero energy access, extreme biodiversity",
    },
    {
        "id": 8,
        "name": "Syncrude Oil Sands Mine",
        "country": "Canada",
        "lat": 57.0200, "lon": -111.5700,
        "union_type": "isolated_industrial_darkness",
        "context": "World's largest oil sands open-pit mine; massive energy production in boreal forest",
    },
    {
        "id": 9,
        "name": "Itaipu Reservoir Shore",
        "country": "Brazil",
        "lat": -25.4200, "lon": -54.4800,
        "union_type": "water_forest",
        "context": "Itaipu hydropower reservoir — world's largest operational hydro plant; water meets Atlantic forest",
    },
    {
        "id": 10,
        "name": "Iowa Corn Belt",
        "country": "USA",
        "lat": 41.8700, "lon": -93.1000,
        "union_type": "pure_cropland",
        "context": "US corn belt interior; large-scale industrial agriculture, high solar potential, bioenergy feedstock",
    },
]


# ── Bbox helper ───────────────────────────────────────────────────────────────

def make_bbox(lat, lon, size_m=512):
    delta_lat = (size_m / 2) / 111320.0
    delta_lon = (size_m / 2) / (111320.0 * np.cos(np.radians(lat)))
    return {
        "min_lat": lat - delta_lat,
        "max_lat": lat + delta_lat,
        "min_lon": lon - delta_lon,
        "max_lon": lon + delta_lon,
    }


# ── Extract all 6 sources for one location ────────────────────────────────────

def extract_all(loc):
    lat, lon = loc["lat"], loc["lon"]
    bbox = make_bbox(lat, lon)
    mn_lat, mx_lat = bbox["min_lat"], bbox["max_lat"]
    mn_lon, mx_lon = bbox["min_lon"], bbox["max_lon"]

    logger.info(f"[{loc['id']:02d}] Extracting: {loc['name']}")

    wc    = extract_worldcover_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)
    ghsl  = extract_ghsl_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)
    solar = extract_solar_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)
    era5  = extract_era5_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)
    viirs = extract_viirs_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)
    osm   = extract_osm_features(lat, lon, mn_lat, mx_lat, mn_lon, mx_lon)

    row = {
        "id":          loc["id"],
        "name":        loc["name"],
        "country":     loc["country"],
        "lat":         lat,
        "lon":         lon,
        "union_type":  loc["union_type"],
        "context":     loc["context"],
        "bbox_min_lat": round(mn_lat, 6),
        "bbox_max_lat": round(mx_lat, 6),
        "bbox_min_lon": round(mn_lon, 6),
        "bbox_max_lon": round(mx_lon, 6),
    }

    # Flatten each source (skip private keys starting with _)
    for src_name, src_data in [("wc", wc), ("ghsl", ghsl), ("solar", solar),
                                ("era5", era5), ("viirs", viirs), ("osm", osm)]:
        if src_data:
            for k, v in src_data.items():
                if not k.startswith("_"):
                    row[k] = v

    return row, bbox


# ── LangChain + Groq caption chain ───────────────────────────────────────────

SYSTEM_PROMPT = """You are an energy access researcher writing concise, factual location profiles
for a geospatial dataset. Given structured feature data about a 512m × 512m geographic location,
write a 3-4 sentence energy profile that describes:
1. What the location is and what land cover it has
2. Its energy access situation (electrification, NTL, infrastructure)
3. Why it is significant for energy research

Be specific with numbers. Do not invent data not in the features. Write in plain English."""

HUMAN_TEMPLATE = """Location: {name} ({country})
Union type: {union_type}
Context: {context}

Feature measurements (512m × 512m bbox):
- Dominant land cover: {wc_dominant_class} ({wc_dominant_pct:.1f}%)
- WorldCover class count: {wc_class_count}
- GHSL population density: {ghsl_population_per_km2:.0f} people/km²
- GHSL building height: {ghsl_building_height_m:.1f} m
- GHSL built surface: {ghsl_built_surface_m2:.0f} m²
- VIIRS nighttime lights: {viirs_ntl_nw_cm2_sr:.1f} nW/cm²/sr
- VIIRS NDVI: {viirs_ndvi:.3f}
- Solar PVOUT: {solar_pvout_kwh_kwp_day:.2f} kWh/kWp/day
- Solar GHI: {solar_ghi_kwh_m2_day:.2f} kWh/m²/day
- ERA5 solar radiation: {era5_ssrd_j_m2_day:.0f} J/m²/day
- OSM building count: {osm_building_count}
- OSM energy infrastructure: {osm_power_line} power lines, {osm_power_substation} substations

Write the energy profile:"""


def maps_link(lat, lon):
    """Google Maps satellite link for a coordinate."""
    return f"https://www.google.com/maps/@{lat},{lon},500m/data=!3m1!1e3"


def build_caption_chain():
    client = Groq(api_key=GROQ_API_KEY)

    def groq_invoke(prompt_text):
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt_text},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    return groq_invoke


def build_prompt(row):
    """Fill the human template with row values, defaulting missing fields to 0."""
    def g(key, default=0):
        v = row.get(key, default)
        return v if v is not None else default

    return HUMAN_TEMPLATE.format(
        name=row["name"],
        country=row["country"],
        union_type=row["union_type"],
        context=row["context"],
        wc_dominant_class=g("wc_dominant_class", "unknown"),
        wc_dominant_pct=g("wc_dominant_pct", 0.0),
        wc_class_count=g("wc_class_count", 0),
        ghsl_population_per_km2=g("ghsl_population_per_km2", 0.0),
        ghsl_building_height_m=g("ghsl_building_height_m", 0.0),
        ghsl_built_surface_m2=g("ghsl_built_surface_m2", 0.0),
        viirs_ntl_nw_cm2_sr=g("viirs_ntl_nw_cm2_sr", 0.0),
        viirs_ndvi=g("viirs_ndvi", 0.0),
        solar_pvout_kwh_kwp_day=g("solar_pvout_kwh_kwp_day", 0.0),
        solar_ghi_kwh_m2_day=g("solar_ghi_kwh_m2_day", 0.0),
        era5_ssrd_j_m2_day=g("era5_ssrd_j_m2_day", 0.0),
        osm_building_count=int(g("osm_building_count", 0)),
        osm_power_line=int(g("osm_power_line", 0)),
        osm_power_substation=int(g("osm_power_substation", 0)),
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run():
    os.makedirs("/srv/THESIS/energy_profiling_thesis/outputs/csv", exist_ok=True)

    # ── Step 1: Extract all features ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Extracting features from 6 sources for 10 locations")
    logger.info("=" * 60)

    rows = []
    bboxes = {}
    for loc in LOCATIONS:
        try:
            row, bbox = extract_all(loc)
            rows.append(row)
            bboxes[loc["id"]] = bbox
            logger.info(f"  [{loc['id']:02d}] OK — WC={row.get('wc_dominant_class','?')} | "
                        f"NTL={row.get('viirs_ntl_nw_cm2_sr', 0):.1f} | "
                        f"pop={row.get('ghsl_population_per_km2', 0):.0f}/km²")
        except Exception as e:
            logger.error(f"  [{loc['id']:02d}] FAILED: {e}")
        time.sleep(0.5)

    # ── Step 2: Save raw dataset ──────────────────────────────────────────────
    logger.info("\nSTEP 2: Saving raw dataset")
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"  CSV saved: {OUTPUT_CSV} ({len(df)} rows, {len(df.columns)} columns)")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    logger.info(f"  JSON saved: {OUTPUT_JSON}")

    # ── Step 3: Generate captions via Groq + LangChain ────────────────────────
    logger.info("\nSTEP 3: Generating energy profile captions via Groq (Llama-3.3-70b)")
    groq_chain = build_caption_chain()

    captioned = []
    for row in rows:
        try:
            prompt = build_prompt(row)
            caption = groq_chain(prompt)
            logger.info(f"  [{row['id']:02d}] Caption generated ({len(caption)} chars)")
        except Exception as e:
            logger.error(f"  [{row['id']:02d}] Caption FAILED: {e}")
            caption = ""

        captioned.append({
            "id":         row["id"],
            "name":       row["name"],
            "country":    row["country"],
            "lat":        row["lat"],
            "lon":        row["lon"],
            "union_type": row["union_type"],
            "google_maps_link": maps_link(row["lat"], row["lon"]),
            "bbox": bboxes.get(row["id"], {}),
            "features": {
                "wc_dominant_class":       row.get("wc_dominant_class"),
                "wc_dominant_pct":         row.get("wc_dominant_pct"),
                "wc_class_count":          row.get("wc_class_count"),
                "ghsl_population_per_km2": row.get("ghsl_population_per_km2"),
                "ghsl_building_height_m":  row.get("ghsl_building_height_m"),
                "ghsl_built_surface_m2":   row.get("ghsl_built_surface_m2"),
                "viirs_ntl_nw_cm2_sr":     row.get("viirs_ntl_nw_cm2_sr"),
                "viirs_ndvi":              row.get("viirs_ndvi"),
                "solar_pvout_kwh_kwp_day": row.get("solar_pvout_kwh_kwp_day"),
                "solar_ghi_kwh_m2_day":    row.get("solar_ghi_kwh_m2_day"),
                "era5_ssrd_j_m2_day":      row.get("era5_ssrd_j_m2_day"),
                "osm_building_count":      row.get("osm_building_count"),
                "osm_power_line":          row.get("osm_power_line"),
                "osm_power_substation":    row.get("osm_power_substation"),
            },
            "caption": caption,
        })
        time.sleep(0.3)  # Groq rate limit buffer

    with open(OUTPUT_CAPTIONED, "w") as f:
        json.dump(captioned, f, indent=2, default=str)
    logger.info(f"\nSTEP 3 complete: {OUTPUT_CAPTIONED}")

    # ── Step 4: Print summary ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE — 10 locations processed")
    logger.info("=" * 60)
    for entry in captioned:
        f = entry["features"]
        print(f"\n[{entry['id']:02d}] {entry['name']} ({entry['country']})")
        print(f"     Union: {entry['union_type']}")
        print(f"     WC: {f.get('wc_dominant_class')} ({f.get('wc_dominant_pct', 0):.1f}%) | "
              f"NTL: {f.get('viirs_ntl_nw_cm2_sr', 0):.1f} nW | "
              f"Pop: {f.get('ghsl_population_per_km2', 0):.0f}/km²")
        print(f"     Caption: {entry['caption'][:200]}...")

    return captioned


if __name__ == "__main__":
    run()
