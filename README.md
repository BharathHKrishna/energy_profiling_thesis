# Energy Profiling Thesis Pipeline

Geospatial energy profiling pipeline for Master's thesis in Energy Informatics.

Extracts energy-relevant features from 6 open data sources for globally stratified
coordinates, and generates natural-language energy profile captions via Groq (Llama-3.3-70b).

---

## What It Does

1. **Stratified sampling** — selects coordinates across 35 land-cover union types
   (port, oil field, informal settlement, megacity fringe, glacier edge, etc.)
   using a two-stage candidate scoring system anchored to real global locations
2. **Feature extraction** — pulls measurements from 6 sources per 512×512m bounding box
3. **Caption generation** — Groq LLM converts feature rows into energy profile text

**Current status**: 35/35 union types verified (35/35 confirmed, mean score 0.713).
Demo pipeline working end-to-end on 10 locations.

---

## Project Structure

```
scripts/
  demo_pipeline.py          # End-to-end demo: 10 locations → features → captions
  extractors/               # One extractor per data source
    worldcover_extractor.py
    ghsl_extractor.py
    solar_atlas_extractor.py
    era5_extractor.py
    viirs_extractor.py
  osm/
    osm_query.py            # OSM Overpass feature extraction
  sampling/
    union_anchor_regions.py # 35 union types + anchor coordinates (ANCHORS_ALL)
    candidate_generator.py  # 10 candidates per anchor, ±20° fan
    candidate_scorer.py     # Stage 1 fast score + Stage 2 full 4-criterion score
    stratified_sampler.py   # Orchestrator: confirmed-first winner selection
  utils/
    config_loader.py
    logger.py

configs/config.yaml         # All paths, thresholds, API settings
notebooks/
  raw_data_explorer_executed.ipynb   # Data source exploration (executed)
  day_verify.ipynb                   # Sampling system verification
  method_comparison.ipynb            # M1/M2/M3 sub-polygon comparison
rasters/                    # Local GeoTIFFs (not in git — download separately)
outputs/                    # Generated CSVs and JSONs (not in git)
docs/
  feature_best_worst_reference.pdf   # Feature value reference across union types
```

---

## Setup

**Requirements**: Python 3.11, GEE authentication

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in API keys
earthengine authenticate    # one-time GEE auth
```

**API keys needed** (add to `.env`):
- `GROQ_API_KEY` — https://console.groq.com (free tier)
- `CDS_API_KEY` — https://cds.climate.copernicus.eu (free)
- `GEE_PROJECT` — your Google Earth Engine project ID

---

## Running

**Demo — 10 locations, full pipeline, ~2.5 minutes:**
```bash
python scripts/demo_pipeline.py
```
Outputs to `outputs/csv/demo_10locations_captioned.json`

**Sampling system — verify 35 union types:**
```bash
# In Python / notebook
from scripts.sampling.stratified_sampler import run_sampling
from scripts.sampling.union_anchor_regions import ANCHORS_ALL
results = run_sampling(anchors=ANCHORS_ALL, seed=42)
```

---

## Data Sources

| Source | Features extracted | Method |
|---|---|---|
| ESA WorldCover 2021 | Land cover class, % per class, std | AWS S3 tiles (10m) |
| GHSL (JRC) | Population/km², building height, built surface m² | Local GeoTIFF (100m) |
| Global Solar Atlas | PVOUT, GHI, DNI kWh/m²/day | Local GeoTIFF |
| ERA5 (Copernicus) | Surface solar radiation J/m²/day | Local NetCDF → GeoTIFF |
| VIIRS via GEE | Nighttime lights nW/cm²/sr, NDVI, reflectance | Google Earth Engine API |
| OSM Overpass | Power infra, buildings, landuse, roads | Overpass API |

---

## Key Design Decisions

- **fiona pinned to 1.9.6** — 1.10.x breaks GeoPandas 0.14.3
- **VIIRS via GEE** — direct EOG/NASA download requires OAuth; GEE API is free and stable
- **OSM via kumi.systems mirror** — overpass-api.de can be IP-blocked; mirror is the primary
- **E criterion removed** — WC contrast was redundant with A for boundaries, zero for pure strata
- **Confirmed-first winner selection** — a confirmed candidate always beats non-confirmed
  regardless of B/C/D scores, preventing data-rich but off-target candidates from winning

---

## Future Work

1. Expand to full 5,250 production run (multiple anchors per union in `union_anchor_regions.py`)
2. Wire LangChain properly using LCEL (`prompt | ChatGroq | StrOutputParser`) with `.batch()`
3. Coordinate API — give lat/lon, get features + caption + satellite image link
