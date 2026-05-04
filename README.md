# Energy Profiling Thesis Pipeline

Geospatial energy profiling pipeline for Master's thesis in Energy Informatics.

Extracts energy-relevant features from 6 open data sources for 2,500 globally
stratified coordinates across 18 land-cover strata, verified via satellite imagery.

---

## Pipeline Phases

| Phase | Name | Status |
|-------|------|--------|
| BORE | Bbox-centred OSM-filtered Raster-Evaluated sampling | **Complete** — 18 anchors validated |
| PORE | Per-coordinate feature extraction + caption generation | Next |

---

## BORE — 18 Strata

Coordinates are selected in three stages per stratum:

1. **OSM Overpass** — queries specific tags (docks, dams, sawmills, coastlines, etc.) in
   known geographic regions to produce a candidate pool
2. **ESA WorldCover** — checks primary + secondary land-cover % thresholds in the 512 m bbox
3. **GHSL** — checks population density, building height, and built-surface bounds

| Tier | Strata | N (2500 run) |
|------|--------|-------------|
| HIGH union | 3 | 250 each |
| HIGH pure  | 3 | 300 each |
| MID        | 7 | 100 each |
| LOW        | 5 | 30 each  |

Full threshold table: `docs/strata_table.html`  
Full methodology with reasoning: `docs/strata_methodology.html`

---

## Project Structure

```
scripts/
  sampling/
    real_anchor_finder.py     # BORE core — OSM queries, ESA+GHSL filtering, anchor output
    coordinate_filter.py      # Strata config: ESA thresholds, GHSL bounds, HTML parsers
    n25_full_validation.py    # N=25 pass-rate validation across all 18 strata
  extractors/
    worldcover_extractor.py   # ESA WorldCover land-cover % from AWS S3 tiles
    ghsl_extractor.py         # GHSL population, building height, built surface
    solar_atlas_extractor.py  # Global Solar Atlas PVOUT / GHI / DNI
    era5_extractor.py         # ERA5 surface solar radiation (NetCDF)
    viirs_extractor.py        # VIIRS nighttime lights + NDVI via GEE
  osm/
    osm_query.py              # OSM Overpass feature extraction (PORE use)
  utils/
    config_loader.py
    logger.py

configs/config.yaml           # Raster paths, API settings, BORE parameters
docs/
  features.html               # ESA WorldCover class definitions (parsed by pipeline)
  2500_coordinate_breakdown.html  # Importance-tier assignments (parsed by pipeline)
  strata_table.html           # 18-stratum quick-reference table
  strata_methodology.html     # Full A–Z methodology per stratum
  feature_best_worst_reference.pdf

notebooks/
  day_verify.ipynb            # 18 final anchors — satellite maps + pass-rate charts
  pool_verify.ipynb           # Top-3 candidates per stratum from N=25 pool
  raw_data_explorer_executed.ipynb  # 6 data sources demonstrated
  method_comparison.ipynb     # Early method comparison (reference only)

outputs/csv/
  filtered_strata_sample.csv  # 18 best anchors (1 per stratum)
  n25_full_validation.csv     # Full N=25 pool (450 rows, 25 per stratum)

rasters/                      # Local GeoTIFFs — not in git, download separately
```

---

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in API keys
earthengine authenticate      # one-time GEE auth
```

**API keys needed** (add to `.env`):
- `GROQ_API_KEY` — https://console.groq.com (free tier)
- `CDS_API_KEY` — https://cds.climate.copernicus.eu (free)
- `GEE_PROJECT` — your Google Earth Engine project ID

---

## Running BORE

```bash
# Full N=25 validation across all 18 strata (~30 min, Overpass rate-limited)
python scripts/sampling/n25_full_validation.py

# Scale control — edit N_PER_TIER in real_anchor_finder.py:
# Anchor validation:  {"all": 1}
# Pilot run:          {"all": 5}
# Full 2500 run:      {"HIGH_union": 250, "HIGH_pure": 300, "MID": 100, "LOW": 30}
```

---

## Data Sources

| Source | Features | Method |
|--------|----------|--------|
| ESA WorldCover 2021 | Land-cover class %, dominant class, std | AWS S3 tiles (10 m) |
| GHSL (JRC) | Population/km², building height m, built surface m² | Local GeoTIFF (100 m) |
| Global Solar Atlas | PVOUT, GHI, DNI kWh/m²/day | Local GeoTIFF |
| ERA5 (Copernicus) | Surface solar radiation J/m²/day | Local NetCDF → GeoTIFF |
| VIIRS via GEE | Nighttime lights nW/cm²/sr, NDVI, reflectance | Google Earth Engine API |
| OSM Overpass | Power infra, buildings, landuse, roads | Overpass API |

---

## Key Design Notes

- **fiona pinned to 1.9.6** — 1.10.x breaks GeoPandas 0.14.3
- **OSM via kumi.systems mirror** — overpass-api.de can be IP-blocked; mirror is primary
- **512 m bbox is fixed** — all strata, all runs, no exceptions
- **Anchor selection** — SECONDARY_BEST strata pick highest secondary_pct; Hydropower picks
  lowest water% (dam wall visible); all others pick highest primary_pct
