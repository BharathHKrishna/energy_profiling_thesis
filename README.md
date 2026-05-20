# Energy Profiling Thesis Pipeline

Geospatial energy profiling pipeline for Master's thesis in Energy Informatics.

Extracts energy-relevant features from 5 open data sources for 2,500 globally
stratified coordinates across 18 land-cover strata, verified via satellite imagery.

---

## Pipeline Phases

| Phase | Name | Status |
|-------|------|--------|
| BORE | Bbox-centred OSM-filtered Raster-Evaluated sampling | **Complete** — 18 strata, N=1→2500 |
| PORE | Per-coordinate feature extraction + segmaps + captions | **Complete** — all 4 stages unified |

---

## Running the Pipeline

```bash
# 1. Edit N_PER_STRATUM in run_pipeline.py (the ONLY thing you change)
# 2. Run:
python run_pipeline.py
```

All 4 stages run in order automatically. All outputs are overwritten fresh each run.

| Stage | What it does | Output |
|-------|-------------|--------|
| bore | BORE coordinate sampling | `outputs/csv/filtered_strata_sample.csv` |
| features | PORE feature extraction (5 sources) | `outputs/csv/pore_features.csv` |
| segmaps | Segmentation maps (MS + OSM overlay) | `outputs/maps/*.png` |
| captions | Groq LLM captions | `outputs/captions/pore_captions.json` |

**Full 2500-run targets** (set in `N_PER_STRATUM`):

| Tier | Strata | N each |
|------|--------|--------|
| HIGH union | Industrial+Water, Urban+Coastal, Informal+Urban | 250 |
| HIGH pure | Dense Urban, Suburban, Industrial | 300 |
| MID | 7 strata | 100 |
| LOW | 5 strata | 30 |

Advanced CLI:
```bash
python run_pipeline.py --stages bore features   # run specific stages only
python run_pipeline.py --skip bore              # skip a stage already done
python run_pipeline.py --status                 # check what outputs exist
```

---

## BORE — Coordinate Selection

Each coordinate passes four gates per stratum:

1. **OSM Overpass** — queries specific tags (docks, dams, sawmills, coastlines, etc.) in
   known geographic regions to produce a candidate pool
2. **ESA WorldCover** — checks primary + secondary land-cover % thresholds in the 512 m bbox
3. **GHSL** — checks population density, building height, and built-surface bounds
4. **Ocean proximity** *(coastal strata only)* — rejects candidates >50 km from the ocean
   using Natural Earth `ne_10m_ocean.shp`; prevents ESA water class from admitting rivers
   and lakes as ocean-coast candidates

Full threshold table: `docs/strata_table.html`  
Full methodology with reasoning: `docs/strata_methodology.html`

---

## Project Structure

```
run_pipeline.py               # ONLY entry point — change N_PER_STRATUM, then run

scripts/
  bore/
    real_anchor_finder.py     # BORE core — OSM queries, ESA+GHSL filtering, anchor output
    coordinate_filter.py      # Strata config: ESA thresholds, GHSL bounds, HTML parsers
    n25_full_validation.py    # N=25 pass-rate validation across all 18 strata
    n_scale_test.py           # Generic scale test — change STRATUM/N_TARGET/M_POOL only
  pore/
    run_pore.py               # PORE orchestrator — loops coordinates, calls extractors
    feature_extractor.py      # Aggregates all 5 sources into one flat dict per coordinate
    segmap_generator.py       # Renders MS+OSM segmentation maps
  captions/
    groq_caption.py           # Groq LLM caption generator
  extractors/
    worldcover_extractor.py   # ESA WorldCover land-cover % from AWS S3 tiles
    ghsl_extractor.py         # GHSL population, building height, built surface
    solar_atlas_extractor.py  # Global Solar Atlas PVOUT / GHI
    viirs_extractor.py        # VIIRS nighttime lights + NDVI via GEE
    osm_extractor.py          # OSM Overpass feature extraction
    msft_buildings_extractor.py  # Microsoft ML building footprints via Azure Blob
  utils/
    config_loader.py
    logger.py

configs/config.yaml           # Raster paths, API settings, BORE parameters
docs/
  features.html               # 23-feature master table (5 sources)
  2500_coordinate_breakdown.html  # Importance-tier assignments (parsed by pipeline)
  strata_table.html           # 18-stratum quick-reference table
  strata_methodology.html     # Full A–Z methodology per stratum

notebooks/
  day_verify.ipynb            # 18 final anchors — satellite maps + pass-rate charts
  pool_verify.ipynb           # Top-3 candidates per stratum from N=25 pool
  pore_verify.ipynb           # PORE feature + segmap verification
  raw_data_explorer_executed.ipynb  # 5 data sources demonstrated
  method_comparison.ipynb     # Early method comparison (reference only)

outputs/
  csv/
    filtered_strata_sample.csv   # BORE output — coordinates (1 per stratum for N=1)
    pore_features.csv            # PORE output — all features per coordinate
  maps/
    *_segmap.png                 # Segmentation maps — one per coordinate
  captions/
    pore_captions.json           # Groq captions — one per stratum

rasters/
  natural_earth/
    ne_10m_ocean.shp             # Ocean polygon for coastal proximity check (download below)
  ...                            # GHSL + Solar Atlas GeoTIFFs — not in git, download separately
```

---

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in API keys
earthengine authenticate      # one-time GEE auth

# Download Natural Earth ocean shapefile (required for coastal strata filtering)
mkdir -p rasters/natural_earth && cd rasters/natural_earth
wget "https://naciscdn.org/naturalearth/10m/physical/ne_10m_ocean.zip"
unzip ne_10m_ocean.zip && cd ../..
```

**API keys needed** (add to `.env`):
- `GROQ_API_KEY` — https://console.groq.com (free tier)
- `GEE_PROJECT` — your Google Earth Engine project ID

---

## Data Sources

| Source | Features | Method |
|--------|----------|--------|
| ESA WorldCover 2021 | Land-cover class %, dominant class, std | AWS S3 tiles (10 m) |
| GHSL (JRC) | Population/km², building height m, built surface m² | Local GeoTIFF (100 m) |
| Global Solar Atlas | PVOUT, GHI kWh/m²/day | Local GeoTIFF |
| VIIRS via GEE | Nighttime lights nW/cm²/sr, NDVI, reflectance | Google Earth Engine API |
| OSM Overpass | Power infra, buildings, landuse, waterway, amenity | Overpass API |
| Microsoft ML Buildings | Building footprints (geometry only) | Azure Blob quadkey tiles |

---

## Key Design Notes

- **fiona pinned to 1.9.6** — 1.10.x breaks GeoPandas 0.14.3
- **OSM via kumi.systems mirror** — overpass-api.de can be IP-blocked; mirror is primary
- **512 m bbox is fixed** — all strata, all runs, no exceptions
- **Ocean proximity filter** — Urban+Coastal, Coastal+Agricultural, Coastal+Solar-Wind Hybrid
  require `requires_ocean=True` and use `_is_near_ocean()` (50 km threshold, Natural Earth
  ocean polygon) to guarantee genuine sea-coast candidates; ESA water class is type-blind
  so without this filter rivers/lakes pass the ESA gate silently
- **Anchor selection** — SECONDARY_BEST strata pick highest secondary_pct; Hydropower picks
  lowest water% (dam wall visible); all others pick highest primary_pct
- **Null contract** — absent key = no data, never sentinel values;
  `osm_building_count=0` is kept (0 buildings is valid information)
- **GEE auth** — if `earthengine authenticate` expires, re-run it before starting the pipeline
