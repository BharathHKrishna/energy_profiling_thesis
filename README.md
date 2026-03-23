# Energy Profiling Thesis Pipeline

Geospatial energy profiling pipeline for master's thesis.
Extracts energy-relevant features from 7 open data sources
for 5000 globally stratified coordinates, converts to text labels,
and generates energy profile captions via Groq + LangChain.

## Setup
1. Python 3.11 required
2. Create virtual environment: `python3.11 -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install: `pip install -r requirements.txt`
5. Copy `.env` and fill in API keys
6. Create `~/.cdsapirc` with Copernicus CDS credentials

## API Keys Required
- Groq API — https://console.groq.com
- Copernicus CDS — https://cds.climate.copernicus.eu
- Google Earth Engine — https://earthengine.google.com
- Sentinel Hub / Planet — https://www.planet.com

## Data Sources
| Source | Layer | Method |
|---|---|---|
| GHSL | Built surface, height, population, DEGURBA | Local GeoTIFF |
| Global Solar Atlas | DNI, GHI, PVOUT | Local GeoTIFF |
| ESA WorldCover | Global land cover | On-demand AWS S3 tiles |
| VIIRS | Nighttime lights, NDVI, reflectance | Google Earth Engine API |
| OSM | 22 infrastructure features | Overpass API |
| ERA5 | Solar radiation | Copernicus CDS API |

## Key Decisions
- LOD2 dropped — too complex, partially redundant
- CLC replaced by ESA WorldCover — CLC is Europe-only
- HRL Imperviousness replaced by GHSL Built Surface — global coverage
- HRL Tree Cover replaced by VIIRS NDVI via GEE — global coverage
- xAI Grok replaced by Groq free tier (llama-3.3-70b-versatile)
- VIIRS downloaded via Google Earth Engine — EOG/NASA OAuth too complex

## Status
- [x] Day 1 — Environment setup, folder structure, Git initialised
- [x] Day 2 — All APIs registered, Overpass + CDS + Groq tested
- [x] Day 3 — GHSL (4 layers) + Global Solar Atlas (DNI/GHI/PVOUT) downloaded locally
- [x] Day 4 — Google Earth Engine registered, authenticated on server, VIIRS and NDVI queries tested and confirmed working
- [x] Day 5 — Stratified sampling script complete. 5,250 coordinates generated across 10 global strata. CSV and Folium world map saved.