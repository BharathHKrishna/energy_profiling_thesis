# Energy Profiling Thesis Pipeline

Geospatial energy profiling pipeline for master's thesis.
Extracts 38 energy-relevant features from 7 open data sources
for 1000 globally stratified coordinates, converts to text labels,
and generates energy profile captions via Grok + LangChain.

## Setup
1. Python 3.11 required
2. Create virtual environment: `python3.11 -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install: `pip install -r requirements.txt`
5. Copy `.env` and fill in API keys
6. Create `~/.cdsapirc` with your Copernicus CDS credentials

## API Keys Required
- Grok API (xAI) — https://console.x.ai
- Copernicus CDS — https://cds.climate.copernicus.eu
- Sentinel Hub — https://www.sentinel-hub.com
- Google Maps Static (optional) — https://developers.google.com/maps

## Status
- [x] Day 1 — Environment setup, folder structure, Git initialised
- [x] Day 2 — All APIs registered, Overpass + CDS + Grok tested and confirmed live