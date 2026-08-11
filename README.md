# Energy Profiling Thesis Pipeline

Geospatial energy profiling pipeline for Master's thesis in Energy Informatics.

Extracts energy-relevant features from 7 open data sources for globally stratified
coordinates across 16 land-cover strata, verified via satellite imagery, and generates
per-coordinate maps and LLM captions.

---

## Pipeline Stages

`run_pipeline.py` runs 4 stages in order:

| Stage | What it does | Output |
|-------|-------------|--------|
| `bore` | Live OSM/ESA/GHSL discovery of real candidate coordinates per stratum | `outputs/csv/filtered_strata_sample.csv` |
| `select` | Seeded random subsample down to the tier-balanced target plan | `outputs/csv/filtered_strata_sample_selected.csv` |
| `osm_carve` | Batch-carve local OSM tiles for every selected coordinate, once | `osm_planet/tiles/*.osm.pbf` |
| `stream` | Per-coordinate: extract 7 sources + demand + demand_score, render 4 maps, generate a caption — extract/maps and captions run as two separate pools, a coordinate's caption fires the moment its own extraction finishes | `outputs/csv/pore_features.csv`, `outputs/maps/{segmaps,detection}/*.png`, `outputs/captions/pore_captions.json` |

## Running the Pipeline

```bash
# 1. Edit N_PER_STRATUM in run_pipeline.py (the ONLY thing you change) —
#    it drives BOTH the bore search quota and the select target count.
# 2. Run:
python run_pipeline.py
```

All 4 stages run in order automatically, resume-safe throughout (a stage skips work
already done — cached BORE candidates, already-extracted coordinates, already-generated
captions — rather than redoing it).

Advanced CLI:
```bash
python run_pipeline.py --stages select osm_carve stream   # run specific stages only
python run_pipeline.py --skip bore                        # skip a stage already done
python run_pipeline.py --status                           # check what outputs exist
python run_pipeline.py --workers 8 --caption-workers 10   # override pool sizes
```

**Important caveat**: `bore`'s resume logic truncates `filtered_strata_sample.csv` to a
header-only file at the start of every run, then only re-appends rows found *beyond* what
was already cached for that stratum. If every stratum's quota (`N_PER_STRATUM`) is already
met by the existing pool, nothing gets re-appended — running `bore` again in that state
wipes the discovered pool to zero rows for no benefit. If you already have a populated
pool and just want to change what gets *selected* from it, run
`python run_pipeline.py --stages select osm_carve stream` instead and leave `bore` alone.

**Current target plan** (16 strata, `docs/10000_coordinate_breakdown.html`):

| Tier | Strata | N each | Tier total |
|------|--------|--------|------------|
| HIGH | Suburban, Dense Urban, Informal + Urban, Industrial | 1600 | 6,400 |
| MID  | Suburban + Agricultural, Industrial + Water, Hydropower Reservoir, Agrivoltaics (Solar + Farmland), Industrial + Arid, Industrial + Forest, Urban + Coastal | 500 | 3,500 |
| LOW  | Agricultural + Water, Mangrove + Industrial, Data Centre + Industrial, Coastal + Solar-Wind Hybrid, Coastal + Agricultural | 20 | 100 |
| | | **Total** | **10,000** |

Two strata — `Utility-scale Solar Farm` and `Airport / Aviation` — were removed entirely
(2026-08-10) after a 100-samples-per-stratum visual review of real satellite imagery found
both strata's BORE classification rule produces a large fraction of non-matching results at
scale. See `notebooks/stratum_reliability_review.ipynb` for the review and
`docs/10000_coordinate_breakdown.html`'s changelog note for the full reasoning.

---

## BORE — Coordinate Discovery

A 3-phase search per stratum in `real_anchor_finder.py`:

1. **Phase 1 — chunked OSM discovery**: polite, small Overpass queries (continent-grid
   splitting, avoids single-query truncation), each candidate checked against ESA
   WorldCover land-cover % thresholds and GHSL population/height/built-surface bounds
   (`coordinate_filter.py` — see `docs/strata_methodology.html` for the full per-stratum
   threshold table and reasoning).
2. **Phase 2 — densify**: for strata whose real-world footprint is a large contiguous zone
   (not a single structural point like a dam wall or coastline), tile a 7×7 grid of new
   candidate points around every accepted anchor and re-check the same gate.
3. **Phase 3 — GHSL seed**: for the 3 largest urban strata, sample points inside real
   satellite-measured GHSL Urban Centre polygons, independent of OSM tagging.

All three phases share the identical, unmodified ESA+GHSL gate — later phases only change
which candidates get *proposed* to it, never loosen the standard itself.

Ocean proximity filter (coastal strata only): rejects candidates >50 km from the ocean
using Natural Earth `ne_10m_ocean.shp`, since ESA's water class doesn't distinguish ocean
from rivers/lakes.

Full methodology: `docs/strata_methodology.html`. Confirmed real discovery counts:
`docs/scale_ceiling_analysis.html`.

---

## PORE — Per-Coordinate Extraction

7 data sources merged into one flat feature dict per coordinate
(`scripts/pore/feature_extractor.py`), plus two derived demand pieces:

| Source | Features | Method |
|--------|----------|--------|
| ESA WorldCover 2021 | Land-cover class %, dominant class, std | Live AWS S3 tiles (10 m), no local persistence |
| GHSL (JRC) | Population/km², building height m, built surface m² | Local GeoTIFF (100 m) |
| Global Solar Atlas | PVOUT, GHI kWh/m²/day | Local GeoTIFF |
| Wind Atlas | Speed + power density @ 100m | Live Open-Meteo ERA5 API |
| VIIRS via GEE | Nighttime lights nW/cm²/sr, NDVI, reflectance | Google Earth Engine API |
| Climate via GEE | HDD/CDD degree-days, mean annual temp | Google Earth Engine API (ERA5-Land, ~11km native res) |
| OSM (offline) | Buildings, infra, landuse | Local pre-carved 512m tiles (`osm_carve` stage), live fallback if a tile is missing |

**Demand**: `heating_MWh`/`cooling_MWh` (physical, Eurostat-fit) — exactly one of the two per
coordinate, chosen by `demand_regime`, never both, never a fabricated zero for the other.
`demand_score`/`demand_tier` (0-100 synthetic score, single-cell) — a separate formula with
its own OSM-gate read. **"Energy demand" unqualified always means the score/tier, never the
physical MWh formulas** — see `docs/pore_extraction_methodology.html` for the full formulas
and the null-contract reasoning.

Full methodology for `select`, `osm_carve`, and `stream`:
`docs/select_methodology.html`, `docs/osm_carve_methodology.html`,
`docs/pore_extraction_methodology.html`.

---

## Project Structure

```
run_pipeline.py               # ONLY entry point — change N_PER_STRATUM, then run

scripts/
  bore/
    real_anchor_finder.py     # BORE core — 3-phase OSM/ESA/GHSL discovery
    coordinate_filter.py      # Strata config: ESA thresholds, GHSL bounds, HTML parsers
  pore/
    feature_extractor.py      # Aggregates all 7 sources + demand + demand_score
    segmap_generator.py       # Segmentation-style maps (base + GHSL pop overlay)
    detection_map.py          # Detection-style maps (base + GHSL pop overlay)
  captions/
    groq_caption.py           # Groq LLM caption generator
  extractors/
    worldcover_extractor.py, ghsl_extractor.py, solar_atlas_extractor.py,
    wind_atlas_extractor.py, viirs_extractor.py, climate_extractor.py,
    osm_extractor.py, osm_offline.py, osm_batch_extract.py,
    demand_extractor.py, demand_score.py, msft_buildings_extractor.py
  utils/
    config_loader.py, logger.py, audit_stream_output.py   # (audit_stream_output.py is a
                                                            # standalone post-run completeness
                                                            # checker, not imported by the
                                                            # pipeline itself — see its own
                                                            # docstring)

configs/config.yaml           # Raster paths, GEE project, Groq model — see the file's own
                               # inline notes for what's actually read vs. historical
docs/
  features.html                       # Feature master table
  10000_coordinate_breakdown.html     # Current tier plan (parsed by coordinate_filter.py)
  strata_methodology.html             # Full BORE methodology per stratum
  select_methodology.html             # select stage methodology
  osm_carve_methodology.html          # osm_carve stage methodology
  pore_extraction_methodology.html    # stream stage methodology
  scale_ceiling_analysis.html         # Confirmed real discovery counts

notebooks/
  pipeline_smoke_test_16coord.ipynb   # Latest end-to-end smoke test (1/stratum)
  stratum_reliability_review.ipynb    # 100-samples-per-stratum visual QC
  bore_scaleup_map.ipynb              # BORE discovery pixel-audit
  energy_demand_formula.ipynb         # demand_score derivation/validation
  heat_demand_sandbox.ipynb, cooling_demand_sandbox.ipynb   # heat/cool formula derivation
  zensus_explorer.ipynb               # German census data exploration (standalone, Germany-only)
  method_comparison.ipynb, day_verify.ipynb, raw_data_explorer_executed.ipynb  # reference/historical

outputs/
  csv/
    filtered_strata_sample.csv          # BORE's full discovered pool
    filtered_strata_sample_selected.csv # select's tier-balanced output — stream reads this
    pore_features.csv                   # stream's feature output
  maps/
    segmaps/*.png, detection/*.png      # 4 maps per coordinate
    stratum_review/*.png                # ad-hoc contact-sheet grids from the reliability review
  captions/
    pore_captions.json                  # Groq captions, one per stratum

rasters/
  natural_earth/ne_10m_ocean.shp        # Ocean polygon for coastal proximity check
  ghsl/, global_solar_atlas/            # GeoTIFFs — not in git, download separately
```

---

## Setup

The pipeline currently runs against **system Python** — `venv/` is not present (an earlier
one was broken beyond repair, dangling symlink to a nonexistent interpreter). `venv_bore/`
exists separately, scoped to BORE-stage dependencies only (has `fiona`, missing `groq`/
`python-dotenv`/`earthengine-api` that `stream` needs).

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in API keys
earthengine authenticate      # one-time GEE auth

# Download Natural Earth ocean shapefile (required for coastal strata filtering)
mkdir -p rasters/natural_earth && cd rasters/natural_earth
wget "https://naciscdn.org/naturalearth/10m/physical/ne_10m_ocean.zip"
unzip ne_10m_ocean.zip && cd ../..
```

`osmium-tool` (the CLI, not the `osmium` Python package) is also required for
`osm_carve`/OSM feature reads — if not available via `apt-get install osmium-tool`
(needs root), it can be installed user-space via `micromamba install -c conda-forge
osmium-tool` and the binary path repointed in `osm_batch_extract.py`/`osm_extractor.py`'s
`OSMIUM` constant.

**API keys needed** (add to `.env`):
- `GROQ_API_KEY` — https://console.groq.com (free tier)
- `GEE_PROJECT` — your Google Earth Engine project ID (also set in `configs/config.yaml`'s `api.gee_project`)

---

## Key Design Notes

- **512 m bbox is fixed** — all strata, all runs, no exceptions
- **Ocean proximity filter** — coastal strata require `_is_near_ocean()` (50 km threshold,
  Natural Earth ocean polygon) to guarantee genuine sea-coast candidates; ESA's water class
  is type-blind so without this filter rivers/lakes pass the ESA gate silently
- **Null contract** — absent key = no data, never a sentinel value;
  `osm_building_count=0` is the one deliberate exception (0 buildings is valid information)
- **OSM is fully offline for PORE** — `stream` never makes a live Overpass call; it reads
  pre-carved local tiles (fast path, ~5-12ms) with a live-fallback only if a tile is
  genuinely missing (slow, full-planet rescan). BORE itself still uses live Overpass, since
  it needs a global search, not a fixed coordinate.
- **GEE auth** — if `earthengine authenticate` expires, re-run it before starting the pipeline
- **Demand vs. energy demand** — "energy demand" always means the 0-100 `demand_score`/
  `demand_tier`, never the physical `heating_MWh`/`cooling_MWh` formulas — a distinction
  enforced throughout the codebase and docs
