# Energy Profiling Thesis Pipeline

Geospatial energy profiling pipeline for Master's thesis in Energy Informatics.

Extracts energy-relevant features from 6 open data sources for globally stratified
coordinates across 16 land-cover strata, verified via satellite imagery, and generates
per-coordinate maps and LLM captions.

---

## Pipeline Stages

`run_pipeline.py` runs 3 stages in order:

| Stage | What it does | Output |
|-------|-------------|--------|
| `bore` | Live OSM/ESA/GHSL discovery of real candidate coordinates per stratum | `outputs/csv/filtered_strata_sample.csv` |
| `select` | Seeded random subsample down to the tier-balanced target plan | `outputs/csv/filtered_strata_sample_selected.csv` |
| `stream` | Per-coordinate: fetch its own OSM tile (live Overpass), extract 6 sources + demand, render 4 maps, generate a caption — extract/maps and captions run as two separate pools, a coordinate's caption fires the moment its own extraction finishes | `outputs/csv/pore_features.csv`, `outputs/maps/{segmaps,detection}/*.png`, `outputs/captions/pore_captions.json` |

`osm_carve` no longer exists as its own upfront batch stage (merged into `stream`
2026-08-24) — each `stream` worker fetches its own coordinate's OSM tile via the live
Overpass API (multiple public mirrors, randomized order) as the first step of its own
per-coordinate work, instead of reading a tile a separate stage pre-carved from a local
planet file.

## Running the Pipeline

```bash
# 1. Edit N_PER_STRATUM in run_pipeline.py (the ONLY thing you change) —
#    it drives BOTH the bore search quota and the select target count.
# 2. Run:
python run_pipeline.py
```

All 3 stages run in order automatically, resume-safe throughout (a stage skips work
already done — cached BORE candidates, already-extracted coordinates, already-generated
captions — rather than redoing it).

Advanced CLI:
```bash
python run_pipeline.py --stages select stream              # run specific stages only
python run_pipeline.py --skip bore                          # skip a stage already done
python run_pipeline.py --status                             # check what outputs exist
python run_pipeline.py --workers 16 --caption-workers 10   # override pool sizes
```

**Important caveat**: `bore`'s resume logic truncates `filtered_strata_sample.csv` to a
header-only file at the start of every run, then only re-appends rows found *beyond* what
was already cached for that stratum. If every stratum's quota (`N_PER_STRATUM`) is already
met by the existing pool, nothing gets re-appended — running `bore` again in that state
wipes the discovered pool to zero rows for no benefit. If you already have a populated
pool and just want to change what gets *selected* from it, run
`python run_pipeline.py --stages select stream` instead and leave `bore` alone.

**Current target plan** (16 strata, `docs/10000_coordinate_breakdown.html`):

| Tier | Strata | N each | Tier total |
|------|--------|--------|------------|
| HIGH | Suburban, Dense Urban, Informal + Urban, Industrial | 1600 | 6,400 |
| MID  | Suburban + Agricultural, Industrial + Water, Hydropower Reservoir, Agriculture / Agrivoltaics, Industrial + Arid, Industrial + Forest, Urban + Coastal | 500 | 3,500 |
| LOW  | Agricultural + Water, Mangrove + Industrial, Data Centre + Industrial, Coastal + Solar-Wind Hybrid, Coastal + Agricultural | 20 | 100 |
| | | **Total** | **10,000** |

Two strata — `Utility-scale Solar Farm` and `Airport / Aviation` — were removed entirely
(2026-08-10) after a 100-samples-per-stratum visual review of real satellite imagery found
both strata's BORE classification rule produces a large fraction of non-matching results at
scale. `Agrivoltaics (Solar + Farmland)` was renamed to `Agriculture / Agrivoltaics`
(2026-08-27) after a separate visual reliability audit found the class is genuinely mostly
real farmland, but the old name's "solar" half badly overclaimed what's actually visible.
See `notebooks/stratum_reliability_review.ipynb` for the removal review and
`docs/10000_coordinate_breakdown.html`'s changelog note for the full reasoning.

The full 10,000-coordinate run completed 2026-08-28: all 10,000 coordinates produced a
complete feature record, four maps, and a caption, zero extraction failures, in 598.2
minutes, drawn from a discovery pool of 223,842 confirmed anchors (deduplicated from a
raw 239,235 — 15,224 coordinates had satisfied more than one class's gate at once, every
instance among the three adjacent density-based classes Suburban/Dense Urban/Informal +
Urban).

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

6 data sources merged into one flat feature dict per coordinate
(`scripts/pore/feature_extractor.py`), extracted concurrently (a thread pool, not
sequentially) since every source is I/O wait, not CPU work, plus a derived demand piece:

| Source | Features | Method |
|--------|----------|--------|
| ESA WorldCover 2021 | Land-cover class %, dominant class, std | Live AWS S3 tiles (10 m), no local persistence |
| GHSL (JRC) | Population/km², building height m, built surface m² | Local GeoTIFF (100 m) |
| Global Solar Atlas | PVOUT, GHI kWh/m²/day | Local GeoTIFF |
| VIIRS via GEE | Nighttime lights nW/cm²/sr, NDVI, reflectance | Google Earth Engine API |
| Climate via GEE | HDD/CDD degree-days, mean annual temp | Google Earth Engine API (ERA5-Land, ~11km native res) |
| OSM (live) | Buildings, infra, landuse | Live Overpass API, own coordinate's tile fetched inline per-coordinate |

A wind resource source (Global Wind Atlas / Open-Meteo ERA5) was piloted and removed
2026-08-27 — not a data-quality issue, but Open-Meteo's free-tier request quota would
have taken multiple days to clear for 10,000 coordinates on this one source alone. See
`thesis/conclusion.tex`'s Future Work section for the full reasoning and what would
resolve it.

**Demand**: `heating_MWh`/`cooling_MWh` — exactly one of the two per coordinate, chosen by
`demand_regime` (a geographic-regression formula, see `demand_extractor.py`), never both,
never a fabricated zero for the other. The earlier synthetic 0-100 `demand_score`/
`demand_tier` formula was retired entirely — this pipeline no longer computes it.

Full methodology for `select` and `stream`:
`docs/select_methodology.html`, `docs/pore_extraction_methodology.html`.

---

## Project Structure

```
run_pipeline.py               # ONLY entry point — change N_PER_STRATUM, then run

scripts/
  bore/
    real_anchor_finder.py     # BORE core — 3-phase OSM/ESA/GHSL discovery
    coordinate_filter.py      # Strata config: ESA thresholds, GHSL bounds, HTML parsers
  pore/
    feature_extractor.py      # Aggregates all 6 sources + demand, concurrently per coordinate
    segmap_generator.py       # Segmentation-style maps (base + GHSL pop overlay)
    detection_map.py          # Detection-style maps (base + GHSL pop overlay)
  captions/
    kit_caption.py            # KIT KI-Toolbox LLM caption generator (the only backend)
    caption_prompt.py         # Shared prompt template (build_prompt()), no API/key/network
  extractors/
    worldcover_extractor.py, ghsl_extractor.py, solar_atlas_extractor.py,
    viirs_extractor.py, climate_extractor.py,
    osm_extractor.py, osm_overpass_extract.py, osm_batch_extract.py,
    demand_extractor.py, msft_buildings_extractor.py
  utils/
    config_loader.py, logger.py, audit_stream_output.py   # (audit_stream_output.py is a
                                                            # standalone post-run completeness
                                                            # checker, not imported by the
                                                            # pipeline itself — see its own
                                                            # docstring)

configs/config.yaml           # Raster paths, GEE project, KIT model — see the file's own
                               # inline notes for what's actually read vs. historical
docs/
  features.html                       # Feature master table
  10000_coordinate_breakdown.html     # Current tier plan (parsed by coordinate_filter.py)
  strata_methodology.html             # Full BORE methodology per stratum
  select_methodology.html             # select stage methodology
  osm_carve_methodology.html          # superseded — kept as historical record only
  pore_extraction_methodology.html    # stream stage methodology
  scale_ceiling_analysis.html         # Confirmed real discovery counts

notebooks/
  pipeline_smoke_test_16coord.ipynb   # Earlier end-to-end smoke test (1/stratum)
  stratum_reliability_review.ipynb    # 100-samples-per-stratum visual QC
  bore_scaleup_map.ipynb              # BORE discovery pixel-audit
  energy_demand_formula.ipynb         # demand formula derivation/validation
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
    pore_captions.json                  # KIT captions, one per coordinate

rasters/
  natural_earth/ne_10m_ocean.shp        # Ocean polygon for coastal proximity check
  ghsl/, global_solar_atlas/            # GeoTIFFs — not in git, download separately
```

---

## Setup

The pipeline currently runs against **system Python** — `venv/` is not present (an earlier
one was broken beyond repair, dangling symlink to a nonexistent interpreter). `venv_bore/`
exists separately, scoped to BORE-stage dependencies only (has `fiona`, missing `openai`/
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

`osmium-tool` (the CLI, not the `osmium` Python package) is required for BORE's own
per-tile writes — if not available via `apt-get install osmium-tool` (needs root), it can
be installed user-space via `micromamba install -c conda-forge osmium-tool` and the
binary path repointed where referenced.

**API keys needed** (add to `.env`):
- `KIT_API_KEY` — https://ki-toolbox.scc.kit.edu/ (KIT account required; see `kit_caption.py` docstring for setup)
- `GEE_PROJECT` — your Google Earth Engine project ID (also set in `configs/config.yaml`'s `api.gee_project`)

---

## Key Design Notes

- **512 m bbox is fixed** — all strata, all runs, no exceptions
- **Ocean proximity filter** — coastal strata require `_is_near_ocean()` (50 km threshold,
  Natural Earth ocean polygon) to guarantee genuine sea-coast candidates; ESA's water class
  is type-blind so without this filter rivers/lakes pass the ESA gate silently
- **Null contract** — absent key = no data, never a sentinel value;
  `osm_building_count=0` is the one deliberate exception (0 buildings is valid information)
- **OSM is live for both BORE and PORE** — `stream` fetches each coordinate's own tile via
  the live Overpass API inline, as the first step of its own per-coordinate work; there is
  no separate offline pre-carve stage anymore
- **GEE auth** — if `earthengine authenticate` expires, re-run it before starting the pipeline
- **Concurrent per-coordinate extraction** — `feature_extractor.py` fires all 6 sources (plus
  the demand formula's own floor-area read) in a thread pool rather than sequentially, since
  every one is I/O wait; a coordinate's total extraction time becomes roughly its slowest
  single source, not the sum of all of them
