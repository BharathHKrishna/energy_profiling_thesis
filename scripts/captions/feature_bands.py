"""Turn a coordinate's numeric feature record into categorical descriptions.

Written 2026-09-09. The captioning prompt used to hand the language model raw
numbers, which made every caption a restatement of the CSV row. This module
converts each feature to a band ("high built-up", "urban-centre density") so the
caption describes the place in words and the numbers stay in the structured
record where they belong.

Band boundaries come from the source's own published semantics wherever the
source defines them, and only from this dataset's own distribution where it does
not. Which of the two applies is recorded per feature in BAND_PROVENANCE so the
thesis can state it rather than assert it.

Sources with published semantics:
  * VIIRS DNB radiance, MODIS NDVI, MODIS surface reflectance -- interpretation
    bands documented in scripts/extractors/viirs_extractor.py against the
    products' own literature.
  * GHSL population -- the Degree of Urbanisation (GHS-SMOD) standard adopted by
    the EU and endorsed by the UN Statistical Commission: urban centre at
    1,500 inhabitants/km2, urban cluster at 300, rural below that.
  * Building height -- expressed in storeys at the 3 m floor-to-floor estimate
    already used by the floor-area formula (Section 3.5), not an invented band.

Sources without published semantics (land-cover share, degree-days) fall back to
this project's own gate thresholds or to plain quantiles of the completed run,
flagged as such below.
"""

BAND_PROVENANCE = {
    "night_lights":      "VIIRS DNB product interpretation (viirs_extractor.py)",
    "vegetation":        "MODIS NDVI product interpretation (viirs_extractor.py)",
    "reflectance":       "MODIS surface-reflectance interpretation (viirs_extractor.py)",
    "population":        "GHS-SMOD Degree of Urbanisation densities (1,500 / 300 / 50 per km2), unit-corrected x100; the two cuts above urban centre are dataset-derived",
    "building_height":   "net height recovered from AGBH by GHSL own relation, then storeys at 3 m",
    "land_cover_share":  "this project's own class-gate thresholds (Appendix A)",
    "degree_days":       "distribution of the completed 10,000-coordinate run",
}


def _band(value, cuts, labels):
    """cuts ascending; returns labels[i] for the first cut the value falls under."""
    if value is None:
        return None
    for cut, lab in zip(cuts, labels):
        if value < cut:
            return lab
    return labels[-1]


# ── land cover: share of the tile ────────────────────────────────────────────
# No published semantics for "how much built-up is a lot", so these follow the
# class gates this thesis already defends in Appendix A (Dense Urban at 80 %,
# Suburban at 50 %, the union classes' secondary conditions around 15-25 %).
# Phrased to read naturally as "<band> the tile", so the model is handed prose
# rather than a label it has to conjugate itself.
def cover_band(pct):
    return _band(pct, [2, 15, 40, 65, 85],
                 ["essentially none of", "a trace of", "some of",
                  "a substantial share of", "an extensive share of",
                  "nearly all of"])


# ESA WorldCover class keys as they appear in the CSV, in readable form.
COVER_NAMES = {
    "built_up":    "built-up surface",
    "tree_cover":  "tree cover",
    "cropland":    "cropland",
    "water":       "water",
    "bare_sparse": "bare or sparse ground",
    "grassland":   "grassland",
    "shrubland":   "shrubland",
    "mangrove":    "mangrove",
    "wetland":     "wetland",
    "moss_lichen": "moss and lichen",
    "snow_ice":    "snow and ice",
}


def cover_name(key):
    if not key:
        return None
    return COVER_NAMES.get(key, str(key).replace("_", " "))


# -- GHSL population ---------------------------------------------------------
# ghsl_population_bbox_total is the overlap-weighted SUM of the GHS-POP cells the
# tile covers, i.e. the people actually inside the 512 m box. GHS-POP's 100 m
# product stores an absolute count per cell, so summing is the right operation.
# The older mean-of-cells column (ghsl_population_per_km2) is kept because the
# class gates read it as a per-cell count, but it is not what this band uses.
#
# Thresholds are GHS-SMOD's published Degree of Urbanisation densities, converted
# once into this tile's own units rather than converting every coordinate: a
# 512 m tile is 0.262144 km2, so 1,500/km2 is 393 people in the tile, 300/km2 is
# 79, and 50/km2 is 13. The two cuts above the urban-centre boundary are this
# project's own, since GHS-SMOD does not subdivide by density above 1,500/km2.
TILE_KM2 = 0.512 * 0.512


def population_band(bbox_total, tile_km2=TILE_KM2):
    """bbox_total is people inside the 512 m tile, not a density."""
    if bbox_total is None:
        return None
    cuts = [d * tile_km2 for d in (50, 300, 1500, 5000, 15000)]
    return _band(bbox_total, cuts,
                 ["effectively unpopulated", "sparsely settled",
                  "urban-cluster density", "urban-centre density",
                  "dense urban-centre density", "very dense urban-centre density"])


# -- GHSL building height, expressed in storeys -------------------------------
# ghsl_building_height_m is now the NET height (building volume over built
# surface), recovered in the extractor from the AGBH raster, which stores a gross
# height that is not the height of anything standing. Verified against fabric
# with regulated heights: Barcelona's Eixample 23.1 m against a documented
# 20-24 m, Paris 17.7 m against 18-20 m, Amsterdam's canal houses 16.0 m against
# 14-17 m. Section 6.9 describes the defect this replaced.
def height_band(metres):
    if metres is None:
        return None
    return _band(metres / 3.0, [0.5, 1.5, 3.0, 8.0],
                 ["negligible built volume", "single-storey", "low-rise",
                  "mid-rise", "high-rise"])


# ── VIIRS nighttime lights, product interpretation ───────────────────────────
def night_lights_band(radiance):
    return _band(radiance, [0.5, 1.0, 10.0, 100.0],
                 ["dark, uninhabited or energy-poor", "very rural, isolated light",
                  "small town or sparse settlement", "suburban or industrial lighting",
                  "major city-centre lighting"])


# ── MODIS NDVI, product interpretation ───────────────────────────────────────
def vegetation_band(ndvi):
    return _band(ndvi, [0.0, 0.1, 0.2, 0.4, 0.6],
                 ["water, snow or cloud", "bare desert or rock",
                  "degraded or semi-arid land", "shrubland or sparse vegetation",
                  "cropland or woodland", "dense forest"])


# ── MODIS surface reflectance ────────────────────────────────────────────────
# The cut points are the source's (0.1 / 0.2 / 0.5), but the wording is
# deliberately neutral rather than the extractor's ecological reading. That
# reading assigns 0.1-0.2 to "dense vegetation", and 72.8 % of this urban-weighted
# sample falls in that interval, so keeping it would have captions calling
# near-total built-up tiles densely vegetated, contradicting their own land cover.
# Brightness is what the band actually measures, so brightness is what it says.
def reflectance_band(refl):
    return _band(refl, [0.1, 0.2, 0.5],
                 ["a very dark surface", "a dark surface",
                  "a moderately bright surface", "a highly reflective surface"])


# ── degree days: no published severity bands, so quantiles of this run ───────
def heating_band(hdd):
    return _band(hdd, [100, 1000, 2500, 5000],
                 ["no meaningful heating season", "a mild heating season",
                  "a moderate heating season", "a cold heating season",
                  "a severe heating season"])


def cooling_band(cdd):
    return _band(cdd, [50, 500, 1500],
                 ["no meaningful cooling season", "a mild cooling season",
                  "a hot cooling season", "a severe cooling season"])


def describe(features: dict) -> dict:
    """Return the banded description of one coordinate's feature record."""
    g = features.get
    out = {
        "dominant_cover":   cover_name(g("wc_dominant_class")),
        "dominant_share":   cover_band(g("wc_dominant_pct")),
        "built_up":         cover_band(g("wc_built_up_pct")),
        "tree_cover":       cover_band(g("wc_tree_cover_pct")),
        "water":            cover_band(g("wc_water_pct")),
        "cropland":         cover_band(g("wc_cropland_pct")),
        "bare_sparse":      cover_band(g("wc_bare_sparse_pct")),
        "population":       population_band(g("ghsl_population_bbox_total")),
        "building_height":  height_band(g("ghsl_building_height_m")),
        "night_lights":     night_lights_band(g("viirs_ntl_nw_cm2_sr")),
        "vegetation":       vegetation_band(g("viirs_ndvi")),
        "reflectance":      reflectance_band(g("viirs_surface_refl")),
        "heating_season":   heating_band(g("climate_hdd")),
        "cooling_season":   cooling_band(g("climate_cdd24")),
        "demand_regime":    g("demand_regime"),
    }
    return {k: v for k, v in out.items() if v is not None}
