"""Heating + cooling demand for a BORE 512 m cell.

SINGLE-BBOX (no 9-tile): trusts BORE's tested cells — the formula converts
whatever building GHSL reports for that exact cell into annual energy demand.

    HEAT = floor_area x 0.3143 x HDD_18^0.75                             [MWh/yr]
    COOL = floor_area x 0.000571 x CDD_24^1.5                            [MWh/yr]

where floor_area = GHSL_built x (height / 3).  HDD/CDD are annual degree-days.

Constants refit 2026-07-31 against real Eurostat panel data (nrg_chdd_a x
nrg_d_hhq x demo_pjan, 2010-2024, EU/EEA countries; see
docs/demand_work_summary_2026-07-31.md for the full derivation):
    HEATING  R2 = 0.64  (335 country-years, space heating only, Luxembourg excluded)
    COOLING  R2 = 0.85  (247 country-years, space cooling only, Luxembourg excluded)

Both are fit at kWh/capita and converted to kWh/m2 via an assumed 35 m2/person
EU floor-area figure — that conversion constant is NOT independently verified
(no working bulk floor-area-per-capita source was found). It only affects the
absolute level, not the fitted HDD/CDD exponents.

Known limitations (real, unresolved — see docs/demand_work_summary_2026-07-31.md §6):
  - HEATING has no domestic hot water (DHW) term. The old formula's
    `+ population x 700 kWh/yr` was dropped, not replaced — Eurostat's
    nrg_d_hhq has a water-heating category (FC_OTH_HH_E_WH) that was never
    pulled. This is a real gap, not an oversight to silently patch back in.
  - COOLING is fit to real METERED electricity in a low-AC-ownership market
    (Europe) — it estimates realistic actual consumption, not climate-driven
    NEED. Expect it to read low relative to a "full need" estimate in hot,
    high-AC climates.
"""
import sys, math
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

from scripts.extractors.ghsl_extractor import extract_ghsl_features

# ---- calibrated constants (Eurostat refit 2026-07-31) ---------------------
FLOOR_PER_CAP = 35.0            # assumed EU m2/person — unverified placeholder, see docstring
HEAT_K      = 11 / FLOOR_PER_CAP     # = 0.3143 kWh/m2 per HDD18^HEAT_P
HEAT_P      = 0.75
COOL_K      = 0.02 / FLOOR_PER_CAP   # = 0.000571 kWh/m2 per CDD24^COOL_P
COOL_P      = 1.5
STOREY_H    = 3.0       # m per floor (footprint -> floor area)
HEAT_BASE   = 18        # HDD counts temperature below this
COOL_BASE   = 24        # CDD counts temperature above this


def _bbox(lat, lon, size_m=512):
    h = size_m / 2.0
    dlat = h / 111320.0
    dlon = h / (111320.0 * math.cos(math.radians(lat)))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def _climate_single(lat, lon):
    """HDD (base 18) + CDD (base 24) + annual mean temp for one 512 m cell, ERA5-Land 2023."""
    import ee
    s, n, w, e = _bbox(lat, lon)
    tc = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
          .filterDate("2023-01-01", "2023-12-31").select("temperature_2m")
          .map(lambda im: im.subtract(273.15)))
    hdd  = tc.map(lambda im: im.multiply(-1).add(HEAT_BASE).max(0)).sum()
    cdd  = tc.map(lambda im: im.subtract(COOL_BASE).max(0)).sum()
    mean = tc.mean()
    reg = ee.Geometry.Rectangle([w, s, e, n])
    rr = lambda img: img.reduceRegion(ee.Reducer.mean(), reg, 500).getInfo().get("temperature_2m")
    return rr(hdd), rr(cdd), rr(mean)


def _demand_regime(mean_temp_c):
    """Plain-language regime label from annual mean temperature. Drives which single
    demand number _demand() below actually reports (changed 2026-08-10 — previously
    both heating_MWh and cooling_MWh were always computed and returned together,
    reasoning that a single annual mean can mask real heating AND cooling need on
    different days; now deliberately simplified to one value per coordinate, tied to
    its regime, per explicit instruction).
    """
    if mean_temp_c is None:
        return None
    if mean_temp_c < HEAT_BASE:
        return "heating"
    if mean_temp_c > COOL_BASE:
        return "cooling"
    return "comfort — no major heating or cooling needed"


def _demand(built, height, hdd, cdd24, mean_temp_c=None):
    """Reports exactly one of heating_MWh / cooling_MWh, chosen by regime — never both,
    never a 0 sentinel for the other. "heating" regime -> heating_MWh only; "cooling" ->
    cooling_MWh only; "comfort" (or regime unknown, mean_temp_c not available) -> neither
    key is included (null-contract: absent key = no data, not a fabricated NaN/0)."""
    floor = built * (max(height, STOREY_H) / STOREY_H)
    regime = _demand_regime(mean_temp_c)
    result = dict(demand_floor_m2=round(floor), demand_regime=regime)
    if regime == "heating":
        heat = floor * HEAT_K * (hdd ** HEAT_P) / 1000.0
        result["heating_MWh"] = round(heat, 1)
    elif regime == "cooling":
        cool = floor * COOL_K * (cdd24 ** COOL_P) / 1000.0
        result["cooling_MWh"] = round(cool, 1)
    return result


def demand_from_features(built_m2, height_m, hdd18, cdd24, mean_temp_c=None):
    """Heating OR cooling demand (MWh/yr) — exactly one, chosen by regime, from
    already-extracted feature values —
    no GHSL read, no GEE call. For use in the main PORE pipeline (feature_extractor.py),
    where ghsl_built_surface_m2 / ghsl_building_height_m / climate_hdd / climate_cdd24 /
    climate_mean_temp_c are already fetched by extract_ghsl_features() and
    extract_climate_features().

    Null contract: needs both a built-surface figure and at least one degree-day figure
    to mean anything — if built_m2 is None (no GHSL data) or both hdd18 and cdd24 are
    None (no climate data, e.g. over open water), returns {} rather than a 0 MWh sentinel.
    mean_temp_c is optional — demand_regime is just omitted (None) if it's not passed.
    """
    if built_m2 is None or (hdd18 is None and cdd24 is None):
        return {}
    height_m = height_m or STOREY_H   # no height data -> assume single storey
    return _demand(built_m2, height_m, hdd18 or 0, cdd24 or 0, mean_temp_c)


def extract_demand(lat, lon):
    """Heating + cooling demand (MWh/yr) for the 512 m cell at (lat, lon).

    Standalone/CLI path — makes its own GHSL read + live GEE climate call.
    The pipeline path (feature_extractor.py) uses demand_from_features() instead,
    reusing values already fetched by extract_ghsl_features()/extract_climate_features().
    """
    s, n, w, e = _bbox(lat, lon)
    g = extract_ghsl_features(lat, lon, s, n, w, e)
    hdd, cdd24, mean_t = _climate_single(lat, lon)
    return demand_from_features(g.get("ghsl_built_surface_m2"), g.get("ghsl_building_height_m"),
                                hdd, cdd24, mean_t)


def extract_demand_batch(coords):
    """Many BORE coords -> demand each. Climate BATCHED (one GEE call), GHSL local.

    coords: iterable of (lat, lon). Returns list of dicts (same order).
    """
    import ee
    coords = list(coords)
    tc = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
          .filterDate("2023-01-01", "2023-12-31").select("temperature_2m")
          .map(lambda im: im.subtract(273.15)))
    hdd_img  = tc.map(lambda im: im.multiply(-1).add(HEAT_BASE).max(0)).sum().rename("hdd")
    cdd_img  = tc.map(lambda im: im.subtract(COOL_BASE).max(0)).sum().rename("cdd")
    mean_img = tc.mean().rename("meant")
    both = hdd_img.addBands(cdd_img).addBands(mean_img)
    feats = []
    for i, (la, lo) in enumerate(coords):
        s, n, w, e = _bbox(la, lo)
        feats.append(ee.Feature(ee.Geometry.Rectangle([w, s, e, n]), {"id": i}))
    fc = ee.FeatureCollection(feats)
    res = both.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=500).getInfo()
    clim = {f["properties"]["id"]: (f["properties"].get("hdd"), f["properties"].get("cdd"),
                                    f["properties"].get("meant"))
            for f in res["features"]}
    out = []
    for i, (la, lo) in enumerate(coords):
        s, n, w, e = _bbox(la, lo)
        g = extract_ghsl_features(la, lo, s, n, w, e)
        hdd, cdd24, mean_t = clim.get(i, (0, 0, None))
        d = demand_from_features(g.get("ghsl_built_surface_m2"), g.get("ghsl_building_height_m"),
                                 hdd, cdd24, mean_t)
        d["lat"], d["lon"] = la, lo
        out.append(d)
    return out


if __name__ == "__main__":
    import ee
    ee.Initialize(project="energy-thesis")
    for name, (la, lo) in {"Dubai": (25.20, 55.27), "Berlin": (52.52, 13.40),
                           "Mumbai": (19.08, 72.88)}.items():
        print(name, extract_demand(la, lo))
