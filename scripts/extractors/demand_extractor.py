"""Heating + cooling demand for a PORE 512 m cell.

Geographic regression, fit on real climate and location data at a set of
independent point locations (118 for heating, 89 for cooling), not a single
literature-derived constant. Replaces an earlier Eurostat-panel-fit version
(HEAT = floor_area x 0.3143 x HDD^0.75, COOL = floor_area x 0.000571 x CDD^1.5)
after this thesis's own validation found the geographic-regression version
generalizes better under leave-one-location-out and spatial-block testing.

    h = HDD18 / 1000
    I_heat = max(0, 28.9846 + 26.5197h + 1.6628h^2 - 38.1175 sin(phi)
             + 47.1338 cos(phi)cos(lambda) - 8.2037 cos(phi)sin(lambda))   [kWh/m^2]

    c = CDD24 / 1000
    I_cool = max(0, 72.1865 - 41.6518c + 55.9437c^2 - 89.1488 sin(phi)
             + 12.3438 cos(phi)cos(lambda) - 8.7934 cos(phi)sin(lambda))   [kWh/m^2]

    demand = floor_area x intensity   [kWh = m^2 x kWh/m^2]

phi, lambda are the coordinate's own latitude/longitude in radians -- the
sin/cos terms represent a coordinate continuously across the whole globe
(including across the 180 degree date line) rather than through a small set
of discrete regional categories, and the outer max(0, .) clamp rules out a
physically impossible negative intensity.

Heating validates well: R^2 = 0.795 for intensity, 0.821 for total demand,
under leave-one-location-out, on 118 independent locations. Cooling, on 89
locations, shows a real gap between on-sample fit (R^2 = 0.866) and
leave-one-location-out (0.737), degrading further to 0.638 under a stricter
spatial-block validation -- a documented, genuine limitation, not smoothed
over: cooling demand depends heavily on air-conditioning adoption, a
behavioral/economic factor with no climate or coordinate signal to carry it,
in a way heating (driven more directly by building fabric and heating-system
prevalence) does not.

Known limitations (real, unresolved):
  - No domestic hot water (DHW) term. TABULA's own standardized per-m2
    constant (10 kWh/m2/yr single-unit, 15 multi-unit) is the flagged
    template for closing this gap in future work, not yet implemented.
  - Cooling's weaker out-of-sample generalization (above) means cooling
    figures should be read as a genuine but lower-confidence estimate
    relative to heating, not two numbers of equal reliability.
"""
import sys, math
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

from scripts.extractors.ghsl_extractor import compute_floor_area
from scripts.utils.geo import bbox as _bbox
from scripts.extractors.climate_extractor import (
    extract_climate_features, _is_valid as _climate_is_valid
)

HEAT_BASE = 18   # HDD counts temperature below this (deg C)
COOL_BASE = 24   # CDD counts temperature above this (deg C)

# Fitted coefficients (geographic regression, see module docstring)
_HEAT_COEF = (28.9846, 26.5197, 1.6628, -38.1175, 47.1338, -8.2037)
_COOL_COEF = (72.1865, -41.6518, 55.9437, -89.1488, 12.3438, -8.7934)


def _intensity(scaled_dd, lat, lon, coef):
    c0, c1, c2, c3, c4, c5 = coef
    phi = math.radians(lat)
    lam = math.radians(lon)
    val = (c0 + c1 * scaled_dd + c2 * scaled_dd ** 2
           + c3 * math.sin(phi)
           + c4 * math.cos(phi) * math.cos(lam)
           + c5 * math.cos(phi) * math.sin(lam))
    return max(0.0, val)


def _demand_regime(mean_temp_c):
    """Plain-language regime label from annual mean temperature. Drives which single
    demand number gets reported — exactly one of heating_MWh/cooling_MWh, never both,
    never a fabricated zero for whichever the regime doesn't call for."""
    if mean_temp_c is None:
        return None
    if mean_temp_c < HEAT_BASE:
        return "heating"
    if mean_temp_c > COOL_BASE:
        return "cooling"
    return "comfort — no major heating or cooling needed"


def demand_from_features(floor_area_m2, hdd18, cdd24, lat, lon, mean_temp_c=None):
    """Heating OR cooling demand (MWh/yr) — exactly one, chosen by regime, from
    already-extracted feature values. No GHSL/GEE call of its own; for use in the main
    PORE pipeline (feature_extractor.py), where floor_area (compute_floor_area()),
    climate_hdd, climate_cdd24, and climate_mean_temp_c are already fetched.

    Null contract: needs a real floor_area to mean anything at all — if floor_area_m2
    is None, returns {} rather than a 0 MWh sentinel. Otherwise always computes
    demand_floor_m2/demand_regime, and omits heating_MWh/cooling_MWh rather than
    fabricating a 0 if the specific degree-day figure that regime needs is missing.
    """
    if floor_area_m2 is None:
        return {}

    regime = _demand_regime(mean_temp_c)
    result = dict(demand_floor_m2=round(floor_area_m2), demand_regime=regime)

    if regime == "heating" and hdd18 is not None:
        h = hdd18 / 1000.0
        intensity = _intensity(h, lat, lon, _HEAT_COEF)
        heat_kwh = floor_area_m2 * intensity
        result["heating_MWh"] = round(heat_kwh / 1000.0, 1)
    elif regime == "cooling" and cdd24 is not None:
        c = cdd24 / 1000.0
        intensity = _intensity(c, lat, lon, _COOL_COEF)
        cool_kwh = floor_area_m2 * intensity
        result["cooling_MWh"] = round(cool_kwh / 1000.0, 1)

    return result


def _climate_single(lat, lon):
    """HDD (base 18) + CDD (base 24) + annual mean temp for one 512 m cell.
    Delegates to climate_extractor.extract_climate_features() instead of running its own
    ERA5-Land GEE query, so this file's standalone path shares the same range-sanity
    checks (_is_valid()) the main pipeline's own climate_hdd/climate_cdd24 get."""
    min_lat, max_lat, min_lon, max_lon = _bbox(lat, lon)
    feat = extract_climate_features(lat, lon, min_lat, max_lat, min_lon, max_lon)
    return feat.get("climate_hdd"), feat.get("climate_cdd24"), feat.get("climate_mean_temp_c")


def extract_demand(lat, lon):
    """Heating + cooling demand (MWh/yr) for the 512 m cell at (lat, lon).

    Standalone/CLI path — makes its own floor_area read + climate call. The pipeline
    path (feature_extractor.py) uses demand_from_features() instead, reusing values
    already fetched earlier in the same per-coordinate concurrent extraction pass.
    """
    min_lat, max_lat, min_lon, max_lon = _bbox(lat, lon)
    floor_area = compute_floor_area(min_lat, max_lat, min_lon, max_lon)
    hdd, cdd24, mean_t = _climate_single(lat, lon)
    return demand_from_features(floor_area, hdd, cdd24, lat, lon, mean_t)


if __name__ == "__main__":
    for name, (la, lo) in {"Dubai": (25.20, 55.27), "Berlin": (52.52, 13.40),
                           "Mumbai": (19.08, 72.88)}.items():
        print(name, extract_demand(la, lo))
