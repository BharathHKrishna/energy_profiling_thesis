"""
Climate extractor — Heating/Cooling Degree Days from ERA5-Land via GEE.

Computes annual HDD and CDD (base 18°C) per 512m bbox, plus mean annual
temperature. These quantify climate-driven energy intensity:
  HDD high → cold climate → heating demand
  CDD high → hot climate  → cooling demand

Source: ECMWF/ERA5_LAND/DAILY_AGGR, band temperature_2m (Kelvin).
Native resolution ~11km (0.1°) — far coarser than the 512m bbox, so a
coord samples its single overlapping climate pixel (climate doesn't vary
at 512m scale; this is physically correct).

Null contract: absent key = no data. ERA5-Land has no ocean coverage, so
coords over open water return None for all climate features — acceptable,
as there is no building heating/cooling demand on water.
"""
import time
import sys
import os

sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import warnings
warnings.filterwarnings("ignore")

from scripts.utils.logger import get_logger
from scripts.utils.config_loader import load_config

logger = get_logger("climate_extractor")
config = load_config("/srv/THESIS/energy_profiling_thesis/configs/config.yaml")

# ── GEE initialisation ────────────────────────────────────────────────────────
try:
    import ee
    ee.Initialize(project=config.get("api", {}).get("gee_project", "energy-thesis"))
    GEE_AVAILABLE = True
    logger.info("GEE initialised successfully")
except Exception as e:
    GEE_AVAILABLE = False
    logger.warning(f"GEE initialisation failed: {e}")

# ── Config ──────────────────────────────────────────────────────────────────
ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
TEMP_BAND       = "temperature_2m"   # Kelvin
BASE_TEMP_C     = 18.0               # degree-day base (standard for buildings)
COOL_BASE_C     = 24.0               # separate cooling base — matches demand_extractor's
                                     # Eurostat-fit cooling formula (CDD24), distinct from
                                     # the base-18 climate_cdd already used elsewhere
DATE_START      = "2023-01-01"
DATE_END        = "2023-12-31"
ERA5_SCALE_M    = 500                # sample at 500m so the 512m bbox always
                                     # catches the overlapping ERA5 pixel; GEE
                                     # resamples the coarse ~11km pixel down —
                                     # the value is the same, the hit is reliable.
                                     # (native scale 11132m can fall between pixel
                                     # centers over a 512m bbox and return None)

# Physical valid ranges (annual totals / mean)
CLIMATE_VALID = {
    "hdd":           (0.0, 12000.0),
    "cdd":           (0.0, 8000.0),
    "cdd24":         (0.0, 8000.0),
    "mean_temp_c":   (-40.0, 45.0),
}


def _is_valid(value, feature_name):
    if value is None:
        return False
    rng = CLIMATE_VALID.get(feature_name)
    if rng is None:
        return True
    return rng[0] <= value <= rng[1]


def _reduce_region(image, min_lat, max_lat, min_lon, max_lon, band_name):
    """Mean-reduce a single-band GEE image over the bbox. Returns float or None.

    Retries transient GEE errors (rate limit, timeout, connection reset) with
    backoff -- found live that occasional GEE calls take 30-95s or fail
    outright under sustained pipeline load with zero underlying data problem,
    a real, retriable latency/availability issue distinct from a genuine
    missing-data case (which still correctly returns None after retries)."""
    if not GEE_AVAILABLE:
        return None
    last_err = None
    for attempt in range(4):
        try:
            region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
            result = (
                image.select(band_name)
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=ERA5_SCALE_M,
                    maxPixels=1e6,
                )
                .getInfo()
            )
            value = result.get(band_name)
            return None if value is None else float(value)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            transient = any(s in msg for s in ("429", "too many", "timeout", "timed out",
                                               "503", "502", "rate", "connection"))
            if attempt < 3 and transient:
                time.sleep(2 * (2 ** attempt))  # 2s, 4s, 8s
                continue
            break
    logger.warning(f"ERA5 bbox reduce failed for {band_name} after retries: {last_err}")
    return None


def extract_climate_features(lat, lon, min_lat, max_lat, min_lon, max_lon):
    """
    Extract annual HDD, CDD, and mean temperature for a bbox.

    Degree days are computed server-side: for each daily mean temperature T(°C),
        HDD_day = max(0, BASE - T)
        CDD_day = max(0, T - BASE)
    then summed across all days in the year.

    Returns:
        dict with any of: climate_hdd, climate_cdd, climate_mean_temp_c.
        Empty dict if GEE unavailable or no land coverage (null contract).
    """
    if not GEE_AVAILABLE:
        logger.info("Climate: GEE unavailable — returning empty")
        return {}

    try:
        daily = (
            ee.ImageCollection(ERA5_COLLECTION)
            .filterDate(DATE_START, DATE_END)
            .select(TEMP_BAND)
        )

        # Convert each day's temperature to Celsius once, reused for all three.
        def to_celsius(img):
            return img.subtract(273.15).rename("tc").copyProperties(img, ["system:time_start"])

        daily_c = daily.map(to_celsius)

        # Server-side degree-day images
        def to_hdd(img):
            return ee.Image(BASE_TEMP_C).subtract(img).max(0).rename("hdd")

        def to_cdd(img):
            return img.subtract(BASE_TEMP_C).max(0).rename("cdd")

        def to_cdd24(img):
            return img.subtract(COOL_BASE_C).max(0).rename("cdd24")

        hdd_img   = daily_c.map(to_hdd).sum()
        cdd_img   = daily_c.map(to_cdd).sum()
        cdd24_img = daily_c.map(to_cdd24).sum()
        mean_img  = daily_c.mean()

        result = {}

        hdd = _reduce_region(hdd_img, min_lat, max_lat, min_lon, max_lon, "hdd")
        if _is_valid(hdd, "hdd"):
            result["climate_hdd"] = round(hdd, 1)

        cdd = _reduce_region(cdd_img, min_lat, max_lat, min_lon, max_lon, "cdd")
        if _is_valid(cdd, "cdd"):
            result["climate_cdd"] = round(cdd, 1)

        # Base-24 CDD — kept separate from the base-18 climate_cdd above. This is the
        # convention scripts/extractors/demand_extractor.py's Eurostat-fit cooling
        # formula (COOL_BASE=24) needs; climate_cdd (base 18) is unrelated and unchanged.
        cdd24 = _reduce_region(cdd24_img, min_lat, max_lat, min_lon, max_lon, "cdd24")
        if _is_valid(cdd24, "cdd24"):
            result["climate_cdd24"] = round(cdd24, 1)

        mean_t = _reduce_region(mean_img, min_lat, max_lat, min_lon, max_lon, "tc")
        if _is_valid(mean_t, "mean_temp_c"):
            result["climate_mean_temp_c"] = round(mean_t, 2)

        if result:
            logger.info(
                f"Climate: HDD={result.get('climate_hdd')}, "
                f"CDD={result.get('climate_cdd')}, CDD24={result.get('climate_cdd24')}, "
                f"T̄={result.get('climate_mean_temp_c')}°C"
            )
        else:
            logger.info("Climate: no land coverage (likely over water) — empty")
        return result

    except Exception as e:
        logger.warning(f"Climate extraction failed: {e}")
        return {}


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, math

    def _bbox(lat, lon, size_m=512):
        half = size_m / 2
        dlat = half / 111320
        dlon = half / (111320 * math.cos(math.radians(abs(lat) or 0.001)))
        return lat - dlat, lat + dlat, lon - dlon, lon + dlon

    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, default=40.758)
    p.add_argument("--lon", type=float, default=-73.985)
    args = p.parse_args()

    mn_lat, mx_lat, mn_lon, mx_lon = _bbox(args.lat, args.lon)
    out = extract_climate_features(args.lat, args.lon, mn_lat, mx_lat, mn_lon, mx_lon)
    print(out)
