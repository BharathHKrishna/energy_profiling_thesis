"""
Wind feature extractor — Open-Meteo ERA5 reanalysis (no download required).

Uses ERA5 hourly wind speed at 100m hub height for a full calendar year
to derive long-term mean wind speed and wind power density.

Source: Open-Meteo Historical Weather API (open-data, no key required)
Reference year: 2023 (full year, ERA5 reanalysis)

Features returned:
    wind_speed_100m_ms           — annual mean wind speed at 100m (m/s)
    wind_power_density_100m_wm2  — annual mean wind power density at 100m (W/m²)
                                   computed as 0.5 × ρ × mean(v³), ρ = 1.225 kg/m³

Interpretation:
    wind_speed_100m_ms < 6      — low potential (below IEC Class III)
    wind_speed_100m_ms 6–7.5    — moderate (IEC Class II)
    wind_speed_100m_ms > 7.5    — high potential (IEC Class I)
    wind_power_density > 400    — commercially viable threshold
"""
import sys
import time
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import requests

from scripts.utils.logger import get_logger

logger = get_logger("wind_atlas_extractor")

_ERA5_URL  = "https://archive-api.open-meteo.com/v1/era5"
_REF_YEAR  = "2023"   # full calendar year used for climatological mean
_AIR_DENSITY = 1.225  # kg/m³ — standard sea-level air density


def extract_wind_features(lat: float, lon: float,
                          min_lat=None, max_lat=None,
                          min_lon=None, max_lon=None) -> dict:
    """
    Fetch ERA5 hourly wind speed at 100m for one full year and return
    annual mean wind speed and wind power density.

    bbox arguments (min/max lat/lon) are accepted for API compatibility
    but ignored — ERA5 is a point query at (lat, lon).

    Returns empty dict on any failure — null contract preserved.
    """
    logger.info(f"Fetching ERA5 wind data for ({lat}, {lon})")
    t0 = time.time()

    try:
        resp = requests.get(
            _ERA5_URL,
            params={
                "latitude":        lat,
                "longitude":       lon,
                "start_date":      f"{_REF_YEAR}-01-01",
                "end_date":        f"{_REF_YEAR}-12-31",
                "hourly":          "wind_speed_100m",
                "wind_speed_unit": "ms",
                "timezone":        "GMT",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

    except Exception as e:
        logger.warning(f"Wind ERA5 request failed for ({lat}, {lon}): {e}")
        return {}

    speeds = [v for v in data.get("hourly", {}).get("wind_speed_100m", [])
              if v is not None and 0 <= v <= 60]

    if not speeds:
        logger.warning(f"Wind ERA5: no valid speed values for ({lat}, {lon})")
        return {}

    mean_speed = round(sum(speeds) / len(speeds), 2)
    mean_v3    = sum(v ** 3 for v in speeds) / len(speeds)
    wind_power = round(0.5 * _AIR_DENSITY * mean_v3, 1)

    elapsed = round(time.time() - t0, 2)
    logger.info(
        f"Wind ERA5 ({lat}, {lon}): {len(speeds)} hourly values, "
        f"mean={mean_speed} m/s, WPD={wind_power} W/m² ({elapsed}s)"
    )

    return {
        "wind_speed_100m_ms":          mean_speed,
        "wind_power_density_100m_wm2": wind_power,
    }


if __name__ == "__main__":
    TEST_COORDS = [
        ("north_sea",  57.0,   3.0),    # expect > 9 m/s
        ("sahara",     26.0,   3.0),    # expect < 4 m/s
        ("patagonia", -50.0, -68.0),    # expect > 10 m/s
        ("denmark",   55.7,   12.5),    # expect ~7 m/s
    ]
    print("\n=== Wind Extractor Test (Open-Meteo ERA5) ===\n")
    for name, lat, lon in TEST_COORDS:
        r = extract_wind_features(lat, lon)
        print(f"[{name}] ({lat}, {lon})")
        for k, v in r.items():
            print(f"  {k}: {v}")
        print()
