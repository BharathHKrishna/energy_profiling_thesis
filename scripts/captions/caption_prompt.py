"""
Shared caption prompt builder — used by kit_caption.py (the only caption
backend the live pipeline calls). Split out 2026-08-27 from what used to be
groq_caption.py, which no longer exists: this project used Groq's API early
on, hit its account-level daily token cap (200,000/day, would have taken
~38 days to caption all 10,000 coordinates), and migrated fully to KIT's
KI-Toolbox. This file is the prompt template only — no API client, no key,
no network call — so it carries no dependency on either backend.
"""


def _fmt(val, decimals=1, unit="", missing="unknown"):
    if val is None:
        return missing
    try:
        return f"{float(val):.{decimals}f}{unit}"
    except (TypeError, ValueError):
        return str(val)


def build_prompt(features: dict, bbox_size_m: int = 512) -> str:
    lat = features.get("lat", 0)
    lon = features.get("lon", 0)

    dominant     = features.get("wc_dominant_class", "unknown")
    dominant_pct = _fmt(features.get("wc_dominant_pct"), unit="%")
    built_pct    = _fmt(features.get("wc_built_up_pct"), unit="%")
    tree_pct     = _fmt(features.get("wc_tree_cover_pct"), unit="%")
    water_pct    = _fmt(features.get("wc_water_pct"), unit="%")
    crop_pct     = _fmt(features.get("wc_cropland_pct"), unit="%")
    bare_pct     = _fmt(features.get("wc_bare_sparse_pct"), unit="%")

    pop    = _fmt(features.get("ghsl_population_per_km2"), decimals=0, unit=" /km²")
    bld_ht = _fmt(features.get("ghsl_building_height_m"), unit=" m")

    pvout = _fmt(features.get("solar_pvout_kwh_kwp_day"), unit=" kWh/kWp/day")
    ghi   = _fmt(features.get("solar_ghi_kwh_m2_day"),   unit=" kWh/m²/day")

    ntl       = _fmt(features.get("viirs_ntl_nw_cm2_sr"), decimals=2, unit=" nW/cm²/sr")
    hdd       = _fmt(features.get("climate_hdd"),         decimals=0, unit=" °C·days")
    cdd24     = _fmt(features.get("climate_cdd24"),       decimals=0, unit=" °C·days")
    mean_temp = _fmt(features.get("climate_mean_temp_c"),             unit=" °C")

    heating_mwh   = _fmt(features.get("heating_MWh"), unit=" MWh/yr")
    cooling_mwh   = _fmt(features.get("cooling_MWh"), unit=" MWh/yr")
    demand_regime = features.get("demand_regime") or "unknown"

    infra_parts = []
    if features.get("osm_power_plant"):
        src = features.get("osm_plant_source", "")
        infra_parts.append(f"power plant ({src})" if src else "power plant")
    if features.get("osm_power_substation"):
        infra_parts.append("substation")
    if features.get("osm_power_line"):
        infra_parts.append("power lines")
    if features.get("osm_power_tower"):
        infra_parts.append("power towers")
    if features.get("osm_generator_source"):
        infra_parts.append(f"generator ({features['osm_generator_source']})")
    if features.get("osm_waterway"):
        infra_parts.append(f"waterway ({features['osm_waterway']})")
    if features.get("osm_landuse"):
        infra_parts.append(f"land-use: {features['osm_landuse']}")
    infra_str = ", ".join(infra_parts) if infra_parts else "none detected"

    bld_count = features.get("osm_building_count", 0)

    return f"""You are a geospatial energy analyst. A user has selected a {bbox_size_m}m \xd7 {bbox_size_m}m area at {lat:.4f}\xb0, {lon:.4f}\xb0. Write a concise 3-5 sentence factual description of this location from an energy perspective.

Land cover (ESA WorldCover): dominant={dominant} ({dominant_pct}), built-up={built_pct}, tree cover={tree_pct}, water={water_pct}, cropland={crop_pct}, bare/sparse={bare_pct}
Solar potential: PVOUT={pvout}, GHI={ghi}
Nighttime lights (VIIRS): {ntl}
Climate (annual): heating degree days (base 18°C)={hdd}, cooling degree days (base 24°C)={cdd24}, mean temp={mean_temp}
Estimated building energy demand (geographic regression model, per 512m cell): heating={heating_mwh}, cooling={cooling_mwh} (regime: {demand_regime})
Population density: {pop}
Mean building height: {bld_ht}
OSM buildings in area: {bld_count}
Energy infrastructure (OSM): {infra_str}

Describe what the land cover, energy potential figures, and infrastructure reveal about this location’s energy profile. Be specific and factual. Write only the description — no headings, no bullet points. Do not use em dashes (—); use commas, periods, or "and" instead."""
