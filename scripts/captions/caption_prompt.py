"""
Shared caption prompt builder — used by kit_caption.py (the only caption
backend the live pipeline calls). Split out 2026-08-27 from what used to be
groq_caption.py, which no longer exists: this project used Groq's API early
on, hit its account-level daily token cap (200,000/day, would have taken
~38 days to caption all 10,000 coordinates), and migrated fully to KIT's
KI-Toolbox. This file is the prompt template only — no API client, no key,
no network call — so it carries no dependency on either backend.

Rewritten 2026-09-09, two changes:

1. Categorical, not numerical. The prompt used to hand the model the CSV row
   verbatim, which made every caption a prose restatement of numbers the
   structured record already holds exactly. It now passes banded descriptions
   from feature_bands.py, so the caption carries the interpretation and the
   record keeps the measurements. Band edges come from each source's own
   published semantics wherever the source defines them; see
   feature_bands.BAND_PROVENANCE for which do and which do not.

2. Global Solar Atlas dropped. PVOUT and GHI sit on a roughly 1 km grid, about
   one pixel per four 512 m tiles, so a solar figure quoted for a tile is a
   regional value wearing a local label. Every other feature in the record
   resolves at or below tile scale and describes the built environment itself.
"""

from scripts.captions.feature_bands import describe


def _infrastructure(features: dict) -> str:
    """OSM energy infrastructure, named but never counted."""
    parts = []
    if features.get("osm_power_plant"):
        src = features.get("osm_plant_source", "")
        parts.append(f"a power plant ({src})" if src else "a power plant")
    if features.get("osm_power_substation"):
        parts.append("an electrical substation")
    if features.get("osm_power_tower"):
        parts.append("transmission towers")
    if features.get("osm_generator_source"):
        parts.append(f"a {features['osm_generator_source']} generator")
    if features.get("osm_waterway"):
        parts.append(f"a waterway ({features['osm_waterway']})")
    if features.get("osm_landuse"):
        parts.append(f"land mapped as {features['osm_landuse']}")
    if features.get("osm_amenity"):
        parts.append(f"an amenity mapped as {features['osm_amenity']}")
    return ", ".join(parts) if parts else "none mapped"


def _buildings(features: dict) -> str:
    """Footprint presence as a category. Absence is reported as unmapped rather
    than as empty: 64.8 % of this dataset's tiles returned no OSM elements at
    all because of the tile-retrieval defect documented in Section 6.6, so an
    empty tile is not evidence that no buildings stand there."""
    try:
        n = int(float(features.get("osm_building_count") or 0))
    except (TypeError, ValueError):
        n = 0
    if n == 0:
        return "no building footprints are mapped here in OpenStreetMap"
    if n <= 5:
        return "a few individual building footprints are mapped"
    if n <= 25:
        return "a small cluster of building footprints is mapped"
    if n <= 100:
        return "many building footprints are mapped"
    return "a dense mass of building footprints is mapped"


def build_prompt(features: dict, bbox_size_m: int = 512) -> str:
    lat = features.get("lat", 0)
    lon = features.get("lon", 0)
    b = describe(features)

    def g(key, fallback="not available"):
        return b.get(key, fallback)

    regime = (g("demand_regime", "not determined") or "not determined")
    # the stored regime string carries an em dash; this project's prose does not
    regime = regime.replace("—", ",").replace(" ,", ",")

    lines = [
        f"Land cover is dominated by {g('dominant_cover', 'no single class')}, "
        f"which covers {g('dominant_share', 'an unclear share of')} the area.",
        f"Built-up surface covers {g('built_up')} the tile.",
        f"Tree cover covers {g('tree_cover')} the tile.",
        f"Cropland covers {g('cropland')} the tile.",
        f"Water covers {g('water')} the tile.",
        f"Bare or sparse ground covers {g('bare_sparse')} the tile.",
        f"Population density (GHS-SMOD bands): {g('population')}.",
        f"Typical building height: {g('building_height')}.",
        f"Nighttime lighting (VIIRS): {g('night_lights')}.",
        f"Vegetation greenness (NDVI): {g('vegetation')}.",
        f"Surface brightness (MODIS): {g('reflectance')}.",
        f"Heating climate: {g('heating_season')}.",
        f"Cooling climate: {g('cooling_season')}.",
        f"Dominant conditioning need: {regime}.",
        f"Building footprints: {_buildings(features)}.",
        f"Energy infrastructure mapped in OpenStreetMap: {_infrastructure(features)}.",
    ]
    body = "\n".join(lines)

    return f"""You are a geospatial energy analyst. A user has selected a {bbox_size_m}m \xd7 {bbox_size_m}m area at {lat:.4f}\xb0, {lon:.4f}\xb0. Write a concise 3-5 sentence factual description of this location from an energy perspective.

Every observation below has already been graded into a category against its own data source's published thresholds. Work only from these categories.

{body}

Describe what this location's land cover, built form, climate and infrastructure together imply about its energy profile: what drives demand here, what the built environment looks like, and what the lighting and population signals suggest about activity.

Rules:
- Write only the description. No headings, no bullet points, no preamble.
- Use the qualitative wording given. Do not invent numbers, percentages, densities, degree-day totals, energy figures, or units, and do not convert the categories back into figures.
- Do not name a specific city, district, country, or landmark. Describe the place, do not identify it.
- Where a value is "not available", either say it is unavailable or leave it out. Never guess it.
- If OpenStreetMap footprints or infrastructure are unmapped, that means the area lacks map coverage, not that nothing stands there. Say it is unmapped, and draw no conclusion about the energy system from it.
- Do not use em dashes (—); use commas, periods, or "and" instead."""
