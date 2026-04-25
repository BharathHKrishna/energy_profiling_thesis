import sys
import os
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import requests
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from collections import Counter

from scripts.utils.logger import get_logger
from scripts.utils.config_loader import load_config

logger = get_logger("osm_query")
config = load_config("/srv/THESIS/energy_profiling_thesis/configs/config.yaml")

# ── Constants ─────────────────────────────────────────────────────────────────
OVERPASS_URL  = config.get("api", {}).get("overpass_url", "https://overpass-api.de/api/interpreter")
REQUEST_DELAY = config.get("api", {}).get("overpass_request_delay", 1.5)
MAX_RETRIES   = config.get("api", {}).get("overpass_max_retries", 3)
TIMEOUT       = config.get("api", {}).get("overpass_timeout", 30)

# Mirror list — kumi.systems first because overpass-api.de returns 406 on this IP.
# overpass-api.de kept as last-resort in case IP situation changes.
OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    OVERPASS_URL,
]

_OVERPASS_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept":       "application/json, */*",
}

# Circuit breaker: after all mirrors fail, skip OSM for this many seconds.
# Prevents 8+ minutes of wasted retries when the network blocks Overpass.
_OVERPASS_CIRCUIT_TIMEOUT_S = 300   # 5 minutes
_overpass_unreachable_until  = 0.0  # epoch timestamp; 0 = circuit closed

# Per-process response cache keyed by bbox tuple.
# extract_osm_features and get_osm_energy_elements share the same query —
# caching avoids a second HTTP round-trip per coordinate.
_overpass_cache: dict = {}


# ── Bounding Box Helper ───────────────────────────────────────────────────────

def generate_bbox(lat, lon, size_m=256):
    """
    Generate a 256×256m bounding box centred on (lat, lon).
    Same logic as stratified_sampler.py — must stay consistent.
    """
    delta_lat = (size_m / 2) / 111320
    delta_lon = (size_m / 2) / (111320 * np.cos(np.radians(lat)))
    return {
        "min_lat": round(lat - delta_lat, 6),
        "max_lat": round(lat + delta_lat, 6),
        "min_lon": round(lon - delta_lon, 6),
        "max_lon": round(lon + delta_lon, 6),
    }


# ── Query Builder ─────────────────────────────────────────────────────────────

def build_overpass_query(min_lat, max_lat, min_lon, max_lon):
    """
    Build Overpass QL query to fetch all energy-relevant OSM features
    within a 256×256m bounding box.

    Overpass bbox format: south, west, north, east
                        = min_lat, min_lon, max_lat, max_lon

    Features covered:
      Buildings     — type, height, material, count, start_date,
                      roof shape/material/colour
      Power         — plant, substation, line, tower, pole,
                      generator:source, plant:source
      Land          — landuse, surface, highway, railway
      Industrial    — pipeline, petroleum_well, storage_tank, dam
      Amenities     — hospital, school, charging_station, supermarket
    """
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    query = f"""
[out:json][timeout:{TIMEOUT}];
(
  way["building"]({bbox});
  node["power"]({bbox});
  way["power"]({bbox});
  relation["power"]({bbox});
  node["generator:source"]({bbox});
  way["generator:source"]({bbox});
  node["plant:source"]({bbox});
  way["plant:source"]({bbox});
  way["landuse"]({bbox});
  node["man_made"="pipeline"]({bbox});
  way["man_made"="pipeline"]({bbox});
  node["man_made"="petroleum_well"]({bbox});
  node["man_made"="storage_tank"]({bbox});
  way["man_made"="storage_tank"]({bbox});
  node["waterway"="dam"]({bbox});
  way["waterway"="dam"]({bbox});
  way["highway"]({bbox});
  way["railway"]({bbox});
  node["amenity"~"hospital|school|charging_station"]({bbox});
  way["amenity"~"hospital|school|charging_station"]({bbox});
  node["shop"="supermarket"]({bbox});
  way["shop"="supermarket"]({bbox});
);
out body;
>;
out skel qt;
"""
    return query.strip()


# ── Request Handler ───────────────────────────────────────────────────────────

def send_overpass_request(query, overpass_url=None):
    """
    POST query to Overpass API using parallel mirror racing + circuit breaker.

    All mirrors are tried concurrently; the first successful 200 response wins
    and cancels the others. This eliminates the sequential 8s-per-mirror wait
    that occurred when some mirrors were temporarily down.

    Response cache: identical queries within the same process return cached
    elements — extract_osm_features and get_osm_energy_elements share one query
    per coordinate, so the second call costs nothing.

    Circuit breaker: if all mirrors fail, subsequent calls return [] immediately
    for _OVERPASS_CIRCUIT_TIMEOUT_S seconds.
    """
    import time as _t
    import concurrent.futures
    global _overpass_unreachable_until, _overpass_cache

    if query in _overpass_cache:
        return _overpass_cache[query]

    if _t.time() < _overpass_unreachable_until:
        logger.info("Overpass circuit breaker active — returning empty (will retry after cooldown)")
        return []

    mirrors = [overpass_url] if overpass_url else list(OVERPASS_MIRRORS)
    _MIRROR_TIMEOUT = 12  # per-mirror timeout in seconds

    def _try_mirror(url):
        resp = requests.post(
            url,
            data={"data": query},
            headers=_OVERPASS_HEADERS,
            timeout=_MIRROR_TIMEOUT,
        )
        if resp.status_code == 200:
            return url, resp.json().get("elements", [])
        return url, None  # non-200 → treated as failure

    elements = None
    winning_url = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(mirrors)) as pool:
        futures = {pool.submit(_try_mirror, url): url for url in mirrors}
        try:
            for future in concurrent.futures.as_completed(futures, timeout=_MIRROR_TIMEOUT + 2):
                try:
                    url, result = future.result()
                    if result is not None:
                        elements    = result
                        winning_url = url
                        break  # cancel remaining — first 200 wins
                except Exception as exc:
                    logger.warning(f"Mirror {futures[future]} error: {exc}")
        except concurrent.futures.TimeoutError:
            logger.warning("All mirrors timed out in parallel race")

    if elements is not None:
        logger.info(f"Overpass OK [{winning_url}] — {len(elements)} elements")
        _overpass_unreachable_until = 0.0
        _overpass_cache[query] = elements
        return elements

    # All mirrors failed — open circuit breaker
    _overpass_unreachable_until = time.time() + _OVERPASS_CIRCUIT_TIMEOUT_S
    logger.error(f"All Overpass mirrors failed — circuit breaker opened for {_OVERPASS_CIRCUIT_TIMEOUT_S}s")
    return []


# ── Response Parser ───────────────────────────────────────────────────────────

def parse_osm_response(elements):
    """
    Parse raw Overpass API elements into a flat feature dict.

    Design rules:
      - Absent tag      → None   (stripped from output by extract_osm_features)
      - Boolean feature → True if present, None if absent  (never False)
        Reason: None means "no data", False would mean "confirmed absent"
                which we cannot know from OSM alone.
      - Count feature   → integer (0 is valid for building_count)
      - Multi-value     → comma-separated unique values
      - Numeric         → float rounded to 2dp, or None
    """

    # ── Accumulators ──────────────────────────────────────────────────────────
    buildings         = []
    power_plants      = []
    power_substations = []
    power_lines       = []
    power_towers      = []
    power_poles       = []
    generator_sources = []
    plant_sources     = []
    landuses          = []
    pipelines         = []
    petroleum_wells   = []
    storage_tanks     = []
    dams              = []
    highways          = []
    railways          = []
    amenities         = []
    supermarkets      = []
    surfaces          = []

    for el in elements:
        tags = el.get("tags", {})
        if not tags:
            continue

        # ── Buildings ─────────────────────────────────────────────────────────
        if "building" in tags:
            buildings.append(tags)

        # ── Power ─────────────────────────────────────────────────────────────
        power_val = tags.get("power")
        if power_val == "plant":
            power_plants.append(tags)
        elif power_val == "substation":
            power_substations.append(tags)
        elif power_val == "line":
            power_lines.append(tags)
        elif power_val == "tower":
            power_towers.append(tags)
        elif power_val == "pole":
            power_poles.append(tags)

        # ── Generator and plant sources ───────────────────────────────────────
        if "generator:source" in tags:
            generator_sources.append(tags["generator:source"])
        if "plant:source" in tags:
            plant_sources.append(tags["plant:source"])

        # ── Land use ──────────────────────────────────────────────────────────
        if "landuse" in tags:
            landuses.append(tags["landuse"])

        # ── Man-made infrastructure ───────────────────────────────────────────
        man_made = tags.get("man_made")
        if man_made == "pipeline":
            pipelines.append(tags)
        elif man_made == "petroleum_well":
            petroleum_wells.append(tags)
        elif man_made == "storage_tank":
            storage_tanks.append(tags)

        # ── Dam ───────────────────────────────────────────────────────────────
        if tags.get("waterway") == "dam":
            dams.append(tags)

        # ── Transport ─────────────────────────────────────────────────────────
        if "highway" in tags:
            highways.append(tags["highway"])
        if "railway" in tags:
            railways.append(tags["railway"])

        # ── Amenities ─────────────────────────────────────────────────────────
        amenity_val = tags.get("amenity")
        if amenity_val in ("hospital", "school", "charging_station"):
            amenities.append(amenity_val)

        # ── Supermarket ───────────────────────────────────────────────────────
        if tags.get("shop") == "supermarket":
            supermarkets.append(tags)

        # ── Surface ───────────────────────────────────────────────────────────
        if "surface" in tags:
            surfaces.append(tags["surface"])

    # ── Helper functions ──────────────────────────────────────────────────────

    def most_common(lst):
        """Most frequent value, or None if list is empty."""
        if not lst:
            return None
        return Counter(lst).most_common(1)[0][0]

    def unique_joined(lst):
        """Unique values as comma-separated string, or None if empty."""
        if not lst:
            return None
        seen = []
        for v in lst:
            if v not in seen:
                seen.append(v)
        return ",".join(seen)

    def presence(lst):
        """True if list is non-empty, None otherwise."""
        return True if lst else None

    # ── Building sub-features ─────────────────────────────────────────────────

    building_types = [
        b.get("building") for b in buildings
        if b.get("building") not in (None, "yes")
    ]

    building_heights = []
    for b in buildings:
        raw_h      = b.get("height")
        raw_levels = b.get("building:levels")
        if raw_h:
            try:
                val = float(str(raw_h).replace("m", "").replace(" ", "").strip())
                building_heights.append(val)
            except ValueError:
                pass
        elif raw_levels:
            try:
                val = float(str(raw_levels).strip()) * 3.0
                building_heights.append(val)
            except ValueError:
                pass

    building_materials   = [b.get("building:material") for b in buildings if b.get("building:material")]
    building_start_dates = [b.get("start_date")        for b in buildings if b.get("start_date")]
    roof_shapes          = [b.get("roof:shape")         for b in buildings if b.get("roof:shape")]
    roof_materials       = [b.get("roof:material")      for b in buildings if b.get("roof:material")]
    roof_colours         = [b.get("roof:colour")        for b in buildings if b.get("roof:colour")]

    # ── Final feature dict ────────────────────────────────────────────────────
    features = {

        # ── Building ──────────────────────────────────────────────────────────
        "osm_building_count":       len(buildings),
        "osm_building_type":        most_common(building_types),
        "osm_building_height_mean": round(sum(building_heights) / len(building_heights), 2)
                                    if building_heights else None,
        "osm_building_material":    most_common(building_materials),
        "osm_building_start_date":  min(building_start_dates) if building_start_dates else None,
        "osm_roof_shape":           most_common(roof_shapes),
        "osm_roof_material":        most_common(roof_materials),
        "osm_roof_colour":          most_common(roof_colours),

        # ── Power infrastructure ───────────────────────────────────────────────
        "osm_power_plant":          presence(power_plants),
        "osm_power_substation":     presence(power_substations),
        "osm_power_line":           presence(power_lines),
        "osm_power_tower":          presence(power_towers),
        "osm_power_pole":           presence(power_poles),
        "osm_generator_source":     unique_joined(generator_sources),
        "osm_plant_source":         unique_joined(plant_sources),

        # ── Land and transport ────────────────────────────────────────────────
        "osm_landuse":              most_common(landuses),
        "osm_highway":              most_common(highways),
        "osm_surface":              most_common(surfaces),
        "osm_railway":              presence(railways),

        # ── Industrial / energy specific ──────────────────────────────────────
        "osm_pipeline":             presence(pipelines),
        "osm_petroleum_well":       presence(petroleum_wells),
        "osm_storage_tank":         presence(storage_tanks),
        "osm_dam":                  presence(dams),

        # ── Amenities ────────────────────────────────────────────────────────
        "osm_amenity":              unique_joined(amenities),
        "osm_supermarket":          presence(supermarkets),
    }

    return features


# ── Single Coordinate Extractor ───────────────────────────────────────────────

def extract_osm_features(lat, lon, min_lat, max_lat, min_lon, max_lon):
    """
    Full extraction pipeline for one coordinate:
      1. Build Overpass query for the 256×256m bounding box
      2. Send POST request to Overpass API (with retry logic)
      3. Parse JSON response into flat feature dict
      4. Strip all None values — absent key = no data (consistent with all extractors)
         Exception: osm_building_count kept even if 0 — 0 is valid data
      5. Return dict with coordinate metadata prepended

    Null contract:
      - Feature present in OSM   → key: value in dict
      - Feature absent from OSM  → key not in dict at all
      - CSV output               → cell empty
      - JSON output              → key absent from coordinate object
    """
    logger.info(f"Extracting OSM features for ({lat}, {lon})")

    query    = build_overpass_query(min_lat, max_lat, min_lon, max_lon)
    elements = send_overpass_request(query)

    if not elements:
        logger.warning(
            f"Zero elements returned for ({lat}, {lon}) — "
            "all features will be absent (sparse OSM region)"
        )

    features = parse_osm_response(elements)

    # ── Strip None values ──────────────────────────────────────────────────────
    # Absent key = no data — consistent with GHSL, Solar Atlas, WorldCover,
    # ERA5, and VIIRS extractors.
    # osm_building_count is kept even when 0 — 0 buildings is valid information.
    features = {
        k: v for k, v in features.items()
        if v is not None or k == "osm_building_count"
    }

    # ── Prepend coordinate metadata ───────────────────────────────────────────
    result = {
        "lat":               lat,
        "lon":               lon,
        "raw_element_count": len(elements),
    }
    result.update(features)

    populated = len(features) - 1   # subtract osm_building_count from meaningful count
    logger.info(
        f"({lat}, {lon}) — {len(elements)} elements → "
        f"{len(features)} features populated "
        f"(building_count={features.get('osm_building_count', 0)})"
    )

    return result


# ── M3 Support: energy element positions for Gaussian density signal ─────────

# Energy relevance weight per OSM tag — used to modulate Gaussian kernel
_ELEMENT_WEIGHTS = {
    # Power infrastructure
    ("power", "plant"):       3.0,
    ("power", "substation"):  2.5,
    ("power", "generator"):   2.5,
    ("power", "line"):        1.5,
    ("power", "tower"):       1.0,
    ("power", "pole"):        0.8,
    # Man-made energy infrastructure
    ("man_made", "petroleum_well"): 3.0,
    ("man_made", "storage_tank"):   2.0,
    ("man_made", "pipeline"):       1.5,
    # Water / energy
    ("waterway", "dam"):      2.5,
    # Land use
    ("landuse", "industrial"):   2.0,
    ("landuse", "commercial"):   1.5,
    ("landuse", "farmland"):     0.8,
    ("landuse", "residential"):  0.5,
}

_BUILDING_WEIGHTS = {
    "industrial": 2.0, "warehouse": 1.8, "factory": 2.0, "refinery": 2.5,
    "commercial": 1.0, "office": 1.0, "retail": 0.8,
    "yes": 0.5, "residential": 0.4, "apartments": 0.4,
}


def _get_element_energy_weight(tags):
    """Return energy relevance weight for an OSM element."""
    for (key, val), weight in _ELEMENT_WEIGHTS.items():
        if tags.get(key) == val:
            return weight
    if tags.get("generator:source"):
        return 2.5
    if tags.get("plant:source"):
        return 2.5
    building = tags.get("building")
    if building:
        return _BUILDING_WEIGHTS.get(building, 0.3)
    return 0.0


def get_osm_energy_elements(lat, lon, min_lat, max_lat, min_lon, max_lon):
    """
    Return a list of energy-relevant OSM element positions for M3 Gaussian density.

    Each entry: {"lat": float, "lon": float, "weight": float}

    Uses the same Overpass query as extract_osm_features() so no extra API call
    is needed — just re-run the query (caller caches if needed).
    Null contract: returns empty list if no elements or API unavailable.
    """
    query    = build_overpass_query(min_lat, max_lat, min_lon, max_lon)
    elements = send_overpass_request(query)

    if not elements:
        return []

    # Build node-id → (lat, lon) lookup from skeleton nodes (member nodes of ways)
    node_positions = {}
    for el in elements:
        if el.get("type") == "node" and "lat" in el and "lon" in el:
            node_positions[el["id"]] = (el["lat"], el["lon"])

    result = []
    for el in elements:
        tags = el.get("tags", {})
        if not tags:
            continue
        weight = _get_element_energy_weight(tags)
        if weight <= 0:
            continue

        if el.get("type") == "node" and "lat" in el:
            result.append({"lat": el["lat"], "lon": el["lon"], "weight": weight})

        elif el.get("type") == "way" and "nodes" in el:
            member_lats, member_lons = [], []
            for nid in el["nodes"]:
                if nid in node_positions:
                    member_lats.append(node_positions[nid][0])
                    member_lons.append(node_positions[nid][1])
            if member_lats:
                result.append({
                    "lat":    float(np.mean(member_lats)),
                    "lon":    float(np.mean(member_lons)),
                    "weight": weight,
                })

    logger.info(
        f"get_osm_energy_elements ({lat}, {lon}): "
        f"{len(result)} energy elements from {len(elements)} raw"
    )
    return result


# ── Main — quick test on 5 representative coordinates ────────────────────────

if __name__ == "__main__":

    TEST_COORDS = [
        ("dense_urban",  52.5200,  13.4050),
        ("industrial",   51.4880,   7.2200),
        ("agricultural", 41.5000, -93.0000),
        ("arid",         26.0000,   3.0000),
        ("forest",       -4.0000, -60.0000),
    ]

    results = []
    for stratum, lat, lon in TEST_COORDS:
        logger.info(f"── Testing {stratum} ({lat}, {lon}) ──")
        bbox   = generate_bbox(lat, lon)
        result = extract_osm_features(
            lat, lon,
            bbox["min_lat"], bbox["max_lat"],
            bbox["min_lon"], bbox["max_lon"]
        )
        result["stratum"] = stratum
        results.append(result)
        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(results)

    # Columns that start with osm_
    feature_cols = [c for c in df.columns if c.startswith("osm_")]

    print("\n=== OSM Extraction Test — Results Summary ===")
    print(f"Coordinates tested : {len(df)}")
    print(f"Features per coord : {len(feature_cols)}")

    print("\nFeature coverage per coordinate:")
    for _, row in df.iterrows():
        # pd.notna handles both None and NaN correctly
        populated = sum(
            1 for c in feature_cols
            if pd.notna(row.get(c)) and not (c == "osm_building_count" and row.get(c) == 0)
        )
        print(
            f"  {row['stratum']:25s} ({row['lat']:8.4f}, {row['lon']:8.4f}) — "
            f"{populated:2d}/{len(feature_cols)} features populated | "
            f"{row['raw_element_count']} raw elements"
        )

    print("\nBuilding counts:")
    for _, row in df.iterrows():
        print(f"  {row['stratum']:25s}: {int(row.get('osm_building_count', 0))} buildings")

    print("\nPower infrastructure found:")
    power_fields = [
        "osm_power_plant", "osm_power_substation", "osm_power_line",
        "osm_power_tower", "osm_power_pole", "osm_generator_source", "osm_plant_source"
    ]
    for _, row in df.iterrows():
        found = [
            f.replace("osm_", "") for f in power_fields
            if pd.notna(row.get(f))
        ]
        print(f"  {row['stratum']:25s}: {found if found else 'none'}")

    print("\nKey features table:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    display_cols = [
        c for c in [
            "stratum", "osm_building_count", "osm_building_type",
            "osm_landuse", "osm_power_plant", "osm_power_line",
            "osm_highway", "raw_element_count"
        ] if c in df.columns
    ]
    print(df[display_cols].to_string(index=False))