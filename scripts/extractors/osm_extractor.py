"""OSM feature extractor — OFFLINE.

Source of truth is the LOCAL OpenStreetMap planet (snapshot 2026-06-13). No Overpass,
no network. OSM for a coord set is pre-carved into tiny 512 m tiles ONCE (see
osm_batch_extract.py), then each coord reads its tile in ~5-12 ms.

  planet-latest.osm.pbf ──osm_batch_extract (one pass)──► tiles/<lat>_<lon>.osm.pbf
  (tiles hold 100% of OSM tags for that 512 m — nothing filtered out)

Public API is UNCHANGED from the old Overpass version — same signatures, same
feature keys, same null contract — so every BORE / PORE / segmap caller keeps working:
  - generate_bbox(lat, lon, size_m=512)
  - extract_osm_features(lat, lon, min_lat, max_lat, min_lon, max_lon) -> dict
  - get_osm_energy_elements(lat, lon, min_lat, max_lat, min_lon, max_lon) -> list

Read paths (auto-selected):
  1. FAST    : pre-extracted tile for this coord -> osmium export -> ~5-12 ms.
  2. FALLBACK: live `osmium extract` from the full planet (complete tags) on demand.
Both yield the SAME unified element list (full tags), so parsing is identical.
"""
import sys
import os
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import json
import subprocess
import tempfile
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from collections import Counter

from scripts.utils.logger import get_logger
from scripts.utils.geo import bbox as _geo_bbox

logger = get_logger("osm_extractor")

# ── Local OSM source ──────────────────────────────────────────────────────────
OSM_DIR   = "/srv/THESIS/osm_planet"
LOCAL_PBF = os.path.join(OSM_DIR, "osm_use.osm.pbf")   # filtered planet (live fallback)
TILES_DIR = os.path.join(OSM_DIR, "tiles")             # pre-extracted 512 m tiles (fast path)
OSMIUM    = "/home/ws/udyyk/.local/opt/osmenv/bin/osmium"

def _tile_name(lat, lon):
    """Deterministic tile filename for a coord (matches osm_batch_extract)."""
    return f"{lat:.5f}_{lon:.5f}.osm.pbf"


# ── Bounding Box Helper ───────────────────────────────────────────────────────

def generate_bbox(lat, lon, size_m=512):
    """
    Generate a 512x512m bounding box centred on (lat, lon), as a dict.
    Math lives in scripts.utils.geo.bbox(); this just reshapes + rounds its output
    (dict shape kept for backwards compatibility with existing callers).
    """
    min_lat, max_lat, min_lon, max_lon = _geo_bbox(lat, lon, size_m)
    return {
        "min_lat": round(min_lat, 6),
        "max_lat": round(max_lat, 6),
        "min_lon": round(min_lon, 6),
        "max_lon": round(max_lon, 6),
    }


# ── Local OSM readers ─────────────────────────────────────────────────────────


def _geojson_to_elements(geojson_text):
    """Parse osmium 'geojsonseq' output -> unified element list.

    Every OSM tag is preserved verbatim as the feature's properties — nothing is
    dropped, so segmap/detection/future code can read any tag, not just the schema.
    """
    from shapely.geometry import shape
    elements = []
    for line in geojson_text.splitlines():
        line = line.strip().lstrip("\x1e")
        if not line:
            continue
        try:
            f = json.loads(line)
            elements.append({"tags": f.get("properties", {}) or {},
                             "geom": shape(f["geometry"]) if f.get("geometry") else None})
        except Exception:
            continue
    return elements


def _read_tile(tile_path):
    """FAST path (~5-12 ms): export a pre-extracted 512 m tile, full tags."""
    out = subprocess.run([OSMIUM, "export", tile_path, "-f", "geojsonseq"],
                         capture_output=True, text=True).stdout
    return _geojson_to_elements(out)


def _read_pbf_extract(min_lat, max_lat, min_lon, max_lon):
    """Live fallback: carve the bbox out of the FULL planet (complete tags) on demand.

    Slower (scans the source), used only when no pre-extracted tile exists.
    """
    src = os.path.join(OSM_DIR, "planet-latest.osm.pbf")
    if not os.path.exists(src):
        src = LOCAL_PBF
    with tempfile.NamedTemporaryFile(suffix=".osm.pbf") as tmp:
        subprocess.run(
            [OSMIUM, "extract", "-b", f"{min_lon},{min_lat},{max_lon},{max_lat}",
             src, "-o", tmp.name, "--overwrite", "-s", "smart"],
            check=True, capture_output=True,
        )
        out = subprocess.run([OSMIUM, "export", tmp.name, "-f", "geojsonseq"],
                             capture_output=True, text=True).stdout
    return _geojson_to_elements(out)


def read_local_osm(min_lat, max_lat, min_lon, max_lon, lat=None, lon=None):
    """Return a unified element list [{'tags': {...}, 'geom': shapely-or-None}, ...].

    FAST path: read the pre-extracted 512 m tile for this coord (~5-12 ms, full tags).
    FALLBACK: live osmium extract from the full planet if the tile isn't present.

    lat/lon: the ORIGINAL coordinate, if the caller has it — used as-is for the tile
    filename lookup. Prefer this over the (min_lat+max_lat)/2 reconstruction below:
    some callers (e.g. generate_bbox) round min/max independently to 6dp, and
    averaging two independently-rounded edges back together doesn't always
    round-trip to the same 5dp value osm_batch_extract.py used to name the tile —
    a coordinate landing near a rounding boundary gets a false "tile missing" and
    falls all the way through to a live multi-minute rescan of the full planet
    (found 2026-08-10 chasing an 18-coordinate test run stall on "Dense Urban").
    """
    if lat is None or lon is None:
        lat = (min_lat + max_lat) / 2
        lon = (min_lon + max_lon) / 2
    tile = os.path.join(TILES_DIR, _tile_name(lat, lon))
    if os.path.exists(tile):
        try:
            return _read_tile(tile)
        except Exception as exc:
            logger.warning(f"Tile read failed ({exc}) — falling back to live extract")
    if os.path.exists(LOCAL_PBF) or os.path.exists(os.path.join(OSM_DIR, "planet-latest.osm.pbf")):
        return _read_pbf_extract(min_lat, max_lat, min_lon, max_lon)
    logger.error(f"No local OSM source found (tile in {TILES_DIR} / {LOCAL_PBF})")
    return []


# ── Compatibility shims for map generators (segmap / detection / ghsl) ────────
# These callers were written for the old Overpass "out geom" element format
# ({type, tags, geometry:[{lat,lon}]}). The offline reader returns shapely geoms,
# so we convert here and the map parsers (extract_osm_buildings / extract_osm_infra)
# keep working unchanged.

def circuit_breaker_wait():
    """No-op. Offline OSM has no rate limit / circuit breaker. Kept for API parity."""
    return None


def _shapely_to_overpass_elements(elements):
    """Convert offline elements [{'tags', 'geom'(shapely)}] → Overpass 'out geom'
    format [{type, tags, geometry:[{lat,lon}]} | {type:node, lat, lon, tags}]."""
    def ring(coords):
        return [{"lat": float(y), "lon": float(x)} for x, y in coords]

    out = []
    for el in elements:
        tags = el.get("tags") or {}
        geom = el.get("geom")
        if geom is None:
            continue
        gt = geom.geom_type
        if gt == "Point":
            out.append({"type": "node", "tags": tags,
                        "lat": float(geom.y), "lon": float(geom.x)})
        elif gt == "LineString":
            out.append({"type": "way", "tags": tags, "geometry": ring(geom.coords)})
        elif gt == "Polygon":
            out.append({"type": "way", "tags": tags, "geometry": ring(geom.exterior.coords)})
        elif gt in ("MultiPolygon", "MultiLineString", "GeometryCollection"):
            for sub in geom.geoms:
                st = sub.geom_type
                if st == "Polygon":
                    out.append({"type": "way", "tags": tags, "geometry": ring(sub.exterior.coords)})
                elif st == "LineString":
                    out.append({"type": "way", "tags": tags, "geometry": ring(sub.coords)})
                elif st == "Point":
                    out.append({"type": "node", "tags": tags,
                                "lat": float(sub.y), "lon": float(sub.x)})
    return out


def fetch_osm_elements_offline(min_lat, max_lat, min_lon, max_lon, lat=None, lon=None):
    """Offline replacement for the old Overpass fetch — returns Overpass-format
    elements for the bbox, read from the local planet tile (live fallback if absent)."""
    return _shapely_to_overpass_elements(
        read_local_osm(min_lat, max_lat, min_lon, max_lon, lat=lat, lon=lon))


# ── Response Parser ───────────────────────────────────────────────────────────

def parse_osm_response(elements):
    """
    Parse local-OSM elements into a flat feature dict.

    Each element is {'tags': {...}, 'geom': ...}; only `tags` is used here.

    Design rules (unchanged):
      - Absent tag      → None   (stripped from output by extract_osm_features)
      - Boolean feature → True if present, None if absent  (never False)
      - Count feature   → integer (0 is valid for building_count)
      - Multi-value     → comma-separated unique values
    """
    buildings         = []
    power_plants      = []
    power_substations = []
    power_lines       = []
    power_towers      = []
    generator_sources = []
    plant_sources     = []
    landuses          = []
    waterways         = []
    amenities         = []

    for el in elements:
        tags = el.get("tags", {})
        if not tags:
            continue

        if "building" in tags:
            buildings.append(tags)

        power_val = tags.get("power")
        if power_val == "plant":
            power_plants.append(tags)
        elif power_val == "substation":
            power_substations.append(tags)
        elif power_val == "line":
            power_lines.append(tags)
        elif power_val == "tower":
            power_towers.append(tags)

        if "generator:source" in tags:
            generator_sources.append(tags["generator:source"])
        if "plant:source" in tags:
            plant_sources.append(tags["plant:source"])

        if "landuse" in tags:
            landuses.append(tags["landuse"])

        if "waterway" in tags:
            waterways.append(tags["waterway"])

        amenity_val = tags.get("amenity")
        if amenity_val in ("hospital", "school", "charging_station"):
            amenities.append(amenity_val)

    def most_common(lst):
        return Counter(lst).most_common(1)[0][0] if lst else None

    def unique_joined(lst):
        if not lst:
            return None
        seen = []
        for v in lst:
            if v not in seen:
                seen.append(v)
        return ",".join(seen)

    def presence(lst):
        return True if lst else None

    building_types = [
        b.get("building") for b in buildings
        if b.get("building") not in (None, "yes")
    ]
    building_start_dates = [b.get("start_date")   for b in buildings if b.get("start_date")]
    roof_shapes          = [b.get("roof:shape")    for b in buildings if b.get("roof:shape")]
    roof_materials       = [b.get("roof:material") for b in buildings if b.get("roof:material")]
    roof_colours         = [b.get("roof:colour")   for b in buildings if b.get("roof:colour")]

    features = {
        "osm_building_count":      len(buildings),
        "osm_building_type":       most_common(building_types),
        "osm_building_start_date": min(building_start_dates) if building_start_dates else None,
        "osm_roof_shape":          most_common(roof_shapes),
        "osm_roof_material":       most_common(roof_materials),
        "osm_roof_colour":         most_common(roof_colours),

        "osm_power_plant":         presence(power_plants),
        "osm_power_substation":    presence(power_substations),
        "osm_power_line":          presence(power_lines),
        "osm_power_tower":         presence(power_towers),
        "osm_generator_source":    unique_joined(generator_sources),
        "osm_plant_source":        unique_joined(plant_sources),

        "osm_landuse":             most_common(landuses),
        "osm_waterway":            most_common(waterways),

        "osm_amenity":             unique_joined(amenities),
    }
    return features


# ── Single Coordinate Extractor ───────────────────────────────────────────────

def extract_osm_features(lat, lon, min_lat, max_lat, min_lon, max_lon):
    """
    Offline extraction pipeline for one coordinate:
      1. Read energy-relevant OSM elements in the 512m bbox from the LOCAL planet
      2. Parse into the flat feature dict (same schema as before)
      3. Strip all None values — absent key = no data (null contract preserved)
         Exception: osm_building_count kept even if 0 — 0 is valid data
      4. Return dict with coordinate metadata prepended

    Null contract (unchanged):
      - Feature present in OSM   → key: value in dict
      - Feature absent from OSM  → key not in dict at all
    """
    logger.info(f"Extracting OSM features for ({lat}, {lon}) [offline]")

    elements = read_local_osm(min_lat, max_lat, min_lon, max_lon, lat=lat, lon=lon)

    if not elements:
        logger.warning(
            f"Zero elements for ({lat}, {lon}) — all features absent "
            "(sparse OSM region or coverage gap)"
        )

    features = parse_osm_response(elements)

    features = {
        k: v for k, v in features.items()
        if v is not None or k == "osm_building_count"
    }

    result = {
        "lat":               lat,
        "lon":               lon,
        "raw_element_count": len(elements),
    }
    result.update(features)

    logger.info(
        f"({lat}, {lon}) — {len(elements)} elements → "
        f"{len(features)} features populated "
        f"(building_count={features.get('osm_building_count', 0)})"
    )
    return result


# ── M3 Support: energy element positions for Gaussian density signal ─────────

_ELEMENT_WEIGHTS = {
    ("power", "plant"):       3.0,
    ("power", "substation"):  2.5,
    ("power", "generator"):   2.5,
    ("power", "line"):        1.5,
    ("power", "tower"):       1.0,
    ("power", "pole"):        0.8,
    ("man_made", "petroleum_well"): 3.0,
    ("man_made", "storage_tank"):   2.0,
    ("man_made", "pipeline"):       1.5,
    ("waterway", "dam"):      2.5,
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
    Return energy-relevant OSM element positions for M3 Gaussian density.

    Each entry: {"lat": float, "lon": float, "weight": float}, positioned at the
    element's representative point (centroid for ways/areas).
    Null contract: empty list if no elements or local source unavailable.
    """
    elements = read_local_osm(min_lat, max_lat, min_lon, max_lon, lat=lat, lon=lon)
    if not elements:
        return []

    result = []
    for el in elements:
        tags = el.get("tags", {})
        if not tags:
            continue
        weight = _get_element_energy_weight(tags)
        if weight <= 0:
            continue
        geom = el.get("geom")
        if geom is None:
            continue
        try:
            pt = geom.representative_point()
            result.append({"lat": float(pt.y), "lon": float(pt.x), "weight": weight})
        except Exception:
            continue

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

    df = pd.DataFrame(results)
    feature_cols = [c for c in df.columns if c.startswith("osm_")]

    print("\n=== OSM Extraction Test (OFFLINE) — Results Summary ===")
    print(f"Source             : {'GeoPackage (fast)' if os.path.exists(LOCAL_GPKG) else 'osmium extract (fallback)'}")
    print(f"Coordinates tested : {len(df)}")
    print(f"Features per coord : {len(feature_cols)}")

    print("\nFeature coverage per coordinate:")
    for _, row in df.iterrows():
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

    print("\nKey features table:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    display_cols = [
        c for c in [
            "stratum", "osm_building_count", "osm_building_type",
            "osm_landuse", "osm_waterway", "osm_power_plant", "osm_power_line",
            "raw_element_count"
        ] if c in df.columns
    ]
    print(df[display_cols].to_string(index=False))
