#!/usr/bin/env python3
"""
real_anchor_finder.py — Find N verified coordinates per stratum using OSM + ESA + GHSL.

Methodology (3 phases per stratum, each checked against quota before starting the next
— set N_PER_STRATUM as high as you want, e.g. 250000; it is a ceiling, not a promise.
Every stratum naturally stops once its real-world supply runs out, whichever phase
that happens in. No phase ever loosens the ESA+GHSL gate to find more — only which
candidates get proposed to that gate changes.):

  Phase 1 — OSM discovery. The stratum's OSM_QUERIES entry is split into small,
    regional chunks (existing bboxes kept as single chunks; any bbox-less/global
    clause is expanded across a 9-region continent grid) and fetched ONE CHUNK AT A
    TIME, politely, from Overpass. Each chunk's candidates are verified in parallel
    (ESA WorldCover + GHSL, unchanged gate), then deduped against everything already
    accepted using a spatial index (not naive pairwise distance — matters once a
    stratum has tens of thousands of accepted points). Stops as soon as quota is hit,
    partway through the chunk list if that's where it happens.

  Phase 2 — densify (only for strata whose real-world footprint is a large contiguous
    zone, not a single structural point — see DENSIFY_STRATA below). One OSM node is
    one candidate, but a real city core or industrial zone spans many independently-
    qualifying 512m patches. Tiles a 5x5 grid around every anchor already accepted in
    Phase 1, verifies each with the same gate, dedupes the same way. Only runs if
    Phase 1 didn't already reach quota.

  Phase 3 — GHSL Degree-of-Urbanisation seeding (Dense Urban + Suburban only). Seeds
    from rasters/ghsl/degurba/..._UC_V2_0.shp — 11,534 real Urban Centre polygons from
    satellite population/built-up measurement, completely independent of OSM tagging.
    Samples points inside each polygon's actual shape, verifies with the same gate.
    Fixes the geographic gap OSM-tag density leaves in poorly-mapped regions. Only
    runs if Phases 1+2 didn't already reach quota.

Usage
-----
    python scripts/bore/real_anchor_finder.py
"""

import csv
import json
import math
import random
import re
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

import fiona
import numpy as np
import requests as _requests
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from scripts.extractors.worldcover_extractor import extract_worldcover_features
from scripts.extractors.ghsl_extractor       import extract_ghsl_features
from scripts.bore.coordinate_filter      import (
    parse_esa_classes, parse_importance_tiers, build_strata_config,
    FEATURES_HTML, BREAKDOWN_HTML,
)

OUTPUT_CSV       = BASE / "outputs" / "csv" / "filtered_strata_sample.csv"
OVERPASS_TIMEOUT = 20    # per-mirror request timeout — a hanging mirror fails fast and
                          # falls through to the next one instead of blocking for minutes
                          # (learned live 2026-08-06: mirrors are intermittently slow/down;
                          # long timeouts turn one bad mirror into a multi-minute stall)
BBOX_M           = 512
MIN_DIST_M       = 512   # minimum separation between accepted coordinates (dedup)
RNG_SEED         = 42
RATE_DELAY       = 2.0   # seconds between Overpass requests — enforced GLOBALLY now
                          # (see _overpass_global's lock), not per-stratum, so this stays
                          # true even with multiple strata running concurrently
VERIFY_WORKERS   = 24    # thread pool size for ESA/GHSL checks — I/O-bound, no Overpass
                          # calls in this step, safe to parallelize regardless of N
STRATUM_WORKERS  = 1     # reduced from 6 (2026-08-09) — the original 6-way concurrency
                          # combined with a logging bug to produce a 26+ hour run that should
                          # have taken hours. The logging bug (fixed: launch with
                          # LOG_LEVEL=WARNING) was confirmed as A cause, but 6 strata each
                          # running their own VERIFY_WORKERS=24 pool simultaneously (up to
                          # 144 concurrent threads hitting ESA's S3 bucket + local GHSL
                          # rasters at once) was never actually verified safe at this scale,
                          # unlike the original one-stratum-at-a-time historical runs that
                          # reliably took hours, not days. Running one stratum at a time again
                          # — matching the proven-fast configuration — until 6-way concurrency
                          # is specifically re-tested and confirmed safe on its own.

# ── Scale control ─────────────────────────────────────────────────────────────
# ── THE ONLY KNOB — set N per stratum here ───────────────────────────────────
# Safe to set arbitrarily high (e.g. 250000) for every stratum — each one stops on
# its own once its real-world supply is exhausted, across all 3 phases. No error, no
# hang. See scale_ceiling_analysis.html for what each stratum's real ceiling looks
# like today — most land far below any large N, a handful (Dense Urban, Suburban)
# can genuinely reach into the tens of thousands to low hundred-thousands.
N_PER_STRATUM: dict[str, int] = {
    # Set to max 2026-08-07 — High/Medium/Small improvement strata (per the confirmed-run
    # comparison table), left at 1 only for the 4 strata already confirmed to show no
    # further improvement (hard ceiling or genuinely no new discovery mechanism added).
    "Industrial + Water":              250000,
    "Urban + Coastal":                 250000,
    "Informal + Urban":                250000,
    "Dense Urban":                     250000,
    "Suburban":                        250000,
    "Industrial":                      250000,
    "Data Centre + Industrial":        250000,
    "Industrial + Arid":               250000,
    "Agrivoltaics (Solar + Farmland)": 250000,
    "Industrial + Forest":             250000,
    "Hydropower Reservoir":            250000,
    "Agricultural + Water":            250000,
    "Coastal + Agricultural":          250000,
    "Mangrove + Industrial":           250000,
    "Suburban + Agricultural":         250000,
    "Coastal + Solar-Wind Hybrid":     250000,
}

# ── Ocean proximity filter ────────────────────────────────────────────────────
OCEAN_SHP      = BASE / "rasters" / "natural_earth" / "ne_10m_ocean.shp"
OCEAN_MAX_KM   = 50   # candidate must be within 50 km of the ocean polygon edge

def _load_ocean_union():
    """Load Natural Earth ocean polygon and return a single unioned geometry."""
    with fiona.open(OCEAN_SHP) as src:
        return unary_union([shape(f["geometry"]) for f in src])

# Loaded once at import time; skips gracefully if file missing (non-coastal runs).
try:
    _OCEAN_UNION = _load_ocean_union()
except Exception as _e:
    _OCEAN_UNION = None
    print(f"[WARN] Ocean shapefile not loaded ({_e}); ocean-proximity check disabled.")

def _is_near_ocean(lat: float, lon: float, max_km: float = OCEAN_MAX_KM) -> bool:
    """True if (lat, lon) is within max_km of the ocean polygon boundary."""
    if _OCEAN_UNION is None:
        return True   # fail-open: don't block if shapefile missing
    # Point is INSIDE the ocean polygon → distance() returns 0 → always passes
    # Point is OUTSIDE → distance() is degrees; 1° ≈ 111 km
    return Point(lon, lat).distance(_OCEAN_UNION) * 111 < max_km


# ── Per-stratum OSM entry queries ─────────────────────────────────────────────
# OSM is the entry point — it narrows the Earth to plausible locations.
# ESA + GHSL then verify the actual land cover / urban character.
# out counts are the per-CHUNK cap now (see split_query_into_chunks) — each existing
# named bbox, and each region a bbox-less clause expands into, gets its own full cap,
# not a cap shared across the whole stratum like a single unchunked query would have.

OSM_QUERIES: dict[str, str] = {
    # Docks are water basins by definition — water guaranteed in bbox.
    # Cranes sit on the quay at the water's edge — bbox captures both built_up + water.
    # Both tags imply the wt≥20% + bu≥35% co-presence that wastewater_plant could not guarantee.
    # out raised 3000→8000 (2026-08-06): confirmed 14.48% true pass rate (N=1000 exhaustive run,
    # see scale_ceiling_analysis.html) means ~1150 verified anchors achievable at this volume.
    "Industrial + Water": """
        [out:json][timeout:120];
        (way[waterway=dock];
         node[man_made=crane];
         node[man_made=petroleum_refinery];
         way[man_made=petroleum_refinery];);
        out center 8000;""",

    "Urban + Coastal": """
        [out:json][timeout:60];
        node[harbour=yes];
        out 4000;""",

    # Expanded 2026-08-06: added South Asia + Latin America — informal settlements are a
    # global phenomenon, not Africa-specific; Africa+ME kept as the original validated core.
    "Informal + Urban": """
        [out:json][timeout:60];
        (node[place=city](-5,10,15,45);
         node[place=city](8,68,35,90);
         node[place=city](-35,-75,15,-35););
        out 3000;""",

    "Dense Urban": """
        [out:json][timeout:60];
        node[place=city];
        out 5000;""",

    # Global small towns — targeted bboxes to avoid server timeout from querying large regions.
    # ESA gate (built_up≥50%, pop≤500, h≤15m) selects low-rise peri-urban fabric globally.
    # Africa deliberately excluded: ESA over-classifies small African settlements as 99%+
    # built_up → bbox lands on forest-settlement fringe, doesn't look suburban.
    # Expanded 2026-08-06: added North America, East Asia, South Asia, Middle East,
    # Western Europe, Oceania alongside the original SE Asia/E.Europe/S.America core.
    "Suburban": """
        [out:json][timeout:90];
        (node[place=town](10,100,20,110);
         node[place=town](46,20,52,32);
         node[place=town](-23,-50,-15,-40);
         node[place=town](25,-125,50,-65);
         node[place=town](30,100,45,145);
         node[place=town](8,68,35,90);
         node[place=town](15,35,40,60);
         node[place=town](40,-10,55,15);
         node[place=town](-45,110,-10,155);
         node[place=town](50,45,60,90);
         node[place=town](15,-90,25,-75););
        out 3000;""",

    # Cooling towers (nodes + ways) + power plants (all fuel types) + ALL landuse=industrial
    # — broadened 2026-08-06: the old tag set ([industrial=manufacturing] subfilter, coal/gas
    # only) hit a confirmed real ceiling (no chunk hit its cap) — more query volume couldn't
    # add anything further. Dropped the manufacturing subfilter (was excluding logistics/
    # warehousing/chemical-plant landuse=industrial land) and added oil/biomass/geothermal/
    # nuclear power plant types — a genuinely bigger real-world population.
    "Industrial": """
        [out:json][timeout:120];
        (node[man_made=cooling_tower];
         way[man_made=cooling_tower];
         way[power=plant]["plant:source"=coal];
         way[power=plant]["plant:source"=gas];
         way[power=plant]["plant:source"=oil];
         way[power=plant]["plant:source"=biomass];
         way[power=plant]["plant:source"=geothermal];
         way[power=plant]["plant:source"=nuclear];
         node[power=plant]["plant:source"=coal];
         node[power=plant]["plant:source"=gas];
         node[power=plant]["plant:source"=oil];
         way[landuse=industrial];);
        out center 10000;""",

    # Bboxed to established hyperscale DC corridors — avoids small ISP offices mislabelled as
    # data_center in OSM (e.g. residential Barquisimeto, Venezuela that passed globally).
    # NOT globalized — that OSM-mistagging failure pattern would likely recur elsewhere,
    # not just in the region it was first found (see docs/scale_ceiling_analysis.html notes).
    "Data Centre + Industrial": """
        [out:json][timeout:75];
        (node[telecom=data_center](38,-78,40,-76);
         way[building=data_center](38,-78,40,-76);
         way[building=data_centre](38,-78,40,-76);
         node[telecom=data_center](32,-97,33,-96);
         way[building=data_center](32,-97,33,-96);
         node[telecom=data_center](41,-88,42,-87);
         way[building=data_center](41,-88,42,-87);
         node[telecom=data_center](52,4.5,53,5.5);
         way[building=data_center](52,4.5,53,5.5);
         way[building=data_centre](52,4.5,53,5.5);
         node[telecom=data_center](50,8,51,9);
         way[building=data_center](50,8,51,9);
         node[telecom=data_center](1,103,2,104);
         way[building=data_center](1,103,2,104);
         node[telecom=data_center](51,-1,52,1);
         way[building=data_center](51,-1,52,1);
         way[building=data_centre](51,-1,52,1););
        out center 2000;""",

    # Deep-desert bboxes: petroleum wells/flares land in open desert (bare_sparse high).
    # way[man_made=works] + way[landuse=industrial] have real building footprints,
    # guaranteeing built_up in bbox. Australian Pilbara: iron ore mines in zero-rainfall desert.
    # Expanded 2026-08-07: added US Permian Basin (Texas oil) + Kazakhstan (Caspian oil
    # fields) — same tag set, same gate, two more real arid-industrial regions not yet
    # searched. Self-limiting (bare_sparse≥40% + the tight pop/bsurf gates) so no new
    # correctness risk, same reasoning as the original bboxes.
    "Industrial + Arid": """
        [out:json][timeout:75];
        (node[man_made=petroleum_well](17,44,24,56);
         node[man_made=petroleum_well](26,8,30,18);
         node[man_made=petroleum_well](25,28,29,33);
         node[man_made=petroleum_well](31,-104,34,-100);
         node[man_made=petroleum_well](43,50,50,70);
         node[man_made=gas_well](17,44,24,56);
         node[man_made=gas_well](26,8,30,18);
         node[man_made=gas_well](31,-104,34,-100);
         node[man_made=gas_well](43,50,50,70);
         node[man_made=flare](17,44,24,56);
         node[man_made=flare](26,8,30,18);
         node[man_made=flare](31,-104,34,-100);
         node[man_made=flare](43,50,50,70);
         way[man_made=works](17,44,24,56);
         way[man_made=works](26,8,30,18);
         way[man_made=works](25,28,29,33);
         way[man_made=works](31,-104,34,-100);
         way[man_made=works](43,50,50,70);
         way[landuse=industrial](17,44,24,56);
         way[landuse=industrial](26,8,30,18);
         way[landuse=industrial](-26,117,-20,121);
         way[landuse=industrial](31,-104,34,-100);
         way[landuse=industrial](43,50,50,70);
         way[man_made=works](-26,117,-20,121);
         way[landuse=quarry](-26,117,-20,121);
         way[power=generator]["generator:source"=solar](17,44,24,56);
         way[power=generator]["generator:source"=solar](26,8,30,18);
         way[power=generator]["generator:source"=solar](-30,-70,-22,-65);
         way[power=generator]["generator:source"=solar](-30,115,-22,125););
        out center 5000;""",

    # Onshore coastal bboxes — turbines at the land-sea boundary.
    # NOT globalized — offshore turbines in open water (China Jiangsu/Bohai, already removed)
    # read as water=100% with no coastal signature; this failure pattern recurs at any
    # offshore wind farm worldwide, not just the region it was first found in.
    "Coastal + Solar-Wind Hybrid": """
        [out:json][timeout:120];
        (way[power=generator]["generator:source"=wind](53,5,57,12);
         way[power=generator]["generator:source"=wind](51,2.5,53,5);
         way[power=generator]["generator:source"=wind](56,10,59,13);
         way[power=generator]["generator:source"=wind](58,4,62,8);
         way[power=generator]["generator:source"=wind](50,-5,58,2);
         way[power=generator]["generator:source"=wind](36,-9,42,-7);
         way[power=generator]["generator:source"=wind](34,130,38,135);
         way[power=generator]["generator:source"=wind](34,126,37,131);
         way[power=generator]["generator:source"=wind](14,-96,18,-92);
         way[power=generator]["generator:source"=wind](-36,136,-34,141);
         node[power=generator]["generator:source"=wind](56,10,59,13);
         node[power=generator]["generator:source"=wind](58,4,62,8););
        out center 2500;""",

    # NOT globalized — floating solar on fish ponds (Anhui/Jiangsu, already removed) miscodes
    # ESA "cropland" from adjacent paddy; this failure pattern recurs at any aquaculture
    # region with nearby solar worldwide, not just the region it was first found in.
    # Expanded 2026-08-07: added Poland (solar+wheat) and Australia Murray-Darling
    # (solar+broadacre grain) — both genuinely dry cropland, deliberately NOT another
    # rice-paddy/aquaculture-adjacent region (that's exactly the Anhui/Jiangsu fish-pond
    # confusion this stratum already had to remove once).
    "Agrivoltaics (Solar + Farmland)": """
        [out:json][timeout:60];
        (way[power=generator]["generator:source"=solar](22,71,26,75);
         way[power=generator]["generator:source"=solar](28,68,32,74);
         way[power=generator]["generator:source"=solar](30,30,32,32);
         way[power=generator]["generator:source"=solar](35,113,38,120);
         way[power=generator]["generator:source"=solar](50,12,52,14);
         way[power=generator]["generator:source"=solar](40,-100,45,-92);
         way[power=generator]["generator:source"=solar](37,-7,40,-4);
         way[power=generator]["generator:source"=solar](43,0,47,3);
         way[power=generator]["generator:source"=solar](50,15,54,22);
         way[power=generator]["generator:source"=solar](-36,140,-32,148);
         way[power=plant]["plant:source"=solar](22,71,26,75);
         way[power=plant]["plant:source"=solar](28,68,32,74);
         way[power=plant]["plant:source"=solar](30,30,32,32);
         way[power=plant]["plant:source"=solar](35,113,38,120);
         way[power=plant]["plant:source"=solar](37,-7,40,-4);
         way[power=plant]["plant:source"=solar](43,0,47,3);
         way[power=plant]["plant:source"=solar](50,15,54,22);
         node[power=plant]["plant:source"=solar](50,12,52,14);
         node[power=plant]["plant:source"=solar](40,-100,45,-92);
         node[power=plant]["plant:source"=solar](37,-7,40,-4);
         node[power=plant]["plant:source"=solar](-36,140,-32,148););
        out center 5000;""",

    # Sawmills and timber processing facilities embedded in/adjacent to boreal/temperate forest.
    # Expanded 2026-08-06: added unrestricted node/way[man_made=sawmill] — self-limiting tag
    # (sawmills only exist near forests), safe to globalize alongside the original 6 regions.
    "Industrial + Forest": """
        [out:json][timeout:75];
        (node[man_made=sawmill](56,40,70,130);
         node[man_made=sawmill](56,20,70,40);
         node[man_made=sawmill](44,22,50,30);
         node[man_made=sawmill](48,-130,58,-115);
         node[man_made=sawmill](43,-125,48,-120);
         node[man_made=sawmill](-47,166,-43,172);
         way[man_made=sawmill](56,40,70,130);
         way[man_made=sawmill](56,20,70,40);
         way[man_made=sawmill](44,22,50,30);
         way[man_made=sawmill](48,-130,58,-115);
         way[man_made=sawmill](43,-125,48,-120);
         way[man_made=sawmill](-47,166,-43,172);
         way[industrial=sawmill](56,40,70,130);
         way[industrial=sawmill](56,20,70,40);
         way[industrial=sawmill](48,-130,58,-115);
         node[man_made=sawmill];
         way[man_made=sawmill];);
        out center 2000;""",

    # Dam way/node approach: centroid sits on the dam WALL — 512m bbox captures concrete wall
    # + terrain + reservoir tail. Expanded 2026-08-06: added unrestricted way/node[waterway=dam]
    # alongside the original 8 named river-basin regions.
    "Hydropower Reservoir": """
        [out:json][timeout:120];
        (way[waterway=dam](26,96,33,110);
         node[waterway=dam](26,96,33,110);
         way[waterway=dam](-27,-54,-20,-46);
         node[waterway=dam](-27,-54,-20,-46);
         way[waterway=dam](59,6,71,32);
         node[waterway=dam](59,6,71,32);
         way[waterway=dam](10,74,25,84);
         node[waterway=dam](10,74,25,84);
         way[waterway=dam](44,-122,50,-116);
         node[waterway=dam](44,-122,50,-116);
         way[waterway=dam](37,34,42,44);
         node[waterway=dam](37,34,42,44);
         way[waterway=dam](-18,26,-15,30);
         node[waterway=dam](-18,26,-15,30);
         way[waterway=dam](22,30,24,34);
         node[waterway=dam](22,30,24,34);
         way[waterway=dam];
         node[waterway=dam];);
        out center 5000;""",

    # Nodes at the structural water-cropland boundary: weir/floodgate/sluice_gate/pumping_station.
    # Expanded 2026-08-06: added unrestricted clauses alongside the original named deltas.
    "Agricultural + Water": """
        [out:json][timeout:75];
        (node[waterway=weir](22,28,32,36);
         node[waterway=weir](24,67,34,74);
         node[waterway=weir](9,100,22,108);
         node[waterway=weir](30,110,40,125);
         node[waterway=weir](9,104,12,107);
         node[man_made=floodgate](22,28,32,36);
         node[man_made=floodgate](24,67,34,74);
         node[man_made=floodgate](9,100,22,108);
         node[man_made=floodgate](9,104,12,107);
         node[man_made=pumping_station](22,28,32,36);
         node[man_made=pumping_station](24,67,34,74);
         node[man_made=pumping_station](9,104,12,107);
         node[waterway=sluice_gate](24,67,34,74);
         node[waterway=sluice_gate](9,100,22,108);
         node[waterway=sluice_gate](9,104,12,107);
         node[waterway=weir];
         node[man_made=floodgate];
         node[man_made=pumping_station];
         node[waterway=sluice_gate];);
        out 4000;""",

    # way[natural=coastline] in delta-FREE ocean-facing rice coasts ONLY.
    # NOT globalized — river-mouth/tidal channels (Myanmar, Bangladesh, already removed)
    # miscode as open sea in ESA; this recurs at any major river delta worldwide, and this
    # stratum is already hard-capped (confirmed exhaustive ceiling ~39), little to gain anyway.
    "Coastal + Agricultural": """
        [out:json][timeout:90];
        (way[natural=coastline](17,120,19,121);
         way[natural=coastline](10,123,12,125);
         way[natural=coastline](16,107,18,109);
         way[natural=coastline](-7,108,-5,112);
         way[natural=coastline](12,99.5,14,101);
         way[natural=coastline](8.5,104.5,9.5,105.3);
         way[natural=coastline](8,81,10,82);
         way[natural=coastline](19,84,21,87);
         way[natural=coastline](23,120,25,121);
         way[natural=coastline](34,126,36,127););
        out center 4000;""",

    # Insight: mangrove IS the limiting class — query for it, let ESA check built_up.
    # Expanded 2026-08-06: added unrestricted way[wetland=mangrove] — self-limiting tag
    # (mangroves only exist in tropical/subtropical coasts).
    "Mangrove + Industrial": """
        [out:json][timeout:90];
        (way[wetland=mangrove](4,6.5,5,7.5);
         way[wetland=mangrove](4.5,5,6,6.5);
         way[wetland=mangrove](22,89,23,90);
         way[wetland=mangrove](2.8,117,4,118.5);
         way[wetland=mangrove](0,108,6,119);
         way[wetland=mangrove](8,99,12,107);
         way[wetland=mangrove];);
        out center 1500;""",

    # Allotment ways + orchards/vineyards in peri-urban rings — always at city-cropland edge.
    # Expanded 2026-08-06: added unrestricted clauses alongside the original named regions.
    "Suburban + Agricultural": """
        [out:json][timeout:75];
        (way[landuse=allotments](28,72,32,78);
         way[landuse=allotments](44,8,47,13);
         way[landuse=allotments](30,29,32,32);
         way[landuse=allotments](50,6,52,10);
         way[landuse=allotments](30,118,33,122);
         way[landuse=allotments](38,-10,42,-6);
         node[landuse=orchard](28,72,32,78);
         node[landuse=orchard](44,8,47,13);
         node[landuse=orchard](50,6,52,10);
         node[landuse=orchard](38,-10,42,-6);
         node[landuse=vineyard](44,8,47,13);
         node[landuse=vineyard](47,1,49,5);
         node[landuse=vineyard](38,-10,42,-6);
         way[landuse=allotments];
         node[landuse=orchard];
         node[landuse=vineyard];);
        out center 2500;""",

}


# ── Overpass mirrors + fetch ────────────────────────────────────────────────────
# Ordered by live-tested reliability (2026-08-06, this network): mail.ru fastest/most
# reliable, overpass-api.de intermittently busy (504, not down), kumi.systems hung with
# no response every time tested. Order matters — first mirror tried gets used most.
_OVERPASS_MIRRORS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
_OVERPASS_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": "thesis-anchor-finder/1.0 (energy_profiling research; contact bharathhk18.bk@gmail.com)",
    "Accept": "application/json",
}


# Global lock, not per-stratum: 2026-08-07, strata now run concurrently (see main()),
# but Overpass itself must stay exactly as polite as a single-stratum run — at most one
# HTTP request in flight at any moment, RATE_DELAY spacing between the end of one call
# and the start of the next, REGARDLESS of which stratum's thread is asking. This is
# what makes concurrency safe: only the non-Overpass work (verify/densify/GHSL) actually
# overlaps across strata; the Overpass-facing part is serialized exactly as before.
_OVERPASS_LOCK = threading.Lock()
_last_overpass_call = [0.0]


def _overpass_global(query: str) -> list:
    """Direct Overpass request — tries each mirror, with one retry on 429/504.
    Globally rate-limited across all concurrently-running strata via _OVERPASS_LOCK."""
    with _OVERPASS_LOCK:
        elapsed = time.time() - _last_overpass_call[0]
        if elapsed < RATE_DELAY:
            time.sleep(RATE_DELAY - elapsed)
        try:
            for attempt in range(2):
                for url in _OVERPASS_MIRRORS:
                    try:
                        resp = _requests.post(
                            url,
                            data={"data": query},
                            headers=_OVERPASS_HEADERS,
                            timeout=OVERPASS_TIMEOUT,
                        )
                        if resp.status_code == 200:
                            return resp.json().get("elements", [])
                        if resp.status_code in (429, 504) and attempt == 0:
                            time.sleep(15)
                            break
                        print(f"\n  [HTTP {resp.status_code} {url}]", end=" ", flush=True)
                    except Exception as exc:
                        print(f"\n  [WARN {url}: {exc}]", end=" ", flush=True)
            return []
        finally:
            _last_overpass_call[0] = time.time()


# ── Query chunking ───────────────────────────────────────────────────────────────
# Splits one OSM_QUERIES entry into small, standalone Overpass requests — one per
# existing bbox clause (kept as-is), or one per synthetic continent bbox for clauses
# with no bbox of their own (i.e. currently-global, unbounded tags). Avoids a single
# oversized request that Overpass may silently truncate or time out on — this is what
# makes N_PER_STRATUM=250000 safe to actually try: each chunk is small and reliable,
# quota is checked after every one, so the search just stops when supply runs out.

_GLOBAL_GRID = [
    (15, -170, 72, -50),    # North America
    (-56, -82, 13, -34),    # South America
    (35, -25, 72, 45),      # Europe
    (-35, -20, 38, 52),     # Africa
    (12, 34, 42, 63),       # Middle East
    (5, 60, 38, 100),       # South Asia
    (18, 100, 55, 150),     # East Asia
    (-50, 95, 25, 180),     # SE Asia + Oceania
    (40, 45, 78, 180),      # Russia + Central Asia
]

_CLAUSE_RE  = re.compile(r'(way|node)((?:\[[^\]]+\])+)(\([\-\d.,\s]+\))?')
_OUT_N_RE   = re.compile(r'out\s+(?:center\s+)?(\d+)\s*;')
_TIMEOUT_RE = re.compile(r'timeout:(\d+)')


def split_query_into_chunks(raw_query: str) -> list[str]:
    """Parse an OSM_QUERIES entry into a list of small, standalone Overpass queries."""
    out_n_match = _OUT_N_RE.search(raw_query)
    tmo_match   = _TIMEOUT_RE.search(raw_query)
    out_n   = out_n_match.group(1) if out_n_match else "3000"
    timeout = tmo_match.group(1) if tmo_match else "60"

    chunks = []
    for m in _CLAUSE_RE.finditer(raw_query):
        elem_type, tags, bbox = m.groups()
        if bbox:
            clause = f"{elem_type}{tags}{bbox}"
            chunks.append(f'[out:json][timeout:{timeout}];{clause};out center {out_n};')
        else:
            for (lat1, lon1, lat2, lon2) in _GLOBAL_GRID:
                clause = f"{elem_type}{tags}({lat1},{lon1},{lat2},{lon2})"
                chunks.append(f'[out:json][timeout:{timeout}];{clause};out center {out_n};')
    return chunks


# ── Geo helpers ───────────────────────────────────────────────────────────────

def _bbox(lat: float, lon: float, size_m: int = BBOX_M) -> dict:
    half = size_m / 2
    dlat = half / 111320
    dlon = half / (111320 * math.cos(math.radians(abs(lat) or 0.001)))
    return dict(min_lat=lat - dlat, max_lat=lat + dlat,
                min_lon=lon - dlon, max_lon=lon + dlon)


def _center(el: dict) -> tuple[float | None, float | None]:
    if "lat" in el and "lon" in el:
        return float(el["lat"]), float(el["lon"])
    c = el.get("center") or {}
    if "lat" in c:
        return float(c["lat"]), float(c["lon"])
    return None, None


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _too_close(lat: float, lon: float, found: list[dict]) -> bool:
    """True if (lat, lon) is within MIN_DIST_M of any already-accepted coordinate.
    O(n) — fine for small `found` lists; _kdtree_dedup below is used instead once a
    stratum has enough accepted anchors that this would be too slow."""
    for r in found:
        if _haversine_m(lat, lon, float(r["lat"]), float(r["lon"])) < MIN_DIST_M:
            return True
    return False


def _kdtree_dedup(candidates: list[dict], already_found: list[dict],
                   min_dist_m: float = MIN_DIST_M) -> list[dict]:
    """
    Given new candidate dicts (each with 'lat','lon') and already-accepted dicts,
    return the candidates that survive greedy min-distance suppression against BOTH
    already_found and each other, processed in list order (earlier = kept over a
    later near-duplicate). Uses an ECEF-projected KD-tree — O(n log n), not the
    O(n^2) pairwise haversine loop _too_close would need at this scale. Learned from
    a real bug during today's multi-script merge: exact-coordinate dedup missed
    near-duplicates from independently-run sources; only a real distance check catches
    that correctly.
    """
    if not candidates:
        return []
    R = 6_371_000

    def to_ecef(lat, lon):
        lat_r, lon_r = np.radians(lat), np.radians(lon)
        return (R * np.cos(lat_r) * np.cos(lon_r),
                R * np.cos(lat_r) * np.sin(lon_r),
                R * np.sin(lat_r))

    all_pts = already_found + candidates
    lats = np.array([float(p["lat"]) for p in all_pts])
    lons = np.array([float(p["lon"]) for p in all_pts])
    x, y, z = to_ecef(lats, lons)
    pts = np.column_stack([x, y, z])
    tree = cKDTree(pts)

    n_existing = len(already_found)
    accepted = np.zeros(len(all_pts), dtype=bool)
    accepted[:n_existing] = True

    kept = []
    for i in range(n_existing, len(all_pts)):
        neighbors = tree.query_ball_point(pts[i], min_dist_m)
        if any(accepted[j] for j in neighbors if j != i):
            continue
        accepted[i] = True
        kept.append(candidates[i - n_existing])
    return kept


def _wc_pcts(wc: dict) -> dict[str, float]:
    raw = wc.get("wc_classes_json") or "{}"
    return {cls: v["pct"] for cls, v in json.loads(raw).items()}


def _esa_ok(pcts: dict, sc: dict) -> bool:
    if pcts.get(sc["primary"], 0.0) < sc["primary_min"]:
        return False
    if sc.get("primary_hi") is not None and pcts.get(sc["primary"], 0.0) > sc["primary_hi"]:
        return False
    if sc["secondary"] and pcts.get(sc["secondary"], 0.0) < sc["secondary_min"]:
        return False
    return True


def _ghsl_ok(ghsl: dict | None, sc: dict) -> bool:
    if ghsl is None:
        return True
    pop   = ghsl.get("ghsl_population_per_km2") or 0
    h     = ghsl.get("ghsl_building_height_m")  or 0
    bsurf = ghsl.get("ghsl_built_surface_m2")
    if sc["pop_lo"]       and pop   < sc["pop_lo"]:                            return False
    if sc["pop_hi"]       and pop   > sc["pop_hi"]:                            return False
    if sc["h_lo"]         and h     < sc["h_lo"]:                              return False
    if sc["h_hi"]         and h     > sc["h_hi"]:                              return False
    if sc.get("bsurf_lo") and (bsurf is None or bsurf < sc["bsurf_lo"]):      return False
    if sc.get("bsurf_hi") and bsurf is not None and bsurf > sc["bsurf_hi"]:   return False
    return True


def _verify_point(sc: dict, lat: float, lon: float) -> dict | None:
    """One candidate's full ESA+GHSL+ocean check. Same logic regardless of which
    phase proposed the candidate — this is the one place correctness actually lives."""
    if lat is None or abs(lat) > 85:
        return None
    bb = _bbox(lat, lon)
    try:
        wc = extract_worldcover_features(lat, lon, bb["min_lat"], bb["max_lat"],
                                          bb["min_lon"], bb["max_lon"])
    except Exception:
        return None
    if wc is None:
        return None
    pcts = _wc_pcts(wc)
    if not _esa_ok(pcts, sc):
        return None

    needs_ghsl = any(sc.get(k) is not None
                     for k in ("pop_lo", "pop_hi", "h_lo", "h_hi", "bsurf_lo", "bsurf_hi"))
    ghsl = None
    if needs_ghsl:
        try:
            ghsl = extract_ghsl_features(lat, lon, bb["min_lat"], bb["max_lat"],
                                          bb["min_lon"], bb["max_lon"])
        except Exception:
            return None
        if not _ghsl_ok(ghsl, sc):
            return None

    if sc.get("requires_ocean") and not _is_near_ocean(lat, lon):
        return None

    p_pct = round(pcts.get(sc["primary"], 0.0), 1)
    s_pct = round(pcts.get(sc["secondary"], 0.0), 1) if sc["secondary"] else ""
    return {"lat": round(lat, 6), "lon": round(lon, 6),
            "primary_class": sc["primary"], "primary_pct": p_pct,
            "secondary_class": sc["secondary"] or "", "secondary_pct": s_pct}


# One shared pool for the whole process, not one per call/per stratum — strata now run
# concurrently (see main()), and each spinning up its own VERIFY_WORKERS threads would
# multiply uncontrolled (e.g. 6 concurrent strata x 24 threads = 144 at once). A single
# shared pool keeps total verification concurrency bounded regardless of how many strata
# are active. Not Overpass-facing, so no politeness concern here — just resource control.
_VERIFY_POOL = ThreadPoolExecutor(max_workers=VERIFY_WORKERS)


def _verify_batch_parallel(sc: dict, points: list[tuple[float, float]]) -> list[dict]:
    """Verify a batch of (lat,lon) candidates in parallel (I/O-bound, no Overpass calls
    in this step — safe regardless of N). Returns passing result dicts, order restored
    to input order for determinism."""
    if not points:
        return []
    indexed = []
    futures = {_VERIFY_POOL.submit(_verify_point, sc, lat, lon): idx
               for idx, (lat, lon) in enumerate(points)}
    for future in as_completed(futures):
        idx = futures[future]
        r = future.result()
        if r is not None:
            indexed.append((idx, r))
    indexed.sort(key=lambda t: t[0])
    return [r for _, r in indexed]


def _build_result(sc, importance, lat, lon, primary_class, primary_pct,
                   secondary_class, secondary_pct, remark: str) -> dict:
    return {
        "stratum_name":    sc["name"],
        "importance_tier": importance,
        "strata_type":     sc["kind"],
        "location_name":   f"{lat:.4f},{lon:.4f}",
        "lat":             round(lat, 6),
        "lon":             round(lon, 6),
        "primary_class":   primary_class,
        "primary_pct":     primary_pct,
        "secondary_class": secondary_class,
        "secondary_pct":   secondary_pct,
        "bbox_m":          BBOX_M,
        "remark":          remark,
    }


def _append_results(sc, importance, found: list[dict], passed: list[dict], remark: str,
                     n: int) -> None:
    """Dedup `passed` against `found` with the spatial index, append survivors (as
    built result dicts) to `found` in place, capped so `found` never exceeds n."""
    kept = _kdtree_dedup(passed, found)
    for r in kept:
        if len(found) >= n:
            break
        found.append(_build_result(sc, importance, r["lat"], r["lon"],
                                    r["primary_class"], r["primary_pct"],
                                    r["secondary_class"], r["secondary_pct"], remark))


# ── Densify (Phase 2) ────────────────────────────────────────────────────────────
# Strata whose real-world footprint is a large contiguous zone — moving 512m in any
# direction plausibly stays inside the same character. Deliberately excludes strata
# defined by a single structural point/boundary (a dam wall, a coastline, a river
# weir) — densifying those would measure something different, not more of the same
# thing. See bore_densify.py's original docstring (2026-08-06) for the full reasoning.
DENSIFY_STRATA = {
    "Industrial", "Dense Urban", "Suburban",
    "Industrial + Water", "Urban + Coastal",
    # Added 2026-08-07: these were wrongly excluded the first time around — a solar farm
    # or an oil/mining facility is a real contiguous zone (like Industrial), not a single
    # structural point (like a dam wall or a coastline). Same gate, same correctness
    # guarantee as every other densified stratum — this only changes what gets proposed.
    "Agrivoltaics (Solar + Farmland)",
    # Added later same day: same reasoning extended to 4 more real zone-type strata.
    # A sawmill complex, an oil/mining facility, a peri-urban allotment ring, and an
    # informal settlement are all contiguous areas, not single points — same gate, same
    # correctness. Still NOT added: Hydropower Reservoir, Agricultural + Water, the 3
    # coastal-boundary strata — those genuinely are single structural points (a dam wall,
    # a weir, a coastline), where densifying would measure something different, not more
    # of the same thing.
    "Industrial + Forest", "Industrial + Arid", "Suburban + Agricultural", "Informal + Urban",
}
GRID_RADIUS = 3   # 7x7 grid, 48 new points per seed — deepened from the original 5x5/24
                   # (2026-08-07) now that the mechanism is validated at real scale;
                   # more coverage per seed, especially helps the strata with the highest
                   # densify pass rates (Industrial, Dense Urban) where there's real signal
                   # further out from the seed, not just noise


def _grid_points(center_lat: float, center_lon: float, radius: int,
                  spacing_m: float) -> list[tuple[float, float]]:
    points = []
    dlat_per_step = spacing_m / 111_320
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i == 0 and j == 0:
                continue
            lat = center_lat + i * dlat_per_step
            dlon_per_step = spacing_m / (111_320 * math.cos(math.radians(abs(lat) or 0.001)))
            lon = center_lon + j * dlon_per_step
            points.append((lat, lon))
    return points


# ── GHSL degurba seeding (Phase 3) ────────────────────────────────────────────────
# Added Informal + Urban 2026-08-07: GHSL Urban Centre polygons mark real dense-population
# areas globally (independent of OSM place=city tagging density) — sampling points inside
# them and running Informal+Urban's own gate (built_up≥40%, height≤7m).
#
# CORRECTNESS FIX (2026-08-07, same day): built_up≥40%+height≤7m alone is NOT a reliable
# "informal settlement" signal without geographic context — it just means "moderately
# dense, low-rise," which matches huge amounts of ordinary development everywhere (a live
# run confirmed this: Atlanta GA and Guangzhou China both passed). Phase 1's OSM query for
# this stratum was ALWAYS deliberately restricted to Africa+ME/South Asia/Latin America for
# exactly this reason — Phase 3 needs the same restriction, not a global search, or it
# admits exactly the kind of false positive the region restriction exists to prevent.
# Dense Urban (built_up≥80%, much stricter) and Suburban (already validated globally
# region-by-region in Phase 1) don't have this problem — only Informal+Urban is restricted.
GHSL_SEED_STRATA = {"Dense Urban", "Suburban", "Informal + Urban"}
GHSL_SEED_REGIONS: dict[str, list[tuple[float, float, float, float]]] = {
    "Informal + Urban": [(-5, 10, 15, 45), (8, 68, 35, 90), (-35, -75, 15, -35)],
    # Same reasoning as Informal + Urban above, caught proactively before running (not from
    # a live failure this time): Suburban's gate has NO bsurf requirement at all (unlike
    # Dense Urban's strict bsurf_lo=2500), and its Phase 1 query deliberately excludes
    # Africa for a documented reason (ESA over-classifies small African settlements as
    # 99%+ built_up). An unrestricted global Phase 3 would readmit exactly that failure
    # mode. Reusing Phase 1's own 11 validated regions here — Phase 3 can never search
    # anywhere Phase 1 wouldn't have.
    "Suburban": [
        (10, 100, 20, 110), (46, 20, 52, 32), (-23, -50, -15, -40),
        (25, -125, 50, -65), (30, 100, 45, 145), (8, 68, 35, 90),
        (15, 35, 40, 60), (40, -10, 55, 15), (-45, 110, -10, 155),
        (50, 45, 60, 90), (15, -90, 25, -75),
    ],
}
UC_SHP = BASE / "rasters" / "ghsl" / "degurba" / "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_UC_V2_0.shp"
MAX_POINTS_PER_POLYGON = 50
_MOLL_TO_WGS84 = Transformer.from_crs("ESRI:54009", "EPSG:4326", always_xy=True)
_GHSL_POLYS_CACHE: list | None = None
_GHSL_LOAD_LOCK = threading.Lock()


def _polygon_centroid_latlon(feat) -> tuple[float, float]:
    p = feat["properties"]
    lon, lat = _MOLL_TO_WGS84.transform(p["PWCentr_X"], p["PWCentr_Y"])
    return lat, lon


def _polygon_in_regions(feat, regions: list[tuple[float, float, float, float]]) -> bool:
    lat, lon = _polygon_centroid_latlon(feat)
    return any(lat1 <= lat <= lat2 and lon1 <= lon <= lon2 for lat1, lon1, lat2, lon2 in regions)


def _load_ghsl_polygons() -> list:
    """Dense Urban, Suburban, and Informal+Urban can all hit this concurrently now —
    double-checked locking so the shapefile is only ever read once, not once per thread
    that happens to race the initial None check."""
    global _GHSL_POLYS_CACHE
    if _GHSL_POLYS_CACHE is None:
        with _GHSL_LOAD_LOCK:
            if _GHSL_POLYS_CACHE is None:
                try:
                    with fiona.open(UC_SHP) as src:
                        _GHSL_POLYS_CACHE = list(src)
                except Exception as e:
                    print(f"  [WARN] GHSL Urban Centre shapefile not available: {e}")
                    _GHSL_POLYS_CACHE = []
    return _GHSL_POLYS_CACHE


def _ghsl_candidate_points_moll(geom, spacing_m: float, cap: int,
                                 rng: random.Random) -> list[tuple[float, float]]:
    """Grid points (Mollweide meters) inside geom's real shape (shapely containment,
    not just bounding box), capped + randomly subsampled if the natural grid would
    exceed cap — keeps huge polygons from dominating."""
    minx, miny, maxx, maxy = geom.bounds
    xs, x = [], minx
    while x <= maxx:
        xs.append(x); x += spacing_m
    ys, y = [], miny
    while y <= maxy:
        ys.append(y); y += spacing_m

    inside = [(x, y) for x in xs for y in ys if geom.contains(Point(x, y))]
    if len(inside) > cap:
        inside = rng.sample(inside, cap)
    return inside


# ── Core search ───────────────────────────────────────────────────────────────

def find_anchors(sc: dict, importance: str, stratum_rng: random.Random,
                 n: int, already_found: list[dict]) -> list[dict]:
    """
    3-phase search — OSM discovery, then densify, then GHSL seeding — each checked
    against quota before starting. n can be set arbitrarily high (e.g. 250000): every
    phase naturally stops once real-world supply runs out, no error, no hang.
    """
    name = sc["name"]
    found = list(already_found)

    if len(found) >= n:
        print(f"  [SKIP] {name:<44} (quota met — {len(found)}/{n} cached)")
        return found

    print(f"  [{importance:>4}] {name}")

    # ── Phase 1: chunked OSM discovery ──────────────────────────────────────
    query = OSM_QUERIES.get(name)
    if query:
        chunks = split_query_into_chunks(query.strip())
        stratum_rng.shuffle(chunks)
        chunks_hit_cap = 0
        for i, chunk_query in enumerate(chunks):
            if len(found) >= n:
                break
            out_n_match = _OUT_N_RE.search(chunk_query)
            requested = int(out_n_match.group(1)) if out_n_match else None

            elements = _overpass_global(chunk_query)
            if requested is not None and len(elements) >= requested:
                chunks_hit_cap += 1
            # No per-chunk sleep here — _overpass_global() itself enforces RATE_DELAY
            # spacing globally now (shared across whichever strata are concurrently
            # fetching), so a local sleep here would just double-delay this thread.
            if not elements:
                continue

            stratum_rng.shuffle(elements)
            points = []
            for el in elements:
                lat, lon = _center(el)
                if lat is None or abs(lat) > 85:
                    continue
                points.append((lat, lon))

            passed = _verify_batch_parallel(sc, points)
            _append_results(sc, importance, found, passed, "osm_base", n)

        cap_note = (f", {chunks_hit_cap}/{len(chunks)} chunks hit their cap (more may exist)"
                    if chunks_hit_cap else "")
        print(f"    phase 1 (osm): {len(found)}/{n}{cap_note}")

    # ── Phase 2: densify (large-contiguous-zone strata only) ───────────────
    if len(found) < n and name in DENSIFY_STRATA:
        seed_coords = [(float(f["lat"]), float(f["lon"])) for f in found]
        BATCH = 300
        for start in range(0, len(seed_coords), BATCH):
            if len(found) >= n:
                break
            batch = seed_coords[start:start + BATCH]
            grid_points = []
            for lat, lon in batch:
                grid_points.extend(_grid_points(lat, lon, GRID_RADIUS, BBOX_M))
            passed = _verify_batch_parallel(sc, grid_points)
            _append_results(sc, importance, found, passed, "densify", n)
        print(f"    phase 2 (densify): {len(found)}/{n}")

    # ── Phase 3: GHSL degurba seeding (Dense Urban + Suburban only) ────────
    if len(found) < n and name in GHSL_SEED_STRATA:
        polys = list(_load_ghsl_polygons())
        regions = GHSL_SEED_REGIONS.get(name)
        if regions is not None:
            polys = [feat for feat in polys if _polygon_in_regions(feat, regions)]
        rng2 = random.Random(RNG_SEED + hash(name) % 10_000)
        rng2.shuffle(polys)
        BATCH_POLY = 200
        for start in range(0, len(polys), BATCH_POLY):
            if len(found) >= n:
                break
            batch = polys[start:start + BATCH_POLY]
            candidate_points = []
            for feat in batch:
                geom = shape(feat["geometry"])
                for x, y in _ghsl_candidate_points_moll(geom, BBOX_M, MAX_POINTS_PER_POLYGON, rng2):
                    lon, lat = _MOLL_TO_WGS84.transform(x, y)
                    candidate_points.append((lat, lon))
            passed = _verify_batch_parallel(sc, candidate_points)
            _append_results(sc, importance, found, passed, "ghsl_seed", n)
        print(f"    phase 3 (ghsl): {len(found)}/{n}")

    got = len(found)
    status = "✓" if got >= n else "~"
    print(f"  {status} {name}: {got}/{n} total\n")
    return found


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    esa_classes   = parse_esa_classes(FEATURES_HTML)
    imp_map       = parse_importance_tiers(BREAKDOWN_HTML)
    imp_lower     = {k.lower(): v for k, v in imp_map.items()}
    strata_config = build_strata_config(esa_classes)

    # Load already-found rows grouped by stratum
    existing: dict[str, list[dict]] = {}
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, newline="") as f:
            for row in csv.DictReader(f):
                existing.setdefault(row["stratum_name"], []).append(row)

    def _quota(sc: dict) -> int:
        return N_PER_STRATUM.get(sc["name"], 1)

    total_quota = sum(_quota(sc) for sc in strata_config)

    print(f"Anchor search — {len(strata_config)} strata  "
          f"target={total_quota} coordinates  "
          f"STRATUM_WORKERS={STRATUM_WORKERS} (concurrent strata; Overpass itself stays "
          f"globally rate-limited to 1-at-a-time regardless)  "
          f"(N_PER_STRATUM={N_PER_STRATUM})\n")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stratum_name", "importance_tier", "strata_type",
        "location_name", "lat", "lon",
        "primary_class", "primary_pct",
        "secondary_class", "secondary_pct",
        "bbox_m", "remark",
    ]
    write_lock = threading.Lock()
    with open(OUTPUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writeheader()

    def _write_rows(rows: list[dict]) -> None:
        # Incremental write as each stratum finishes — a multi-hour run that gets
        # interrupted keeps whatever finished so far instead of losing everything to
        # one final batch write at the end.
        if not rows:
            return
        with write_lock:
            with open(OUTPUT_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerows(rows)

    results_by_stratum: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=STRATUM_WORKERS) as pool:
        futures = {}
        for idx, sc in enumerate(strata_config):
            tier   = imp_lower.get(sc["name"].lower(), "UNKNOWN")
            quota  = _quota(sc)
            stratum_rng = random.Random(RNG_SEED + idx)
            cached = existing.get(sc["name"], [])
            future = pool.submit(find_anchors, sc, tier, stratum_rng, quota, cached)
            futures[future] = (sc, quota, len(cached))

        for future in as_completed(futures):
            sc, quota, n_cached = futures[future]
            found = future.result()
            results_by_stratum[sc["name"]] = found
            _write_rows(found[n_cached:])   # only the NEW rows this run found
            if len(found) < quota:
                print(f"  [WARN] {sc['name']}: only {len(found)}/{quota} found "
                      f"(real-world supply exhausted across all 3 phases)")

    total_found = sum(len(v) for v in results_by_stratum.values())
    print(f"\n[OUT] {total_found}/{total_quota} coordinates → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
