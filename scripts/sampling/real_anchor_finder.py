#!/usr/bin/env python3
"""
real_anchor_finder.py — Find N verified coordinates per stratum using OSM + ESA + GHSL.

Methodology
-----------
  For each stratum:
  1. Query Overpass API globally using the stratum's primary OSM entry tag.
  2. Shuffle candidates with a per-stratum seed (reproducible).
  3. For each candidate in the full pool:
       a. Compute 512m × 512m bbox around the OSM element centre.
       b. Run ESA WorldCover extractor → check primary + secondary % thresholds.
       c. If GHSL bounds are set for this stratum → run GHSL extractor → check.
       d. Skip if within MIN_DIST_M of an already-accepted coordinate.
       e. Collect until N_PER_TIER[tier] coordinates are found.
  4. Write all results to OUTPUT_CSV.

Scale
-----
  To change the number of coordinates per stratum, edit N_PER_TIER below.
  Everything else adapts automatically.

Usage
-----
    python scripts/sampling/real_anchor_finder.py
"""

import csv
import json
import math
import random
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

import requests as _requests

from scripts.extractors.worldcover_extractor import extract_worldcover_features
from scripts.extractors.ghsl_extractor       import extract_ghsl_features
from scripts.sampling.coordinate_filter      import (
    parse_esa_classes, parse_importance_tiers, build_strata_config,
    FEATURES_HTML, BREAKDOWN_HTML,
)

OUTPUT_CSV       = BASE / "outputs" / "csv" / "filtered_strata_sample.csv"
OVERPASS_URL     = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 120
BBOX_M           = 512
MIN_DIST_M       = 512   # minimum separation between accepted coordinates (dedup)
RNG_SEED         = 42
RATE_DELAY       = 2.0   # seconds between Overpass requests

# ── Scale control ─────────────────────────────────────────────────────────────
# Change these to scale BORE output. Currently 1 per stratum (anchor validation).
# For the full 2500 run: {"HIGH_union": 250, "HIGH_pure": 300, "MID": 100, "LOW": 30}
# 18 strata: 3×250 + 3×300 + 7×100 + 5×30 = 2500
N_PER_TIER: dict[str, int] = {
    "HIGH_union": 1,
    "HIGH_pure":  1,
    "MID":        1,
    "LOW":        1,
}


# ── Per-stratum OSM entry queries ─────────────────────────────────────────────
# OSM is the entry point — it narrows the Earth to plausible locations.
# ESA + GHSL then verify the actual land cover / urban character.
# out counts are set high so a single query can supply N > 1 results.

OSM_QUERIES: dict[str, str] = {
    # Docks are water basins by definition — water guaranteed in bbox.
    # Cranes sit on the quay at the water's edge — bbox captures both built_up + water.
    # Both tags imply the wt≥20% + bu≥35% co-presence that wastewater_plant could not guarantee.
    "Industrial + Water": """
        [out:json][timeout:60];
        (way[waterway=dock];
         node[man_made=crane];
         node[man_made=petroleum_refinery];
         way[man_made=petroleum_refinery];);
        out center 1000;""",

    "Urban + Coastal": """
        [out:json][timeout:30];
        node[harbour=yes];
        out 1000;""",

    "Informal + Urban": """
        [out:json][timeout:25];
        node[place=city](-5,10,15,45);
        out 500;""",

    "Dense Urban": """
        [out:json][timeout:30];
        node[capital=yes];
        out 500;""",

    # Global small towns — targeted bboxes to avoid server timeout from querying large regions.
    # Africa/W.Africa (4,3,12,15): Nigeria/Cameroon — dense OSM town coverage, low-rise fabric.
    # SE Asia (10,100,20,110): Vietnam/Thailand — rapidly growing peri-urban areas.
    # Eastern Europe (46,20,52,32): Romania/Ukraine — post-socialist suburban towns.
    # South America (-23,-50,-15,-40): São Paulo suburban ring — well-mapped peri-urban.
    # ESA gate (built_up≥50%, pop≤500, h≤15m) selects low-rise peri-urban fabric globally.
    # Africa (4,3,12,15) removed: ESA over-classifies small African settlements as 99%+ built_up
    # → bbox lands on forest-settlement fringe, doesn't look suburban. European/SE Asian/Brazilian
    # towns have genuine suburban character (houses, roads, gardens in mixed proportion).
    "Suburban": """
        [out:json][timeout:45];
        (node[place=town](10,100,20,110);
         node[place=town](46,20,52,32);
         node[place=town](-23,-50,-15,-40););
        out 1000;""",

    # Cooling towers (nodes + ways) + coal/gas power plants — all have dense industrial ESA signature
    "Industrial": """
        [out:json][timeout:60];
        (node[man_made=cooling_tower];
         way[man_made=cooling_tower];
         way[power=plant]["plant:source"=coal];
         way[power=plant]["plant:source"=gas];
         node[power=plant]["plant:source"=coal];
         node[power=plant]["plant:source"=gas];
         way[landuse=industrial][industrial=manufacturing];);
        out center 1500;""",

    # Bboxed to established hyperscale DC corridors — avoids small ISP offices mislabelled as
    # data_center in OSM (e.g. residential Barquisimeto, Venezuela that passed globally).
    # US Virginia corridor (38,-78,40,-76): AWS/Azure/Google campus clusters.
    # US Dallas (32,-97,33,-96) + Chicago (41,-88,42,-87): major Equinix/CyrusOne hubs.
    # Amsterdam (52,4.5,53,5.5): AMS-IX, Digital Realty, Equinix.
    # Frankfurt (50,8,51,9): DE-CIX, NTT, Equinix.
    # Singapore (1,103,2,104): Asia-Pacific hub.
    # London (51,-1,52,1): Slough/Docklands corridor.
    "Data Centre + Industrial": """
        [out:json][timeout:60];
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
        out center 1000;""",

    # Deep-desert bboxes: petroleum wells/flares land in open desert (bare_sparse high),
    # but individual well nodes often have zero built_up within 512m → fail the built_up≥1% gate.
    # Fix: add way[man_made=works] (refineries/processing plants) and way[landuse=industrial]
    # which have real building footprints, guaranteeing built_up in bbox.
    # Added Australian Pilbara (-26,117,-20,121): iron ore mines in zero-rainfall desert —
    # massive industrial footprint + pure arid landscape = ideal arid+industrial signature.
    # Rule: all bboxed entries = SAFE.
    "Industrial + Arid": """
        [out:json][timeout:60];
        (node[man_made=petroleum_well](17,44,24,56);
         node[man_made=petroleum_well](26,8,30,18);
         node[man_made=petroleum_well](25,28,29,33);
         node[man_made=gas_well](17,44,24,56);
         node[man_made=gas_well](26,8,30,18);
         node[man_made=flare](17,44,24,56);
         node[man_made=flare](26,8,30,18);
         way[man_made=works](17,44,24,56);
         way[man_made=works](26,8,30,18);
         way[man_made=works](25,28,29,33);
         way[landuse=industrial](17,44,24,56);
         way[landuse=industrial](26,8,30,18);
         way[landuse=industrial](-26,117,-20,121);
         way[man_made=works](-26,117,-20,121);
         way[landuse=quarry](-26,117,-20,121);
         way[power=generator]["generator:source"=solar](17,44,24,56);
         way[power=generator]["generator:source"=solar](26,8,30,18);
         way[power=generator]["generator:source"=solar](-30,-70,-22,-65);
         way[power=generator]["generator:source"=solar](-30,115,-22,125););
        out center 4000;""",

    # Onshore coastal bboxes — turbines at the land-sea boundary.
    # ESA water≥15% + bare_sparse≥5% gates select installations with visible tidal flat / beach.
    # German/Danish North Sea (53,5,57,12): highest density — turbines in tidal flats.
    # Netherlands/Belgium North Sea (51,2.5,53,5): flat coast, turbines at sea wall.
    # Swedish Kattegat (56,10,59,13): turbines right on shallow coast.
    # Norwegian coast (58,4,62,8): fjord-edge coastal wind.
    # UK west/north coast (50,-5,58,2): many coastal sites.
    # Portugal/Spain Atlantic (36,-9,42,-7): densely mapped coastal wind corridor.
    # Japan Sea coast (34,130,38,135): dense coastal turbine mapping.
    # South Korea coast (34,126,37,131): coastal wind well mapped.
    # Tehuantepec Mexico (14,-96,18,-92): densest onshore wind zone in Americas.
    # South Australia (-36,136,-34,141): Great Australian Bight coastal wind.
    # China Jiangsu (31,120,34,122) + Bohai (38,117,40,122) REMOVED: offshore turbines in open
    # tidal water → water=100%, no bare_sparse tidal flat visible, no coastal character.
    # Rule: bboxed ways + bboxed nodes = SAFE.
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

    # Removed Anhui/Jiangsu China (fish pond floating solar → water not cropland).
    # Added Henan/Shandong China (35,113,38,120): Yellow River wheat plains — dryland wheat
    # with ground-mounted solar deployed over active fields, no fish pond interference.
    # pass_rate≈3.8% → need ~2632 evaluated for 100 targets → out center 5000.
    #   Gujarat India (22,71,26,75)          — solar-irrigation co-location, wheat/cotton
    #   Punjab Pakistan (28,68,32,74)        — solar + wheat/rice
    #   Nile Delta Egypt (30,30,32,32)       — solar + cotton/rice
    #   Henan/Shandong China (35,113,38,120) — solar + dryland wheat (Yellow River plains)
    #   Saxony Germany (50,12,52,14)         — solar on cropland (EU power=plant nodes)
    #   Nebraska/Iowa USA (40,-100,45,-92)   — solar + corn/soy
    #   Extremadura Spain (37,-7,40,-4)      — solar + wheat/olive, dense mapping
    #   Nouvelle-Aquitaine France (43,0,47,3)— solar + sunflower/maize, well mapped
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
         way[power=plant]["plant:source"=solar](22,71,26,75);
         way[power=plant]["plant:source"=solar](28,68,32,74);
         way[power=plant]["plant:source"=solar](30,30,32,32);
         way[power=plant]["plant:source"=solar](35,113,38,120);
         way[power=plant]["plant:source"=solar](37,-7,40,-4);
         way[power=plant]["plant:source"=solar](43,0,47,3);
         node[power=plant]["plant:source"=solar](50,12,52,14);
         node[power=plant]["plant:source"=solar](40,-100,45,-92);
         node[power=plant]["plant:source"=solar](37,-7,40,-4););
        out center 5000;""",

    # Sawmills and timber processing facilities embedded in/adjacent to boreal/temperate forest.
    # man_made=sawmill (nodes + ways): primary OSM tag for timber cutting facilities.
    # industrial=sawmill under landuse=industrial: large mill complexes with rail/road sidings.
    # Bboxed to forested regions only — prevents urban sawmills in city centres.
    # Russia/Siberia, Scandinavia, Canada BC, US Pacific NW, Carpathians, New Zealand.
    # Rule: bboxed ways + bboxed nodes = SAFE.
    "Industrial + Forest": """
        [out:json][timeout:60];
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
         way[industrial=sawmill](48,-130,58,-115););
        out center 1000;""",

    # Dam way/node approach: centroid sits on the dam WALL — 512m bbox captures concrete wall
    # + terrain + reservoir tail. Mixed cover (water≥20%) passes the ESA gate cleanly.
    # Reservoir polygon approach was discarded: centroid in open water → pure-water bbox,
    # dam structure invisible. Dam wall centroid is the energy infrastructure we want to show.
    # Bboxes: China Yangtze/Mekong, Brazil Paraná/Itaipu, Norway fjords, India Deccan,
    # US Pacific NW, Turkey, Zambia/Zimbabwe Kariba, Egypt Aswan, South Africa.
    # Rule: bboxed ways + bboxed nodes = SAFE.
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
         node[waterway=dam](22,30,24,34););
        out center 2000;""",

    # Utility-scale solar in desert/scrubland regions only — NOT agricultural (that is Agrivoltaics).
    # Morocco Ouarzazate (29,5,35,12), India Rajasthan (23,68,28,76),
    # USA Southwest (32,-118,37,-108), Chile Atacama (-26,-72,-20,-66),
    # China Gansu/Tengger (37,97,40,107), UAE/Saudi (22,46,26,56).
    # Rule: multiple bboxed ways = SAFE.
    "Utility-scale Solar Farm": """
        [out:json][timeout:60];
        (way[power=plant]["plant:source"=solar](29,5,35,12);
         way[power=plant]["plant:source"=solar](23,68,28,76);
         way[power=plant]["plant:source"=solar](32,-118,37,-108);
         way[power=plant]["plant:source"=solar](-26,-72,-20,-66);
         way[power=plant]["plant:source"=solar](37,97,40,107);
         way[power=plant]["plant:source"=solar](22,46,26,56);
         way[power=generator]["generator:source"=solar](23,68,28,76);
         way[power=generator]["generator:source"=solar](37,97,40,107);
         way[power=generator]["generator:source"=solar](29,5,35,12););
        out center 1000;""",

    # way[aeroway=aerodrome]["iata"]: airport boundary POLYGON — centroid lands at the airfield
    # center, not on the terminal building. 512m bbox captures runway tarmac + grass safety
    # strips in the same frame, so built_up≥40% + grassland≥8% both pass cleanly.
    # Old approach (terminal ways + aerodrome nodes) put the bbox on the terminal building
    # where grassland strips are far away → 3.7% pass rate. Polygon centroid → 14.7%.
    # pass_rate≈14.7% → need ~680 evaluated for 100 targets → out center 2500.
    "Airport / Aviation": """
        [out:json][timeout:60];
        way[aeroway=aerodrome]["iata"];
        out center 2500;""",

    # Nodes at the structural water-cropland boundary: weir/floodgate/sluice_gate/pumping_station.
    # Added lower Mekong delta Vietnam/Cambodia (9,104,12,107): dense rice irrigation
    # infrastructure — pumping stations and sluice gates right at cropland-canal interface.
    "Agricultural + Water": """
        [out:json][timeout:60];
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
         node[waterway=sluice_gate](9,104,12,107););
        out 1500;""",

    # way[natural=coastline] in delta-FREE ocean-facing rice coasts ONLY.
    # Myanmar (93°E) and Bangladesh Ganges mouth REMOVED — ESA water there = tidal channels
    # and river distributaries (looks like ponds), not open sea.
    # All retained bboxes are direct ocean-facing coasts with known rice cultivation:
    #   Philippines Ilocos/Luzon (17,120,19,121): Pacific Ocean + rice terraces
    #   Philippines Visayas (10,123,12,125): Visayan Sea + coastal rice
    #   Vietnam central (16,107,18,109): South China Sea + flat coastal rice plains
    #   Java north (−7,108,−5,112): Java Sea + densest SE Asian rice
    #   Gulf of Thailand (12,99.5,14,101): Thai Gulf coast + rice belt
    #   Ca Mau Vietnam (8.5,104.5,9.5,105.3): South China Sea peninsula + rice
    #   Sri Lanka east (8,81,10,82): Indian Ocean + paddy rice
    #   Odisha India (19,84,21,87): Bay of Bengal + Mahanadi coastal rice
    #   Taiwan west (23,120,25,121): Taiwan Strait + rice plains
    #   South Korea Honam (34,126,36,127): Yellow Sea + coastal rice paddies
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
    # way[wetland=mangrove] gives 3.5% pass (3.5× better than landuse=industrial approach).
    # The fix is VOLUME: out center 3000 gives ~105 passing @ 3.5% → N=25 ✓, N=100 ✓.
    # Targeted sub-bboxes for known mangrove+industrial co-location zones:
    #   Bonny Island Nigeria (4,6.5,5,7.5)  — Nigeria LNG plant surrounded by Niger Delta mangrove
    #   Warri Delta (4.5,5,6,6.5)           — Shell facilities in Niger Delta mangrove
    #   Sundarbans Bangladesh (22,89,23,90)  — Mongla industrial port at Sundarbans edge
    #   Tarakan Kalimantan (2.8,117,4,118.5) — oil port in Borneo mangrove coast
    # Broad SE Asian bboxes provide the volume; targeted bboxes concentrate quality.
    "Mangrove + Industrial": """
        [out:json][timeout:90];
        (way[wetland=mangrove](4,6.5,5,7.5);
         way[wetland=mangrove](4.5,5,6,6.5);
         way[wetland=mangrove](22,89,23,90);
         way[wetland=mangrove](2.8,117,4,118.5);
         way[wetland=mangrove](0,108,6,119);
         way[wetland=mangrove](8,99,12,107););
        out center 1500;""",

    # Allotment ways + orchards/vineyards in peri-urban rings — always at city-cropland edge.
    # Added Iberian Peninsula (38,-10,42,-6): Lisbon/Madrid suburban fringe meets vineyard belts.
    # Punjab India, Po Valley Italy, Nile fringe, Rhine Valley Germany, Yangtze Delta unchanged.
    "Suburban + Agricultural": """
        [out:json][timeout:60];
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
         node[landuse=vineyard](38,-10,42,-6););
        out center 1000;""",

}


_OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
_OVERPASS_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": "thesis-anchor-finder/1.0 (energy_profiling research; contact bharathhk18.bk@gmail.com)",
    "Accept": "application/json",
}


def _overpass_global(query: str) -> list:
    """Direct Overpass request — tries each mirror, with one retry on 429/504."""
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


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    """True if (lat, lon) is within MIN_DIST_M of any already-accepted coordinate."""
    for r in found:
        if _haversine_m(lat, lon, float(r["lat"]), float(r["lon"])) < MIN_DIST_M:
            return True
    return False


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


# ── Core search ───────────────────────────────────────────────────────────────

def find_anchors(sc: dict, importance: str, stratum_rng: random.Random,
                 n: int, already_found: list[dict]) -> list[dict]:
    """
    Query OSM globally → shuffle → verify with ESA + GHSL → collect up to n
    passing coordinates that are ≥ MIN_DIST_M apart from each other and from
    already_found (existing rows loaded from a previous run).

    Returns a list of result dicts (may be shorter than n if pool is exhausted).
    """
    query = OSM_QUERIES.get(sc["name"])
    if not query:
        print(f"  [SKIP] {sc['name']} — no OSM query defined")
        return []

    need = n - len(already_found)
    if need <= 0:
        print(f"  [SKIP] {sc['name']:<44} (quota met — {len(already_found)}/{n} cached)")
        return list(already_found)

    print(f"  [{importance:>4}] {sc['name']:<44}", end=" ", flush=True)

    elements = _overpass_global(query.strip())
    if not elements:
        print("✗ no OSM results")
        return list(already_found)

    stratum_rng.shuffle(elements)
    print(f"{len(elements):>4} candidates →", end=" ", flush=True)

    needs_ghsl = any(sc.get(k) is not None
                     for k in ("pop_lo", "pop_hi", "h_lo", "h_hi", "bsurf_lo", "bsurf_hi"))

    found: list[dict] = list(already_found)   # start with any cached results
    tried = 0

    for el in elements:
        if len(found) >= n:
            break

        lat, lon = _center(el)
        if lat is None or abs(lat) > 85:
            continue
        tried += 1

        bb = _bbox(lat, lon)

        wc = extract_worldcover_features(lat, lon,
                                         bb["min_lat"], bb["max_lat"],
                                         bb["min_lon"], bb["max_lon"])
        if wc is None:
            continue
        pcts = _wc_pcts(wc)
        if not _esa_ok(pcts, sc):
            continue

        if needs_ghsl:
            ghsl = extract_ghsl_features(lat, lon,
                                         bb["min_lat"], bb["max_lat"],
                                         bb["min_lon"], bb["max_lon"])
            if not _ghsl_ok(ghsl, sc):
                continue
        else:
            ghsl = None

        if _too_close(lat, lon, found):
            continue

        found.append(_build_result(sc, importance, el, pcts, ghsl, tried))

    got  = len(found)
    new  = got - len(already_found)
    p_pct = found[-1]["primary_pct"] if found else 0
    status = "✓" if got >= n else "~"
    print(f"{status} {got}/{n}  (verified {tried} candidates)")
    return found


def _build_result(sc, importance, el, pcts, ghsl, tries) -> dict:
    lat, lon = _center(el)
    name     = (el.get("tags") or {}).get("name") or f"{lat:.4f},{lon:.4f}"
    p_pct    = round(pcts.get(sc["primary"], 0.0), 1)
    s_pct    = round(pcts.get(sc["secondary"], 0.0), 1) if sc["secondary"] else ""
    return {
        "stratum_name":    sc["name"],
        "importance_tier": importance,
        "strata_type":     sc["kind"],
        "location_name":   name,
        "lat":             round(lat, 6),
        "lon":             round(lon, 6),
        "primary_class":   sc["primary"],
        "primary_pct":     p_pct,
        "secondary_class": sc["secondary"] or "",
        "secondary_pct":   s_pct,
        "bbox_m":          BBOX_M,
        "remark":          f"osm_verified tries={tries}",
    }


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
        key = sc.get("tier_key") or imp_lower.get(sc["name"].lower(), "LOW")
        return N_PER_TIER.get(key, 1)

    total_quota = sum(_quota(sc) for sc in strata_config)

    print(f"Anchor search — {len(strata_config)} strata  "
          f"target={total_quota} coordinates  "
          f"(N_PER_TIER={N_PER_TIER})\n")

    results = []
    for idx, sc in enumerate(strata_config):
        tier  = imp_lower.get(sc["name"].lower(), "UNKNOWN")
        quota = _quota(sc)
        stratum_rng   = random.Random(RNG_SEED + idx)
        cached        = existing.get(sc["name"], [])
        found         = find_anchors(sc, tier, stratum_rng, quota, cached)
        results.extend(found)
        if len(found) < quota:
            print(f"  [WARN] {sc['name']}: only {len(found)}/{quota} found")
        time.sleep(RATE_DELAY)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stratum_name", "importance_tier", "strata_type",
        "location_name", "lat", "lon",
        "primary_class", "primary_pct",
        "secondary_class", "secondary_pct",
        "bbox_m", "remark",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[OUT] {len(results)}/{total_quota} coordinates → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
