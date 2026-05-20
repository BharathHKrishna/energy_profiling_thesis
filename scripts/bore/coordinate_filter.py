#!/usr/bin/env python3
"""
coordinate_filter.py — Strata config and HTML parsers for real_anchor_finder.py.

Exports:
    FEATURES_HTML        Path to ESA WorldCover feature definitions
    BREAKDOWN_HTML       Path to importance-tier assignments
    parse_esa_classes()  {class_name: wc_code} from features.html
    parse_importance_tiers()  {stratum_name_lower: 'HIGH'|'MID'|'LOW'}
    build_strata_config()     16-stratum filter config list
"""

import re
from pathlib import Path

from bs4 import BeautifulSoup

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE           = Path(__file__).resolve().parents[2]
FEATURES_HTML  = BASE / "docs" / "features.html"
BREAKDOWN_HTML = BASE / "docs" / "2500_coordinate_breakdown.html"

# Full WC class list — HTML legend only shows 7 of 11 so supplement the rest
_WC_SUPPLEMENT = {"wetland": 90, "mangrove": 95, "snow": 70, "moss": 100}


# ── 1. Parse ESA WorldCover classes from features.html ───────────────────────
def parse_esa_classes(path: Path) -> dict[str, int]:
    """
    Returns {normalised_class_name: wc_code} from the ESA WorldCover table row.
    Supplements with known WC classes absent from the HTML legend.
    """
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    result: dict[str, int] = {}

    for td in soup.find_all("td"):
        note = td.find(class_="subnote")
        if note and "Full legend" in note.get_text():
            raw = re.search(r"Full legend[:\s]+([^\n<]+)", note.get_text())
            if not raw:
                continue
            _NAME_MAP = {
                "trees": "tree_cover", "shrubs": "shrubs",
                "grassland": "grassland", "cropland": "cropland",
                "built_up": "built_up", "bare": "bare_sparse", "water": "water",
            }
            for chunk in raw.group(1).split(","):
                m = re.match(r"\s*(\d+)=([A-Za-z][A-Za-z_\-]*)", chunk.strip())
                if m:
                    code  = int(m.group(1))
                    label = m.group(2).lower().replace("-", "_")
                    result[_NAME_MAP.get(label, label)] = code
            break

    for name, code in _WC_SUPPLEMENT.items():
        result.setdefault(name, code)

    return result


# ── 2. Parse importance tiers from 2500_coordinate_breakdown.html ────────────
def parse_importance_tiers(path: Path) -> dict[str, str]:
    """
    Returns {stratum_name_lower: 'HIGH'|'MID'|'LOW'}.
    Reads bar-lbl (union) and pure-lbl (pure) elements within each section.
    """
    soup   = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    tiers: dict[str, str] = {}
    for section in soup.find_all(class_="section"):
        title = (section.find(class_="sec-title") or section).get_text().lower()
        tier  = "HIGH" if "high" in title else ("MID" if "mid" in title else "LOW")
        for css_class in ("bar-lbl", "pure-lbl"):
            for el in section.find_all(class_=css_class):
                tiers[el.get_text(strip=True).lower()] = tier
    return tiers


# ── 3. Strata filter config ───────────────────────────────────────────────────
def build_strata_config(esa: dict[str, int]) -> list[dict]:
    """
    Returns the 16-stratum config list. All ESA class keys reference the dict
    returned by parse_esa_classes — no class names are hardcoded in filter logic.

    Keys per entry:
        name           stratum label (must match breakdown HTML bar-lbl / pure-lbl)
        kind           'union' | 'pure'
        primary        ESA class key (primary_min threshold applied)
        primary_min    minimum % for primary class
        secondary      ESA class key for union; None for pure
        secondary_min  minimum % for secondary class; None for pure
        pop_lo/pop_hi  GHSL raw cell population bounds (None = no bound)
                       NOTE: GHSL POP layer returns count per 100m cell, not per km².
                       Multiply by 100 for approximate /km². E.g. pop_hi=300 ≈ 30,000/km².
        h_lo/h_hi      GHSL mean building height bounds in metres (None = no bound)
        bsurf_lo/hi    GHSL built surface m² per 100m cell (None = no bound)
                       Reliable global proxy for built density: ~100 (sparse) → ~5000 (dense city)
    """
    for k in ("built_up", "bare_sparse", "water", "cropland", "tree_cover", "mangrove"):
        if k not in esa:
            raise ValueError(f"ESA key '{k}' missing from features.html — check parsing")

    bu, bs, wt, cr, tc, mn, gr = (
        "built_up", "bare_sparse", "water", "cropland", "tree_cover", "mangrove", "grassland"
    )

    def u(name, pri, pri_min, sec, sec_min,
          tier_key=None,
          primary_hi=None,
          pop_lo=None, pop_hi=None, h_lo=None, h_hi=None,
          bsurf_lo=None, bsurf_hi=None,
          requires_ocean=False):
        return dict(name=name, kind="union", tier_key=tier_key,
                    primary=pri,  primary_min=pri_min, primary_hi=primary_hi,
                    secondary=sec, secondary_min=sec_min,
                    pop_lo=pop_lo, pop_hi=pop_hi,
                    h_lo=h_lo, h_hi=h_hi,
                    bsurf_lo=bsurf_lo, bsurf_hi=bsurf_hi,
                    requires_ocean=requires_ocean)

    def p(name, pri, pri_min,
          tier_key=None,
          primary_hi=None,
          pop_lo=None, pop_hi=None, h_lo=None, h_hi=None,
          bsurf_lo=None, bsurf_hi=None):
        return dict(name=name, kind="pure", tier_key=tier_key,
                    primary=pri,  primary_min=pri_min, primary_hi=primary_hi,
                    secondary=None, secondary_min=None,
                    pop_lo=pop_lo, pop_hi=pop_hi,
                    h_lo=h_lo, h_hi=h_hi,
                    bsurf_lo=bsurf_lo, bsurf_hi=bsurf_hi,
                    requires_ocean=False)

    return [
        # ── HIGH union (3) — 250 each ────────────────────────────────────────
        # water ≥ 20% (river/canal visible) + built_up ≥ 35% (solid industrial footprint)
        # bsurf_lo=300: any real industrial facility has ≥300m² built footprint per 100m cell
        u("Industrial + Water",
          wt, 20, bu, 35,
          tier_key="HIGH_union",
          bsurf_lo=300),

        # water ≥ 12% (waterway visible in 512m bbox — a 60m-wide channel suffices)
        # built_up ≥ 25%: real port infrastructure present.
        # bsurf_lo=500: port quays, warehouses, cranes → always ≥500m² per 100m cell.
        u("Urban + Coastal",
          wt, 12, bu, 25,
          tier_key="HIGH_union",
          bsurf_lo=500,
          requires_ocean=True),

        # built_up ≥ 40%: dense urban fabric confirmed.
        # h_hi = 7m: the universal informal discriminator — informal cities are low-rise
        # (single/double story, avg GHSL ≤ 7m across 512m bbox). Formal Dense Urban averages
        # >7m (Paris 8.4m, Cairo 8.7m, Beirut 9.4m). African informal cities: Kampala 2.7m,
        # Ouagadougou 3.3m, Conakry 3.7m, Adama 2.1m — all well below 7m.
        p("Informal + Urban",
          bu, 40,
          tier_key="HIGH_union",
          h_hi=7),

        # ── HIGH pure (3) — 300 each ─────────────────────────────────────────
        # Dense Urban: bsurf_lo=2500 confirms dense built fabric (Paris=4131, Cairo=3588).
        # h_lo=3: multi-story character (avg ≥3m rules out single-story sprawl).
        # pop_lo=20: excludes Phoenix-style commercial strips (pop=2) and pure industrial
        # zones that pass built_up+bsurf gates but have zero residential population.
        p("Dense Urban",
          bu, 80,
          tier_key="HIGH_pure",
          pop_lo=20, h_lo=3, bsurf_lo=2500),

        # Suburban: built_up ≥ 50%, capped at 85% so anchor doesn't blur with Dense Urban.
        # primary_hi=85: excludes compact town cores that look identical to city centres.
        # pop_hi=500 raw (≈50,000/km²) prevents city-centre hits.
        # h_hi=15: Phoenix-style strip malls average 13m GHSL despite being suburban.
        p("Suburban",
          bu, 50,
          tier_key="HIGH_pure",
          primary_hi=85, pop_hi=500, h_hi=15),

        # Industrial: pop_hi=300 (≈30,000/km²) — factories are uninhabited.
        # bsurf_lo=500: ensures a real industrial campus (not just a rural field with 1 shed).
        p("Industrial",
          bu, 75,
          tier_key="HIGH_pure",
          pop_hi=300, bsurf_lo=500),

        # ── MID (4) — 150 each ───────────────────────────────────────────────
        # built_up ≥ 40%: large building campus confirmed.
        # OSM query (telecom=data_center, building=data_center) guarantees DC context.
        # pop_hi=1500 keeps out dense residential city centres.
        # bsurf_lo=500: large building footprint (DC server halls are wide, flat structures).
        p("Data Centre + Industrial",
          bu, 40,
          tier_key="MID",
          pop_hi=1500, bsurf_lo=500),

        # bare_sparse ≥ 40% (arid landscape visually dominant) + built_up ≥ 1% (infrastructure)
        # Raised from 25% → 40%: 25% let UAE industrial parks pass (bs=29%, bu=71%) where the
        # frame looks like a city, not a desert. 40% guarantees the landscape reads as arid.
        # 1% of 512m bbox ≈ 51×51m — a well pad or flare stack is clearly visible at zoom 15.
        # pop_hi lowered 300→100: Libyan towns (pop≈300 raw) passed despite looking urban.
        #   100 raw ≈ 10,000/km² — true petroleum/mining sites are uninhabited or near-zero.
        # bsurf_lo=200: rules out bare desert with only a dirt road (which can pass bu≥1%
        #   via road pixels). A real facility (wellhead, refinery, mine) has ≥200m² footprint.
        u("Industrial + Arid",
          bs, 40, bu, 1,
          tier_key="MID",
          pop_hi=100, bsurf_lo=200),

        # cropland ≥ 25% (actual farmland clearly present) + built_up ≥ 15% (solar panels).
        # Raised cropland from 20%→25%: 20% admitted fish-pond floating-solar where "cropland"
        # was miscoded adjacent paddy — 25% ensures genuine farmland is prominent in frame.
        u("Agrivoltaics (Solar + Farmland)",
          cr, 25, bu, 15,
          tier_key="MID"),

        # tree_cover ≥ 20% (forest visually present) + built_up ≥ 5% (sawmill / timber facility).
        # Raised tree_cover from 15% → 20%: 15% let suburban sawmills pass (tc=16%, bu=49%)
        # where the frame looks like a city with some trees, not forest. 20% ensures forest
        # is clearly visible alongside the sawmill buildings.
        # pop_hi=300: sawmill areas are sparsely populated.
        u("Industrial + Forest",
          tc, 20, bu, 5,
          tier_key="MID",
          pop_hi=300),

        # water ≥ 20% (dam wall + reservoir tail both visible in 512m bbox).
        # built_up ≥ 3%: dam concrete structure guaranteed in frame — rejects pure-reservoir
        # bboxes where dam wall is out of frame and only open water is visible (audit: 3/25
        # had water=64-100% with dam invisible). ~26m of dam wall = 3% of 512m bbox.
        # pop_hi=50: dam zones are uninhabited — discriminates from port/coastal water strata.
        u("Hydropower Reservoir",
          wt, 20, bu, 3,
          tier_key="MID",
          primary_hi=60,
          pop_hi=50),

        # bare_sparse ≥ 25% (desert/scrubland setting) + built_up ≥ 15% (panel arrays).
        # Key differentiator from Industrial+Arid: that stratum requires only built_up≥1%.
        # Utility solar farms cover 15-40% of the bbox with panels (ESA classifies as built_up).
        # pop_hi=100: solar farms are uninhabited.
        u("Utility-scale Solar Farm",
          bs, 25, bu, 15,
          tier_key="MID",
          pop_hi=100),

        # built_up ≥ 40% (terminals, aprons, tarmac) + grassland ≥ 8%
        # OSM entry is now way[aeroway=aerodrome]["iata"] — polygon centroid sits at airfield
        # center, capturing runway tarmac + grass safety strips in the same 512m bbox.
        # Lowered from ≥12% to ≥8%: grass strips between runways read 8-15% at this scale.
        # pop_hi=300: airports adjacent to dense cities (Changi, HK, CDG) still pass.
        # bsurf_lo=200: confirms large terminal/hangar footprint present.
        u("Airport / Aviation",
          bu, 40, gr, 8,
          tier_key="MID",
          pop_hi=300, bsurf_lo=200),

        # ── LOW (5) — 30 each ────────────────────────────────────────────────
        # cropland ≥ 25% (irrigated fields) + water ≥ 12% (water channel clearly visible)
        # Raised from 8% → 12%: 8% let barely-visible irrigation channels pass (water just a
        # thin strip). 12% of 512m bbox ≈ 31,000 m² — a canal or river section you can see.
        u("Agricultural + Water",
          cr, 25, wt, 12,
          tier_key="LOW"),

        # cropland ≥ 20% (sea-edge coastal paddies) + water ≥ 20% (open sea visibly present)
        # Raised from 15% → 20%: 15% let estuarine/mangrove deltas pass (no clear open sea).
        u("Coastal + Agricultural",
          cr, 20, wt, 20,
          tier_key="LOW",
          requires_ocean=True),

        # mangrove ≥ 15% (coastal fringe clearly visible) + built_up ≥ 8% (port/settlement).
        # Raised from 8%→15%: 8% admitted 3/25 candidates where mangrove was barely a fringe
        # strip — at 15% the dark-green canopy is unambiguously visible in the 512m frame.
        # pop_hi=400: mangrove-adjacent industrial sites are low-density.
        u("Mangrove + Industrial",
          mn, 15, bu, 8,
          tier_key="LOW",
          pop_hi=400),

        # built_up ≥ 15% (settlement present) + cropland ≥ 10% (orchard/fields at edge)
        # bsurf_lo=100: ensures actual built structures (not just a barn in a field).
        # pop_hi=2000: keeps out dense urban cores.
        u("Suburban + Agricultural",
          bu, 15, cr, 10,
          tier_key="LOW",
          pop_hi=2000, bsurf_lo=100),

        # water ≥ 20%: coastal character confirmed — ocean visible in bbox.
        # grassland ≥ 3%: coastal meadow/dune/salt-marsh land guaranteed in frame — rejects
        # pure-offshore bboxes with water=100% where no coastline is visible (audit: 23/25
        # had water=61-100%). European tidal-flat turbines sit at shoreline; ESA codes
        # beach/dune/saltmarsh as grassland(30). 3% ≈ one 87m strip of coastal land.
        # China Jiangsu/Bohai bboxes removed (offshore turbines in open water).
        u("Coastal + Solar-Wind Hybrid",
          wt, 20, gr, 3,
          tier_key="LOW",
          requires_ocean=True),
    ]
