"""
PORE Detection Map Generator

Renders a single combined detection map for a coordinate overlaying:
  - Microsoft ML Building Footprints (cyan, transparent)
  - OSM Buildings (cyan, transparent)
  - OSM Energy Infrastructure (per-type colour, transparent)

Usage (standalone — no pipeline needed):
    python scripts/pore/detection_map.py                        # all 18 anchors
    python scripts/pore/detection_map.py --lat 52.52 --lon 13.405 --name test
"""
import sys, os, argparse
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# OSM elements come from segmap_generator.fetch_osm_elements (offline, shared).
# GHSL sampling (_sample_pop_grid/_sample_epoch_grid) and colormaps also live in
# segmap_generator.py — imported here rather than duplicated, since render_ghsl_det
# below needs the exact same population/age data as render_ghsl_segmap does.
from scripts.extractors.msft_buildings_extractor import fetch_msft_buildings
from scripts.utils.naming import slug_name
from scripts.pore.segmap_generator import (
    fetch_esri_img, _bbox,
    extract_osm_buildings, extract_osm_infra,
    _remove_shells, _filter_min_area,
    _sample_pop_grid, _sample_epoch_grid,
    CMAP_POP, POP_MIN, POP_MAX, NORM_POP,
    CMAP_AGE, EPOCH_MIN, EPOCH_MAX,
    OSM_CLR,
    ALL_COORDS,
)
from scripts.utils.logger import get_logger

logger = get_logger("detection_map")

BASE       = "/srv/THESIS/energy_profiling_thesis"
OUTPUT_DIR = os.path.join(BASE, "outputs/maps/detection")
BG         = "#0f172a"
CLR_BLDG   = "#00e5ff"   # cyan — MS + OSM buildings

# Contrasting infra colours for the GHSL-coloured variants — visible against both
# red (pop) and dark-red-to-white (age) building fills, unlike the standard OSM_CLR
# palette used by render_detection_panels() above.
INFRA_LINE_CLR  = "#00ffff"   # cyan for lines (power, pipeline)
INFRA_POLY_CLR  = "#00ff88"   # bright green for polygons (substation etc.)
INFRA_LINE_KEYS = {"power_line", "pipeline", "aeroway_line"}


# ── Panel renderers ────────────────────────────────────────────────────────────

def _draw_buildings(ax, rings, colour):
    patches = []
    for ring in rings:
        try:
            arr = np.array(ring)
            if arr.shape[1] == 2:
                patches.append(MplPolygon(arr))
        except Exception:
            continue
    if patches:
        pc = PatchCollection(patches, facecolor=colour, edgecolor=colour,
                             linewidths=1.5, alpha=0.3, zorder=3)
        ax.add_collection(pc)


def _merge_no_overlap(ms_rings, osm_rings):
    """
    One polygon per building.
    1. Merge MS + OSM, skipping OSM polygons that overlap >30% with any MS polygon.
    2. Remove any polygon that is >80% contained within a larger polygon (nested buildings).
    """
    from shapely.geometry import Polygon as SPoly

    # Step 1: merge, OSM skipped if overlaps MS
    if not ms_rings and not osm_rings:
        return []
    if not ms_rings:
        combined = [r for r in osm_rings if len(r) != 16]
    elif not osm_rings:
        combined = list(ms_rings)
    else:
        ms_polys = []
        for r in ms_rings:
            try:
                p = SPoly(r)
                ms_polys.append(p if p.is_valid and p.area > 0 else None)
            except Exception:
                ms_polys.append(None)
        combined = list(ms_rings)
        for ring in osm_rings:
            if len(ring) == 16:
                continue
            try:
                op = SPoly(ring)
                if not op.is_valid or op.area == 0:
                    continue
                if not any(mp is not None and op.intersects(mp) and
                           op.intersection(mp).area / op.area > 0.3
                           for mp in ms_polys):
                    combined.append(ring)
            except Exception:
                continue

    # Step 2: remove polygons >80% contained in a larger polygon
    polys = []
    for r in combined:
        try:
            p = SPoly(r)
            polys.append((r, p if p.is_valid and p.area > 0 else None))
        except Exception:
            polys.append((r, None))

    result = []
    for i, (ring, p) in enumerate(polys):
        if p is None:
            result.append(ring)
            continue
        nested = False
        for j, (_, q) in enumerate(polys):
            if i == j or q is None or q.area <= p.area:
                continue
            try:
                if p.intersection(q).area / p.area > 0.8:
                    nested = True
                    break
            except Exception:
                continue
        if not nested:
            result.append(ring)
    return result


def _draw_infra(ax, infra):
    LINE_KEYS = {"power_line", "pipeline", "aeroway_line"}
    for key, items in infra.items():
        clr = OSM_CLR.get(key, "#ffffff")
        for geom in items:
            if not geom:
                continue
            arr = np.array(geom)
            if arr.ndim != 2 or arr.shape[1] != 2:
                continue
            if key in LINE_KEYS:
                ax.plot(arr[:, 1], arr[:, 0], color=clr, linewidth=1.2,
                        alpha=0.9, zorder=4)
            else:
                try:
                    poly = MplPolygon(arr[:, ::-1])
                    pc = PatchCollection([poly], facecolor=clr, edgecolor=clr,
                                         linewidths=0, alpha=0.35, zorder=4)
                    ax.add_collection(pc)
                except Exception:
                    pass


def _ax_setup(ax, min_lon, max_lon, min_lat, max_lat, esri_img, title):
    ax.set_facecolor(BG)
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")
    ax.imshow(esri_img, extent=[min_lon, max_lon, min_lat, max_lat],
              origin="upper", zorder=1, aspect="auto")
    ax.set_title(title, color="#94a3b8", fontsize=8, fontfamily="monospace", pad=4)


# ── Data fetch ─────────────────────────────────────────────────────────────────

def _fetch(name, lat, lon, elements=None, esri_img=None, ms_rings=None):
    """esri_img/ms_rings: pass in already-fetched values to avoid re-fetching the same
    bbox's imagery/buildings a 2nd time — render_detection_panels and render_ghsl_det
    both call this for the same coordinate otherwise. Fixed 2026-08-11."""
    min_lat, max_lat, min_lon, max_lon = _bbox(lat, lon)

    if esri_img is None:
        try:
            esri_img = fetch_esri_img(min_lon, min_lat, max_lon, max_lat)
        except Exception as e:
            logger.warning(f"[{name}] ESRI fetch failed: {e}")
            esri_img = np.zeros((512, 512, 3), dtype=np.uint8)

    if ms_rings is None:
        try:
            ms_rings = fetch_msft_buildings(min_lat, max_lat, min_lon, max_lon)
            logger.info(f"[{name}] MS buildings: {len(ms_rings)}")
        except Exception as e:
            logger.warning(f"[{name}] MS buildings failed: {e}")
            ms_rings = []

    osm_rings = []
    infra     = {k: [] for k in OSM_CLR}

    if elements is None:
        from scripts.pore.segmap_generator import fetch_osm_elements
        elements = fetch_osm_elements(name, min_lat, max_lat, min_lon, max_lon)

    if elements:
        osm_rings = extract_osm_buildings(elements)
        infra     = extract_osm_infra(elements)
        logger.info(f"[{name}] OSM buildings: {len(osm_rings)}, "
                    f"infra: {sum(len(v) for v in infra.values())}")

    ms_rings = _remove_shells(_filter_min_area(ms_rings, lat))
    return min_lat, max_lat, min_lon, max_lon, esri_img, ms_rings, osm_rings, infra


# ── Render ─────────────────────────────────────────────────────────────────────

def render_detection_panels(name, lat, lon, save=True, elements=None, esri_img=None, ms_rings=None):
    min_lat, max_lat, min_lon, max_lon, esri_img, ms_rings, osm_rings, infra = \
        _fetch(name, lat, lon, elements=elements, esri_img=esri_img, ms_rings=ms_rings)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.patch.set_facecolor(BG)
    _ax_setup(ax, min_lon, max_lon, min_lat, max_lat, esri_img,
              f"Detection Map — {name}  ·  {lat:.4f}, {lon:.4f}")

    osm_rings = [r for r in osm_rings if len(r) != 16]
    all_rings = _merge_no_overlap(ms_rings, osm_rings)
    _draw_buildings(ax, all_rings, CLR_BLDG)
    _draw_infra(ax, infra)

    legend_items = [
        mpatches.Patch(color=CLR_BLDG, label=f"Buildings MS+OSM ({len(all_rings)})"),
    ]
    for key, items in infra.items():
        if items:
            legend_items.append(mpatches.Patch(color=OSM_CLR[key],
                                               label=key.replace("_", " ")))
    ax.legend(handles=legend_items, loc="lower left", facecolor=BG,
              labelcolor="#e2e8f0", fontsize=7, edgecolor="#334155",
              ncol=2, framealpha=0.92)

    plt.tight_layout()

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out  = os.path.join(OUTPUT_DIR, f"{slug_name(name)}_detection.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        logger.info(f"Saved: {out}")
        return out

    plt.show()
    plt.close()
    return None


# ── GHSL-coloured detection variants (population / urbanisation-epoch) ─────────

def _colour_buildings_by_ghsl(ax, rings, ghsl_grid, h, w,
                               min_lon, max_lon, min_lat, max_lat,
                               cmap, vmin, vmax, norm=None):
    """Draw building outlines colour-coded by GHSL grid value at centroid."""
    norm = norm if norm is not None else mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat

    for ring in rings:
        try:
            arr = np.array(ring)
            if arr.shape[1] != 2:
                continue
            cx = arr[:, 0].mean()
            cy = arr[:, 1].mean()
            # map centroid geo → pixel index into ghsl_grid
            px = int((cx - min_lon) / lon_range * w)
            py = int((max_lat - cy) / lat_range * h)
            px = np.clip(px, 0, w - 1)
            py = np.clip(py, 0, h - 1)
            val = ghsl_grid[py, px] if ghsl_grid is not None else 0
            colour = cmap(norm(val)) if val > 0 else (1, 1, 1, 0.6)
            poly = MplPolygon(arr, closed=True)
            pc = PatchCollection([poly], facecolor=colour[:3],
                                 edgecolor="white", linewidths=1.0,
                                 alpha=0.55, zorder=3)
            ax.add_collection(pc)
        except Exception:
            continue


def render_ghsl_det(name, lat, lon, variant="pop", save=True, elements=None, esri_img=None, ms_rings=None):
    """
    variant: 'pop' = population density, 'age' = urbanisation epoch

    Reuses _fetch()/_merge_no_overlap() above for buildings+infra — only the GHSL
    colouring and contrasting infra palette differ from render_detection_panels().
    esri_img/ms_rings: pass in already-fetched values (render_detection_panels already
    fetched them for this same coordinate) to skip re-fetching.
    """
    min_lat, max_lat, min_lon, max_lon, esri_img, ms_rings, osm_rings, infra = \
        _fetch(name, lat, lon, elements=elements, esri_img=esri_img, ms_rings=ms_rings)

    osm_rings = [r for r in osm_rings if len(r) != 16]
    all_rings = _merge_no_overlap(ms_rings, osm_rings)

    h, w = esri_img.shape[:2]
    if variant == "pop":
        ghsl_grid = _sample_pop_grid(min_lat, max_lat, min_lon, max_lon, (h, w))
        ghsl_grid = np.where(ghsl_grid > 0, ghsl_grid, 1).astype(np.float32)
        cmap, vmin, vmax, norm = CMAP_POP, POP_MIN, POP_MAX, NORM_POP
        title = f"Detection — Population (GHSL 2020) · {name}"
        legend_label = "Population (persons per 100m cell, log scale)"
    else:
        ghsl_grid = _sample_epoch_grid(min_lat, max_lat, min_lon, max_lon, (h, w))
        cmap, vmin, vmax, norm = CMAP_AGE, EPOCH_MIN, EPOCH_MAX, None
        title = f"Detection — Urbanisation Epoch (GHSL) · {name}"
        legend_label = "First built-up epoch"

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.patch.set_facecolor(BG)
    _ax_setup(ax, min_lon, max_lon, min_lat, max_lat, esri_img, title)
    ax.set_ylabel(name, color="#e2e8f0", fontsize=8, fontfamily="monospace", labelpad=6)

    _colour_buildings_by_ghsl(ax, all_rings, ghsl_grid, h, w,
                               min_lon, max_lon, min_lat, max_lat,
                               cmap, vmin, vmax, norm=norm)

    # OSM energy infra — contrasting colours (see INFRA_LINE_CLR/INFRA_POLY_CLR above)
    for key, items in infra.items():
        for geom in items:
            if not geom:
                continue
            arr = np.array(geom)
            if arr.ndim != 2 or arr.shape[1] != 2:
                continue
            if key in INFRA_LINE_KEYS:
                ax.plot(arr[:, 1], arr[:, 0], color=INFRA_LINE_CLR,
                        linewidth=1.2, alpha=0.9, zorder=5)
            else:
                try:
                    poly = MplPolygon(arr[:, ::-1])
                    pc = PatchCollection([poly], facecolor=INFRA_POLY_CLR,
                                         edgecolor=INFRA_POLY_CLR,
                                         linewidths=0, alpha=0.5, zorder=5)
                    ax.add_collection(pc)
                except Exception:
                    pass

    # Colorbar
    cb_norm = norm if norm is not None else mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cb_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.6)
    cbar.set_label(legend_label, color="#e2e8f0", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="#e2e8f0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#e2e8f0", fontsize=6)

    ax.text(0.02, 0.98, f"Buildings: {len(all_rings)}",
            transform=ax.transAxes, color="#e2e8f0", fontsize=7,
            va="top", fontfamily="monospace")

    plt.tight_layout()

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = os.path.join(OUTPUT_DIR, f"{slug_name(name)}_{variant}_det.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info(f"Saved: {out}")
        return out
    return fig


def generate_multi(coords=None, save=True):
    coords = coords or ALL_COORDS
    saved  = []
    for name, lat, lon in coords:
        logger.info(f"[{name}] Processing...")
        out = render_detection_panels(name, lat, lon, save=save)
        if out:
            saved.append(out)
    logger.info(f"Done — {len(saved)} detection maps saved to {OUTPUT_DIR}")
    return saved


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lat",  type=float)
    p.add_argument("--lon",  type=float)
    p.add_argument("--name", type=str, default="anchor")
    args = p.parse_args()

    if args.lat and args.lon:
        render_detection_panels(args.name, args.lat, args.lon)
    else:
        generate_multi()
