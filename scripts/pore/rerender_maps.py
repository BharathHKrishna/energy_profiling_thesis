"""
Re-render the four map images per coordinate, without touching the feature table.

Why this is separate from the pipeline's own worker: that worker re-extracts every
feature before it draws anything, which would rewrite pore_features.csv. The
features are already correct, so this renders only. It reads the coordinate list
from the CSV and never writes to it.

The reason to re-render at all is the OSM read path. Tiles used to be written to
PBF and read back with `osmium export`, which needs ID-ordered input and silently
dropped every way when it did not get it (see osm_json_tiles.py). Buildings and
energy infrastructure are ways, so the maps rendered so far carry no OSM overlay
at all: they show Microsoft footprints and land cover only. With tiles now read as
Overpass JSON the same coordinate returns its real buildings, so the detection
maps can finally draw them.

Every tile is already cached on disk, so this makes no Overpass request and can
run alongside the caption job without competing for the same rate limit.

Usage:
    python -m scripts.pore.rerender_maps --limit 20        # try a few first
    python -m scripts.pore.rerender_maps --workers 16
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = "/srv/THESIS/energy_profiling_thesis"
sys.path.insert(0, BASE)

FEATURES_CSV = os.path.join(BASE, "outputs/csv/pore_features.csv")


def render_one(row):
    """Four maps for one coordinate. Returns (name, ok, error)."""
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, BASE)
    import logging
    logging.disable(logging.INFO)
    import numpy as _np

    lat, lon = float(row["lat"]), float(row["lon"])
    # The renderers put the name they are handed into BOTH the figure title and the
    # output filename. A stratum name alone is not unique (sixteen classes across ten
    # thousand coordinates), so passing it bare sends all 1,600 Suburban tiles to one
    # file; passing it with the coordinate appended fixes the filename but then prints
    # the coordinate twice in the title. So the plain name is used for rendering, which
    # reproduces the original titles exactly, and the four files are renamed afterwards
    # to the slug(stratum)_LAT_LON convention the existing map set uses.
    name = str(row.get("stratum_name", ""))
    import shutil, tempfile
    tmp_root = tempfile.mkdtemp(prefix="maprender_")
    tmp_seg = os.path.join(tmp_root, "seg"); tmp_det = os.path.join(tmp_root, "det")
    os.makedirs(tmp_seg, exist_ok=True); os.makedirs(tmp_det, exist_ok=True)
    seg_dir = os.environ.get("PORE_SEGMAP_DIR") or os.path.join(BASE, "outputs/maps/segmaps")
    det_dir = os.environ.get("PORE_DETECTION_DIR") or os.path.join(BASE, "outputs/maps/detection")
    try:
        from scripts.pore.segmap_generator import (
            generate_single, fetch_osm_elements, _bbox, render_ghsl_segmap, fetch_esri_img)
        from scripts.pore.detection_map import render_detection_panels, render_ghsl_det
        from scripts.extractors.worldcover_extractor import fetch_worldcover_pixels
        from scripts.extractors.msft_buildings_extractor import fetch_msft_buildings

        # Each coordinate renders into its OWN directory. The renderers name their
        # output from the stratum name alone, which is not unique, so with several
        # workers in flight two tiles of the same class would otherwise write the
        # same file and one would rename the other's image under its own
        # coordinate. That produced 32 hard failures and silently mismatched
        # images in an earlier pass. OUTPUT_DIR is read at call time from module
        # globals, so redirecting it per coordinate removes the shared path
        # entirely rather than trying to sequence access to it.
        import scripts.pore.segmap_generator as _seg
        import scripts.pore.detection_map as _det
        _seg.OUTPUT_DIR = tmp_seg
        _det.OUTPUT_DIR = tmp_det

        min_lat, max_lat, min_lon, max_lon = _bbox(lat, lon)
        elements = fetch_osm_elements(name, min_lat, max_lat, min_lon, max_lon,
                                      lat=lat, lon=lon)

        # shared inputs fetched once and handed to all four renderers, the same
        # way the pipeline worker does it
        try:
            esri_img = fetch_esri_img(min_lon, min_lat, max_lon, max_lat)
        except Exception:
            esri_img = _np.zeros((512, 512, 3), dtype=_np.uint8)
        wc_raw, _, wc_ok = fetch_worldcover_pixels(min_lat, max_lat, min_lon, max_lon, lat, lon)
        wc_pixels = (wc_raw, wc_ok)
        try:
            ms_rings = fetch_msft_buildings(min_lat, max_lat, min_lon, max_lon)
        except Exception:
            ms_rings = []

        # exactly the call sequence run_pipeline uses, argument names and order
        # included; generate_single takes (lat, lon) first and does not take
        # elements at all, and the detection renderers name the Microsoft
        # footprints ms_rings
        generate_single(lat, lon, stratum_name=name, esri_img=esri_img, wc_pixels=wc_pixels)
        render_ghsl_segmap(name, lat, lon, variant="pop", esri_img=esri_img,
                           wc_pixels=wc_pixels)
        render_detection_panels(name, lat, lon, elements=elements, esri_img=esri_img,
                                ms_rings=ms_rings)
        render_ghsl_det(name, lat, lon, variant="pop", elements=elements,
                        esri_img=esri_img, ms_rings=ms_rings)

        # move the four outputs onto their coordinate-bearing filenames
        from scripts.utils.naming import slug_name
        stem = slug_name(name)
        target = f"{stem}_{lat:.5f}_{lon:.5f}"
        for tmpd, dest, suffix in ((tmp_seg, seg_dir, "segmap"),
                                   (tmp_seg, seg_dir, "pop_segmap"),
                                   (tmp_det, det_dir, "detection"),
                                   (tmp_det, det_dir, "pop_det")):
            src = os.path.join(tmpd, f"{stem}_{suffix}.png")
            if os.path.exists(src):
                os.makedirs(dest, exist_ok=True)
                shutil.move(src, os.path.join(dest, f"{target}_{suffix}.png"))
        return f"{target}", True, None
    except Exception as e:
        return name, False, repr(e)[:160]
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=FEATURES_CSV)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if args.limit:
        rows = rows[:args.limit]
    print(f"coordinates: {len(rows)}  workers: {args.workers}")

    ok = 0
    failures = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(render_one, r): r for r in rows}
        for n, fut in enumerate(as_completed(futs), 1):
            name, good, err = fut.result()
            if good:
                ok += 1
            else:
                failures.append((name, err))
            if n % 50 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(rows)}  {rate:.2f}/s  "
                      f"~{(len(rows)-n)/rate/60:.0f} min left  {len(failures)} failed",
                      flush=True)

    mins = (time.time() - t0) / 60
    print(f"\nCOMPLETE  {ok}/{len(rows)} rendered, {len(failures)} failed, {mins:.1f} min")
    for name, err in failures[:10]:
        print(f"  FAIL {name}: {err}")


if __name__ == "__main__":
    main()
