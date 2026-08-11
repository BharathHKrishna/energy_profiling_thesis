"""
Download GHSL Built-up Surface epoch TIFs (R2023A, 100m Mollweide).

Downloads all 10 epochs (1975–2020) and creates a derived 'first_epoch' raster
where each cell = the earliest year it became built-up.

Output:
    rasters/ghsl/built_epochs/GHS_BUILT_S_E{year}_100m.tif  (10 files)
    rasters/ghsl/built_epochs/GHS_BUILT_FIRST_EPOCH.tif     (derived)

Usage:
    python scripts/utils/download_ghsl_epochs.py
"""
import os, sys, zipfile, shutil, requests
import numpy as np
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")
from scripts.utils.logger import get_logger

logger = get_logger("ghsl_download")

BASE      = "/srv/THESIS/energy_profiling_thesis"
OUT_DIR   = os.path.join(BASE, "rasters/ghsl/built_epochs")
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]

# JRC FTP — R2023A built-up surface, 100m Mollweide
URL_TMPL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_BUILT_S_GLOBE_R2023A/"
    "GHS_BUILT_S_E{year}_GLOBE_R2023A_54009_100/V1-0/"
    "GHS_BUILT_S_E{year}_GLOBE_R2023A_54009_100_V1_0.zip"
)
TIF_TMPL = "GHS_BUILT_S_E{year}_GLOBE_R2023A_54009_100_V1_0.tif"
OUT_TMPL = os.path.join(OUT_DIR, "GHS_BUILT_S_E{year}_100m.tif")


def download_epoch(year):
    out_path = OUT_TMPL.format(year=year)
    if os.path.exists(out_path):
        logger.info(f"[{year}] Already exists — skipping")
        return out_path

    url = URL_TMPL.format(year=year)
    zip_path = out_path.replace(".tif", ".zip")
    logger.info(f"[{year}] Downloading {url}")

    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r  {pct:.1f}%  ({done//1024//1024} MB)", end="", flush=True)
    print()

    logger.info(f"[{year}] Extracting...")
    tmp_dir = zip_path.replace(".zip", "_tmp")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_dir)

    tif_name = TIF_TMPL.format(year=year)
    src = os.path.join(tmp_dir, tif_name)
    if not os.path.exists(src):
        # search recursively
        for root, _, files in os.walk(tmp_dir):
            for fn in files:
                if fn.endswith(".tif"):
                    src = os.path.join(root, fn)
                    break

    shutil.move(src, out_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.remove(zip_path)
    logger.info(f"[{year}] Saved → {out_path}")
    return out_path


def build_first_epoch(tif_paths):
    """
    Create derived raster: each cell = earliest year it became built-up (>0).
    Keeps all 10 epoch files open simultaneously to avoid repeated open/close overhead.
    Cells never built-up = 0.
    """
    import rasterio, contextlib
    from rasterio.windows import Window

    derived_path = os.path.join(OUT_DIR, "GHS_BUILT_FIRST_EPOCH.tif")
    if os.path.exists(derived_path):
        logger.info("First-epoch raster already exists — skipping")
        return derived_path

    logger.info("Building first-epoch raster (chunked, all files open)...")
    CHUNK_ROWS = 2000

    with rasterio.open(tif_paths[0]) as ref:
        profile = ref.profile.copy()
        profile.update(dtype="float32", nodata=0.0, compress="lzw", bigtiff="YES")
        height, width = ref.height, ref.width
        nodatas = []

    with contextlib.ExitStack() as stack:
        srcs = [stack.enter_context(rasterio.open(p)) for p in tif_paths]
        nodatas = [s.nodata if s.nodata is not None else -200.0 for s in srcs]

        with rasterio.open(derived_path, "w", **profile) as dst:
            for row_off in range(0, height, CHUNK_ROWS):
                rows = min(CHUNK_ROWS, height - row_off)
                win = Window(0, row_off, width, rows)
                chunk = np.zeros((rows, width), dtype=np.float32)

                for src, year, nodata in zip(srcs, EPOCHS, nodatas):
                    data = src.read(1, window=win).astype(np.float32)
                    built = (data > 0) & (data != nodata)
                    chunk[built & (chunk == 0)] = year

                dst.write(chunk, 1, window=win)
                pct = (row_off + rows) / height * 100
                logger.info(f"  {pct:.1f}%  (rows {row_off}–{row_off+rows})")

    logger.info(f"First-epoch raster saved → {derived_path}")
    return derived_path


if __name__ == "__main__":
    tif_paths = []
    for year in EPOCHS:
        try:
            p = download_epoch(year)
            tif_paths.append(p)
        except Exception as e:
            logger.error(f"[{year}] Failed: {e}")

    if len(tif_paths) == len(EPOCHS):
        build_first_epoch(tif_paths)
        logger.info("All done. Add to config.yaml:")
        logger.info("  ghsl.built_epoch: rasters/ghsl/built_epochs/GHS_BUILT_FIRST_EPOCH.tif")
    else:
        logger.warning(f"Only {len(tif_paths)}/{len(EPOCHS)} epochs downloaded — skipping derived raster")
