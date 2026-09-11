"""
Build a regular 512 m grid covering one city, clipped to its real administrative
boundary.

This is the sampling stage of the city case study, and it deliberately replaces
BORE and SELECT rather than reusing them. BORE proposes candidates from
OpenStreetMap tags and accepts them only if ESA WorldCover and GHSL agree, which
makes the resulting sample selective by construction: it contains the places the
gates were written to look for and nothing else. A grid asks a different
question. It takes every tile in a city, in order, whatever is on it, and so
covers the parks, the water, the rail yards and the empty lots that no class gate
would ever propose.

Nothing downstream is aware of the difference. The grid is written in the same
column layout SELECT produces, so PORE extracts, computes demand, renders maps
and captions exactly as it does for the thesis dataset.

The boundary comes from OpenStreetMap via Nominatim rather than a bounding box,
because a box around Frankfurt includes a good deal of Offenbach and Eschborn.
Clipping to the real administrative polygon is what makes "the city" a defensible
unit rather than a rectangle drawn by hand.

Usage:
    python -m scripts.casestudy.city_grid --city "Frankfurt am Main"
    python -m scripts.casestudy.city_grid --city Vancouver --spacing 512
"""

import argparse
import csv
import json
import math
import os
import urllib.parse
import urllib.request

BASE = "/srv/THESIS/energy_profiling_thesis"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
UA = "kit-thesis-casestudy/1.0"


def fetch_boundary(city: str) -> dict:
    """The city's administrative polygon as GeoJSON geometry."""
    q = urllib.parse.urlencode({
        "q": city, "format": "json", "polygon_geojson": 1, "limit": 5,
    })
    req = urllib.request.Request(f"{NOMINATIM}?{q}", headers={"User-Agent": UA})
    results = json.load(urllib.request.urlopen(req, timeout=120))
    for r in results:
        if r.get("class") == "boundary" and r.get("type") == "administrative":
            return r
    raise SystemExit(f"No administrative boundary found for {city!r}")


def grid_points(geom, spacing_m: float):
    """Tile centres on a regular grid, keeping only those inside the boundary.

    The grid is laid out in metres on an equal-area projection rather than in
    degrees, so tiles stay the same size from the south of the city to the north
    instead of narrowing with latitude. Each centre is then converted back to
    latitude and longitude, which is what the extractors take.
    """
    import pyproj
    from shapely.geometry import shape, Point
    from shapely.ops import transform
    from shapely.prepared import prep

    poly = shape(geom)
    to_m = pyproj.Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True).transform
    to_deg = pyproj.Transformer.from_crs("ESRI:54009", "EPSG:4326", always_xy=True).transform
    poly_m = transform(to_m, poly)
    ready = prep(poly_m)

    minx, miny, maxx, maxy = poly_m.bounds
    # start half a tile in, so centres sit in the middle of their cells
    xs = [minx + spacing_m * (i + 0.5) for i in range(int((maxx - minx) // spacing_m) + 1)]
    ys = [miny + spacing_m * (j + 0.5) for j in range(int((maxy - miny) // spacing_m) + 1)]

    pts = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if ready.contains(Point(x, y)):
                lon, lat = to_deg(x, y)
                pts.append((j, i, round(lat, 6), round(lon, 6)))
    return pts, poly_m.area / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--slug", default=None, help="output folder name; derived from city if unset")
    ap.add_argument("--spacing", type=float, default=512.0)
    ap.add_argument("--bbox-m", type=int, default=512)
    args = ap.parse_args()

    slug = args.slug or args.city.split(",")[0].split()[0].lower()
    outdir = os.path.join(BASE, "outputs/casestudy", slug)
    os.makedirs(outdir, exist_ok=True)

    rec = fetch_boundary(args.city)
    geom = rec["geojson"]
    with open(os.path.join(outdir, "boundary.geojson"), "w") as fh:
        json.dump({"type": "Feature",
                   "properties": {"name": rec["display_name"], "osm_id": rec["osm_id"]},
                   "geometry": geom}, fh)

    pts, area_km2 = grid_points(geom, args.spacing)

    out = os.path.join(outdir, "grid.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        # same columns SELECT writes, so PORE needs no change to read this
        w.writerow(["stratum_name", "importance_tier", "strata_type", "location_name",
                    "lat", "lon", "primary_class", "primary_pct",
                    "secondary_class", "secondary_pct", "bbox_m", "remark"])
        for row, col, lat, lon in pts:
            # stratum_name doubles as the map filename stem downstream, so it has to
            # be unique per tile and safe as a filename
            name = f"{slug}_r{row:03d}_c{col:03d}"
            w.writerow([name, "GRID", "grid", f"{lat:.4f},{lon:.4f}",
                        lat, lon, "", "", "", "", args.bbox_m, "city_grid"])

    print(f"city        : {rec['display_name']}")
    print(f"area        : {area_km2:.1f} km2")
    print(f"spacing     : {args.spacing:.0f} m")
    print(f"tiles inside: {len(pts)}")
    print(f"written     : {out}")


if __name__ == "__main__":
    main()
