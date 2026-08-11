"""Shared coordinate-box math — was independently reimplemented in 5 different files
(demand_extractor.py, osm_batch_extract.py, feature_extractor.py, osm_extractor.py,
segmap_generator.py) before 2026-08-11, with inconsistent edge-case handling. This is the
one place that owns the actual degrees-per-meter conversion; callers that need a different
output shape (list, dict, rounded) reshape this function's result themselves rather than
reimplementing the math.
"""
import math


def bbox(lat, lon, size_m=512):
    """(min_lat, max_lat, min_lon, max_lon) for a size_m x size_m box centred on (lat, lon).
    Unrounded — round at the call site if the caller specifically needs that (e.g. for a
    filename that must match elsewhere; see osm_batch_extract.py's tile naming)."""
    half = size_m / 2
    dlat = half / 111320
    dlon = half / (111320 * math.cos(math.radians(abs(lat) or 0.001)))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon
