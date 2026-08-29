"""One-off N=250 real end-to-end test, proportional to the real 10k tier plan
(same 4:7:5 class-count ratios, just scaled by 250/10000=0.025), run through
the REAL production run_pipeline.py functions (run_select/run_stream) -- not
a reimplementation -- so a clean pass here is real evidence the same code
will behave correctly at 10k, not just that a toy script works.

Temporarily monkeypatches run_pipeline.N_PER_STRATUM down to the N=250 plan,
runs select -> stream (no separate osm_carve stage -- merged inline into the
stream worker 2026-08-24, each coordinate carves its own OSM tile via Overpass
right before extracting its other features). The real 10k run_pipeline.py
invocation afterward re-runs `select` (instant) with the real N_PER_STRATUM and
the existing stale-row pruning in run_stream() cleans up these 250 test rows
automatically wherever they don't overlap the real 10k draw.
"""
import json, time, sys
sys.path.insert(0, "/srv/THESIS/energy_profiling_thesis")

import run_pipeline as rp

N_PER_STRATUM_250 = {
    "Suburban":                        40,
    "Dense Urban":                     40,
    "Informal + Urban":                40,
    "Industrial":                      40,
    "Suburban + Agricultural":         13,
    "Industrial + Water":              12,
    "Hydropower Reservoir":            12,
    "Agriculture / Agrivoltaics": 12,
    "Industrial + Arid":               12,
    "Industrial + Forest":             12,
    "Urban + Coastal":                 12,
    "Agricultural + Water":            1,
    "Mangrove + Industrial":           1,
    "Data Centre + Industrial":        1,
    "Coastal + Solar-Wind Hybrid":     1,
    "Coastal + Agricultural":          1,
}
assert sum(N_PER_STRATUM_250.values()) == 250, sum(N_PER_STRATUM_250.values())

rp.N_PER_STRATUM = N_PER_STRATUM_250
# A fresh seed each retry means the built-in stale-row pruning in run_stream()
# naturally clears the previous attempt's rows (they're not in this new selection)
# and every coordinate gets a genuinely fresh caption -- no resume-skip masking
# whether the fix actually worked, and no manual file deletion needed.
import os as _os
rp.RNG_SEED = int(_os.environ.get("N250_SEED", "1001"))
print(f"[N250] using RNG_SEED={rp.RNG_SEED}")

timings = {}

t0 = time.time()
rp.run_select()
timings["select"] = time.time() - t0
print(f"[N250] select done in {timings['select']:.1f}s")

t0 = time.time()
rp.run_stream()
timings["stream"] = time.time() - t0
print(f"[N250] stream done in {timings['stream']:.1f}s ({timings['stream']/250:.2f}s/coord)")

timings["total"] = sum(timings.values())
print(f"[N250] TOTAL: {timings['total']:.1f}s ({timings['total']/60:.1f} min)")

with open("/srv/THESIS/energy_profiling_thesis/logs/n250_timings.json", "w") as f:
    json.dump(timings, f, indent=2)
print("[N250] timings saved to logs/n250_timings.json")
