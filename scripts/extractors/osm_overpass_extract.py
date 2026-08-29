"""Live-Overpass, multi-mirror carve — replaces the local 88GB-planet-scan
carve for the one-time bulk tile build. Writes tiles into the SAME
directory/filename convention as osm_batch_extract.py, so `stream` and every
downstream reader needs zero changes.

Why: `osmium extract -s simple` costs ~53s PER COORDINATE at 10k scale
(confirmed live) -- ~150h total, not viable. Overpass has its own
server-side spatial index, so a single small-bbox query costs a few
seconds regardless of planet size. Splitting the same coordinate list
across 3 independent public mirrors (each running its own pacing loop
in parallel, not as sequential fallback) roughly 3x's throughput without
exceeding any one server's normal load -- these are the same 3 mirrors
`real_anchor_finder.py` already uses for BORE's live discovery queries.

Query uses the standard Overpass "recurse down" idiom (`out body; >; out
skel qt;`) -- pulls in every node referenced by a matched way/relation
even if outside the bbox, so ways come back geometrically COMPLETE
(stricter than local `-s simple`, which can leave a way's off-bbox nodes
out). Verified against the existing local-extraction baseline in
scratchpad/verify_tiles/direct on 21 real coordinates before trusting
this on the real run.
"""
import datetime, json, math, os, threading, time
import requests as _requests

_TS = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)

from scripts.extractors.osm_batch_extract import tile_name, TILES_DIR

_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]
_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": "thesis-pore-carve/1.0 (energy_profiling research; contact bharathhk18.bk@gmail.com)",
    "Accept": "application/json",
}
RATE_DELAY = 2.0     # seconds between requests, PER MIRROR (independent, not shared)
TIMEOUT = 30
OVERPASS_QL_TIMEOUT = 25


def _bbox_deg(lat, lon, size_m=512):
    from scripts.utils.geo import bbox as _geo_bbox
    min_lat, max_lat, min_lon, max_lon = _geo_bbox(lat, lon, size_m)
    return min_lat, min_lon, max_lat, max_lon


def _build_query(lat, lon):
    s, w, n, e = _bbox_deg(lat, lon)
    return (f"[out:json][timeout:{OVERPASS_QL_TIMEOUT}];"
            f"(node({s},{w},{n},{e});way({s},{w},{n},{e});relation({s},{w},{n},{e}););"
            f"out body;>;out skel qt;")


def _fetch_one(mirror_url, lat, lon, rate_state, rate_lock):
    """One coord, one mirror. Enforces this mirror's OWN 2s pacing (independent
    of the other mirrors' timers). Returns parsed Overpass JSON elements or None."""
    with rate_lock:
        elapsed = time.time() - rate_state[0]
        if elapsed < RATE_DELAY:
            time.sleep(RATE_DELAY - elapsed)
        rate_state[0] = time.time()
    query = _build_query(lat, lon)
    for attempt in range(2):
        try:
            resp = _requests.post(mirror_url, data={"data": query}, headers=_HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200:
                return resp.json().get("elements", [])
            if resp.status_code in (429, 504) and attempt == 0:
                time.sleep(10)
                continue
        except Exception:
            if attempt == 0:
                time.sleep(5)
                continue
    return None


def _write_pbf(elements, out_path):
    """Overpass JSON elements (nodes with lat/lon, ways/relations with refs)
    -> a real .osm.pbf via pyosmium, matching what osmium extract would
    produce structurally (nodes, then ways, then relations)."""
    import osmium as _osm

    nodes = [e for e in elements if e.get("type") == "node"]
    ways = [e for e in elements if e.get("type") == "way"]
    rels = [e for e in elements if e.get("type") == "relation"]

    if os.path.exists(out_path):
        os.remove(out_path)
    writer = _osm.SimpleWriter(out_path)
    try:
        for n in nodes:
            writer.add_node(_osm.osm.mutable.Node(
                id=n["id"], location=(n["lon"], n["lat"]),
                tags=n.get("tags", {}), version=1, visible=True,
                changeset=1, uid=1, timestamp=_TS))
        for w in ways:
            writer.add_way(_osm.osm.mutable.Way(
                id=w["id"], nodes=w.get("nodes", []),
                tags=w.get("tags", {}), version=1, visible=True,
                changeset=1, uid=1, timestamp=_TS))
        for r in rels:
            members = [(m["type"][0], m["ref"], m.get("role", "")) for m in r.get("members", [])]
            writer.add_relation(_osm.osm.mutable.Relation(
                id=r["id"], members=members,
                tags=r.get("tags", {}), version=1, visible=True,
                changeset=1, uid=1, timestamp=_TS))
    finally:
        writer.close()


def fetch_tile_any_mirror(lat, lon, tiles_dir=TILES_DIR, mirrors=None):
    """Single-coordinate version for use INSIDE the stream worker (one process,
    one coordinate, called directly -- not the batch queue machinery). Tries
    mirrors in a random order (spreads load across processes automatically,
    no cross-process coordination needed) until one succeeds or all fail.
    Returns True/False. Each mirror gets its own fresh, local pacing state --
    fine here because this function only ever fires ONE request per mirror
    per call, so there's nothing to pace against within a single call."""
    import random
    mirrors = list(mirrors or _MIRRORS)
    random.shuffle(mirrors)
    for mirror_url in mirrors:
        rate_state = [0.0]
        rate_lock = threading.Lock()
        try:
            if fetch_and_write_tile(mirror_url, lat, lon, rate_state, rate_lock, tiles_dir):
                return True
        except Exception:
            continue
    return False


def fetch_and_write_tile(mirror_url, lat, lon, rate_state, rate_lock, tiles_dir=TILES_DIR):
    elements = _fetch_one(mirror_url, lat, lon, rate_state, rate_lock)
    if elements is None:
        return False
    out_path = os.path.join(tiles_dir, tile_name(lat, lon))
    _write_pbf(elements, out_path)
    return True


def batch_overpass_extract(coords, tiles_dir=TILES_DIR, mirrors=None, log_every=50):
    """Carve tiles for every coord via Overpass, shared-queue across all
    mirrors (not a fixed per-mirror split) -- if a mirror goes down or slows
    down, the other mirrors' worker threads simply pull more from the same
    queue, so total throughput degrades gracefully instead of a chunk of
    coordinates being stuck behind one dead server. A coordinate that fails
    (timeout / bad response / write error) goes back on the queue, up to
    MAX_RETRIES times total across all mirrors, before being logged as a
    real failure -- never silently dropped.
    """
    os.makedirs(tiles_dir, exist_ok=True)
    mirrors = mirrors or _MIRRORS
    MAX_RETRIES = 5

    import queue
    q = queue.Queue()
    for la, lo in coords:
        q.put((la, lo, 0, frozenset()))  # (lat, lon, attempt_count, tried_mirrors)

    done = []
    failed = []
    lock = threading.Lock()
    stop = threading.Event()
    t0 = time.time()
    all_mirrors = frozenset(mirrors)

    def _worker(mirror_url):
        rate_state = [0.0]
        rate_lock = threading.Lock()
        while not stop.is_set():
            try:
                la, lo, attempt, tried = q.get(timeout=2)
            except queue.Empty:
                return
            if mirror_url in tried and tried != all_mirrors:
                # already failed on THIS mirror for this coord -- don't waste
                # another attempt on the same one, let a different worker take it
                q.put((la, lo, attempt, tried))
                q.task_done()
                time.sleep(0.2)
                continue
            ok = False
            try:
                ok = fetch_and_write_tile(mirror_url, la, lo, rate_state, rate_lock, tiles_dir)
            except Exception:
                ok = False
            with lock:
                if ok:
                    done.append((la, lo))
                else:
                    new_tried = tried | {mirror_url}
                    if new_tried < all_mirrors and attempt + 1 < MAX_RETRIES:
                        # genuine mirrors left untried -- always give it another shot
                        q.put((la, lo, attempt + 1, new_tried))
                    elif attempt + 1 < MAX_RETRIES:
                        # already tried every mirror at least once -- reset and retry all
                        q.put((la, lo, attempt + 1, frozenset()))
                    else:
                        failed.append((la, lo))
                n = len(done) + len(failed)
                if n % log_every == 0:
                    elapsed = time.time() - t0
                    print(f"  {n}/{len(coords)} done ({len(failed)} failed), "
                          f"{elapsed:.0f}s elapsed, {elapsed/max(n,1):.1f}s/coord avg")
            q.task_done()

    threads = [threading.Thread(target=_worker, args=(m,), daemon=True) for m in mirrors]
    for t in threads:
        t.start()
    q.join()
    stop.set()
    for t in threads:
        t.join(timeout=5)

    print(f"batch_overpass_extract: {len(done)} ok, {len(failed)} failed, "
          f"{time.time()-t0:.0f}s total")
    return done, failed
