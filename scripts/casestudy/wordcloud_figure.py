"""
Word clouds for the city case study.

Why this is not a plain frequency cloud. Every caption comes from the same
generator, so a few words sit in nearly all 948 of them: heating in 100 percent
of captions, lighting in 100, energy in 99, infrastructure in 97. A cloud sized
by raw count shows those words and little else, which describes the caption
template rather than the city.

So a term is admitted only when it is over-represented against a background
corpus, scored as a smoothed log odds ratio. For the city cloud the background
is the 10,000 caption dataset, so the figure answers "what do Frankfurt's
captions say that the global sample does not". For the zone clouds the
background is the rest of the city, so each panel answers "what separates this
part of Frankfurt from the other parts".

Inside that selection, size is ordinary document frequency, so a large word is
one that genuinely appears in many captions. Colour carries the log odds, so the
darkest words are the most distinctive.

Negation is handled explicitly. Frankfurt has no cooling season, and the
captions say so, so a bare two word phrase would put "meaningful cooling" on the
page with its meaning reversed. A phrase may therefore open with a negator or a
qualifier and run to four words, those opening words never stand alone, and a
short phrase is dropped when a longer one containing it occurs about as often.
That prints "no meaningful cooling season" rather than leaving the misreading up.

Layout is a grid packer. Words are placed largest first, each at the free
position nearest the centre on a fine grid, trying smaller type and then a
quarter turn before giving up. No wordcloud library is involved: matplotlib
measures the glyphs and the packer only has to avoid overlaps.

Usage:
    python -m scripts.casestudy.wordcloud_figure --slug frankfurt
"""

import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

BASE = "/srv/THESIS/energy_profiling_thesis"
GLOBAL_CAPTIONS = os.path.join(BASE, "outputs/captions/pore_captions.json")

# The figures are drawn at the width they are printed at. A4 with the thesis's
# 2.5 cm margins leaves a 160 mm text block, so a figure included at
# \textwidth is not rescaled and the font sizes below are the sizes that reach
# the page. Drawing wide and letting LaTeX shrink the result put the smallest
# words on the zone figure at about three points, which is unreadable in print.
TEXT_WIDTH_IN = 6.3

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# Connectives, framing verbs and empty nouns. None of them says anything about a
# place: they are how the generator glues clauses together, they occur at much
# the same rate everywhere, and leaving them in would only cost cloud space.
STOP = set("""
a an the and or of to in is are was were for with this that it as by on at from be been being has have had
which their they there here where what when how not nor but if then than so such can could would should may
might also very more most much many few all both each into out up down over under again further once other
suggests suggesting indicating indicates indicate implying implies implied reflecting reflects reflect
consistent consists consisting characterized making makes made shows showing show reinforcing given
typical typically likely appears appear seems seem while though although however overall across around
its it's location area tile site place places region within between along near
features feature includes include including included lack lacks lacking mix aligns align aligned
contributing contributes contribute character characteristics terms used using well still rather quite only
potential primarily driven lack lacks lacking absence absent devoid mass
points point concentration concentrated conditions due
reinforce reinforces primary dominates dominate nearly
""".split())

# These may open a phrase but never stand alone. "effectively unpopulated" and
# "minimal human presence" already read as low values, so they need no help; they
# are here only so that the bare adjective does not float in the cloud on its own.
QUALIFIERS = {"no", "not", "non", "without", "essentially", "virtually", "almost",
              "minimal", "little", "barely", "hardly", "negligible", "limited",
              "predominantly", "mostly", "largely", "mainly", "some", "substantial",
              "extensive", "moderate", "significant", "considerable", "effectively",
              "generally", "relatively", "low", "high", "trace", "traces"}

# True negations, which invert whatever noun phrase follows them. The captions
# write the same fact three ways, "no meaningful cooling season", "lack of a
# meaningful cooling season" and "absence of a meaningful cooling season", so
# every one of them is normalised to a single term beginning "no". Without that
# the cloud prints "meaningful cooling" in 89 percent of Frankfurt's captions and
# states the exact opposite of what the city's climate does.
#
# "limited" is deliberately absent: "infrastructure is limited to transmission
# towers" means the towers are there, and treating it as a negation would erase
# them.
NEGATORS = {"no", "not", "non", "without", "lack", "lacks", "lacking",
            "absence", "absent", "devoid"}

# Single words that name nothing on their own. Each still counts inside a phrase
# ("solar generation", "agricultural landscape"); it just never floats in the
# cloud by itself, where "generation" or "mixed" leaves the reader guessing what
# is generated or mixed. The list was read off Frankfurt's actual candidates
# rather than written in advance.
NO_SOLO = {"generation", "mixed", "seasonal", "grass", "landscape", "agricultural",
           "renewable", "power", "continuous", "shading", "coverage", "compact",
           "developed", "interspersed", "green", "share", "reliance", "open"}

# The generator words one fact several ways. Those wordings are merged before
# counting, the same way negations are, so the fact is counted once at its true
# frequency instead of splitting into fragments. "minimal human" reached the
# cloud cut off because development, activity, modification and presence each
# fell below the threshold on their own.
PARAPHRASES = [
    (re.compile(r"\bminimal human (development|activity|modification|presence|influence|footprint)\b"),
     "minimal human activity"),
    (re.compile(r"\blow energy (demand|consumption|use|needs|requirements)\b"), "low energy demand"),
    (re.compile(r"\b(heating|cooling) (needs|requirements)\b"), r"\1 demand"),
    (re.compile(r"\bdense mass of\b"), "dense"),
]

# One hue, light to dark, for one magnitude: how much more common a term is here
# than in the background. It interpolates between three anchors and starts at a
# mid blue rather than near white, because every mark is text and the smallest
# words must still read at 5:1 on white in print.
RAMP = ["#256abf", "#184f95", "#0d366b"]
INK = LinearSegmentedColormap.from_list("ink", RAMP)

WORD = re.compile(r"[a-z][a-z\-]*")
MAX_NGRAM = 4


def terms_of(text):
    """The set of words and phrases in one caption.

    Phrases never cross a comma or full stop, so two words that merely sit near
    each other in different clauses are never glued into a term that no caption
    actually contains. A phrase under negation is emitted with a leading "no",
    which both fixes its meaning and merges the generator's three ways of saying
    the same thing.
    """
    out = set()
    text = text.lower()
    for pat, sub in PARAPHRASES:
        text = pat.sub(sub, text)
    for clause in re.split(r"[.;,:]", text):
        ws = WORD.findall(clause)
        n = len(ws)
        content = [len(w) >= 3 and w not in STOP for w in ws]
        neg = [False] * n
        for i in range(n):
            # a negator a few slots back with nothing of substance in between,
            # which is what "lack of a meaningful cooling season" looks like
            for j in range(max(0, i - 4), i):
                if ws[j] in NEGATORS and not any(content[k] for k in range(j + 1, i)):
                    neg[i] = True
                    break
            # negation then runs on through the noun phrase and stops at the
            # first word of no substance, so "effectively unpopulated with small
            # town lighting" leaves the lighting alone
            if not neg[i] and i and neg[i - 1] and content[i - 1]:
                neg[i] = True
        for i, w in enumerate(ws):
            if not (content[i] or w in QUALIFIERS):
                continue
            if neg[i]:
                # Only the head of a negated run is emitted, and only as a
                # phrase. Prefixing every word inside the run instead would turn
                # "no built-up surface" into the extra terms "no surface" and
                # "no demand", which are artefacts of the prefixing rather than
                # anything a caption says.
                if i and neg[i - 1]:
                    continue
                for k in range(2, MAX_NGRAM + 1):
                    if i + k > n or not all(content[i + m] for m in range(1, k)):
                        break
                    out.add("no " + " ".join(ws[i:i + k]))
                continue
            if content[i] and w not in QUALIFIERS and w not in NO_SOLO:
                out.add(w)
            for k in range(2, MAX_NGRAM + 1):
                if i + k > n or not all(content[i + m] for m in range(1, k)):
                    break
                out.add(" ".join(ws[i:i + k]))
    return out


def doc_freq(captions):
    df = Counter()
    for t in captions:
        df.update(terms_of(t))
    return df


def score(df_fg, n_fg, df_bg, n_bg, min_share, min_lo):
    """Smoothed log odds for every term, keeping only what is common and distinctive."""
    keep = []
    floor = max(8, int(min_share * n_fg))
    for term, c in df_fg.items():
        if c < floor:
            continue
        p_fg = (c + 0.5) / (n_fg + 1.0)
        p_bg = (df_bg.get(term, 0) + 0.5) / (n_bg + 1.0)
        lo = math.log(p_fg / p_bg)
        if lo >= min_lo:
            keep.append((term, c, lo))
    return keep


def dedupe(rows, ratio=1.6):
    """Drop a short phrase that a longer selected phrase already carries.

    "meaningful cooling" and "no meaningful cooling season" describe nearly the
    same captions, so only the longer one is kept. A short phrase survives when
    it is much more widespread than any phrase containing it, which means it also
    occurs on its own.
    """
    rows = sorted(rows, key=lambda r: (-len(r[0].split()), -r[1]))
    kept = []
    for term, c, lo in rows:
        if any(term != k and c <= ratio * kc and re.search(rf"\b{re.escape(term)}\b", k)
               for k, kc, _ in kept):
            continue
        kept.append((term, c, lo))
    return kept


def _candidates(rng, nx=400, ny=260):
    """Every position a word may be centred on, nearest the middle first.

    This replaced a spiral. A spiral turns by a fixed angle per step, so its
    samples spread further apart the further out it winds, and in the outer half
    of a panel it stepped straight over gaps a word would have fitted: a grid
    scan of the finished Frankfurt cloud found hundreds of free positions for
    seven of the eight words the spiral had dropped. A fine grid ordered by
    distance from the centre keeps the look, largest words in the middle, and
    misses no gap wider than one cell. The small jitter stops words lining up
    along visible rings.
    """
    gx, gy = (g.ravel() for g in np.meshgrid(np.linspace(0.004, 0.996, nx),
                                             np.linspace(0.004, 0.996, ny)))
    jitter = np.random.default_rng(rng.randrange(2**32)).random(gx.size) * 0.015
    order = np.argsort(np.hypot(gx - 0.5, gy - 0.5) + jitter)
    return gx[order], gy[order]


def _first_free(gx, gy, w, h, placed):
    """Index of the first candidate where a w by h box fits clear of every placed box."""
    ok = ((gx - w / 2 >= 0.004) & (gx + w / 2 <= 0.996)
          & (gy - h / 2 >= 0.004) & (gy + h / 2 <= 0.996))
    for x0, y0, x1, y1 in placed:
        ok &= (gx + w / 2 <= x0) | (gx - w / 2 >= x1) | (gy + h / 2 <= y0) | (gy - h / 2 >= y1)
    hit = np.flatnonzero(ok)
    return int(hit[0]) if hit.size else None


def draw_cloud(fig, ax, rows, rng, smin, smax, cmap, pad=0.004):
    """Place every term, largest first. Returns the terms that still did not fit."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ab = ax.get_window_extent(rend)
    gx, gy = _candidates(rng)

    counts = [c for _, c, _ in rows]
    los = [lo for _, _, lo in rows]
    cmax, cmin = max(counts), min(counts)
    lmax, lmin = max(los), min(los)

    placed, dropped = [], []
    for term, c, lo in sorted(rows, key=lambda r: -r[1]):
        frac = (c - cmin) / (cmax - cmin) if cmax > cmin else 1.0
        size = smin + (smax - smin) * math.sqrt(frac)
        shade = (lo - lmin) / (lmax - lmin) if lmax > lmin else 1.0
        t = ax.text(0.5, 0.5, term, fontsize=size, ha="center", va="center",
                    color=cmap(shade), transform=ax.transAxes)
        ok = False
        # Shrink before rotating, and rotate before dropping. A word that will
        # not fit at its natural size still belongs on the page, and a packer
        # working on bounding boxes leaves enough dead space that dropping on
        # the first miss loses half the cloud. Trying rotation first instead
        # turns a narrow panel almost entirely sideways, which is unreadable.
        for rot in (0, 90):
            t.set_rotation(rot)
            for scale in (1.0, 0.88, 0.76, 0.64, 0.52, 0.42, 0.34):
                # never below the print floor: a word too small to read is
                # worth less than the space it occupies, so it is dropped instead
                if size * scale < 6.3 and scale < 1.0:
                    break
                t.set_fontsize(size * scale)
                bb = t.get_window_extent(rend)
                w = bb.width / ab.width + pad
                h = bb.height / ab.height + pad
                if w > 0.99 or h > 0.99:
                    continue
                hit = _first_free(gx, gy, w, h, placed)
                if hit is not None:
                    cx, cy = float(gx[hit]), float(gy[hit])
                    t.set_position((cx, cy))
                    placed.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
                    ok = True
                if ok:
                    break
            if ok:
                break
        if not ok:
            t.remove()
            dropped.append(term)
    return dropped


def coord_key(lat, lon):
    """The pipeline's own caption key. Trailing zeros are stripped by round(),
    so formatting with %.6f here would miss more than a hundred tiles."""
    return f"{round(float(lat), 6)},{round(float(lon), 6)}"


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError):
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="frankfurt")
    ap.add_argument("--city", default=None, help="display name, defaults to the slug")
    ap.add_argument("--terms", type=int, default=40)
    ap.add_argument("--zone-terms", type=int, default=21)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    city = args.city or args.slug.capitalize()
    outdir = os.path.join(BASE, "outputs/casestudy", args.slug)
    caps = json.load(open(os.path.join(outdir, "captions.json")))
    rows = list(csv.DictReader(open(os.path.join(outdir, "features.csv"))))
    figdir = os.path.join(BASE, "thesis/figures")
    os.makedirs(figdir, exist_ok=True)

    matched = sum(1 for r in rows if coord_key(r["lat"], r["lon"]) in caps)
    print(f"{city}: {len(rows)} tiles, {len(caps)} captions, {matched} matched")
    texts = list(caps.values())
    world = list(json.load(open(GLOBAL_CAPTIONS)).values())
    print(f"world background: {len(world)} captions")

    df_city = doc_freq(texts)
    df_world = doc_freq(world)

    sel = dedupe(score(df_city, len(texts), df_world, len(world),
                       min_share=0.045, min_lo=0.40))
    sel.sort(key=lambda r: -r[2])
    sel = sel[:args.terms]
    print(f"\ncity cloud: {len(sel)} terms")
    for term, c, lo in sorted(sel, key=lambda r: -r[1])[:20]:
        print(f"   {term:<34} {c:4d}  {100*c/len(texts):3.0f}%  logodds {lo:.2f}")

    rng = random.Random(args.seed)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.3))
    dropped = draw_cloud(fig, ax, sel, rng, 8.0, 28, INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext in ("pdf", "png"):
        p = os.path.join(figdir if ext == "pdf" else outdir,
                         f"fig_{args.slug}_wordcloud.{ext}")
        fig.savefig(p, dpi=200)
        print("wrote", p)
    print(f"placed {len(sel)-len(dropped)}/{len(sel)}"
          + (f", dropped: {', '.join(dropped)}" if dropped else ", none dropped"))
    plt.close(fig)

    # Zones. Built-up share orders a city from centre to edge without needing an
    # administrative boundary for every district, and the three bands it cuts
    # here are close to equal in size.
    zones = [("Built-up core", lambda r: num(r, "wc_built_up_pct") >= 50),
             ("Mixed fringe", lambda r: 15 <= num(r, "wc_built_up_pct") < 50),
             ("Green periphery", lambda r: num(r, "wc_built_up_pct") < 15)]
    members = {n: [caps[coord_key(r["lat"], r["lon"])] for r in rows
                   if f(r) and coord_key(r["lat"], r["lon"]) in caps]
               for n, f in zones}

    # stacked, not side by side: a third of the text block is too narrow to set
    # "urban-centre population density" horizontally at a readable size
    fig, axes = plt.subplots(3, 1, figsize=(TEXT_WIDTH_IN, 7.8))
    for ax, (name, _) in zip(axes, zones):
        mine = members[name]
        rest = [t for n, v in members.items() if n != name for t in v]
        z = dedupe(score(doc_freq(mine), len(mine), doc_freq(rest), len(rest),
                         min_share=0.10, min_lo=0.45))
        z.sort(key=lambda r: -r[2])
        z = z[:args.zone_terms]
        drop = draw_cloud(fig, ax, z, random.Random(args.seed), 7.5, 20, INK)
        ax.set_title(f"{name}   ({len(mine)} tiles)", fontsize=9.5, pad=4)
        print(f"\n{name}: {len(mine)} tiles, {len(z)} terms, {len(drop)} dropped")
        for term, c, lo in sorted(z, key=lambda r: -r[1])[:9]:
            print(f"   {term:<32} {100*c/len(mine):3.0f}%  logodds {lo:.2f}")
        if drop:
            print("   dropped:", ", ".join(drop))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.01, hspace=0.16)
    for ext in ("pdf", "png"):
        p = os.path.join(figdir if ext == "pdf" else outdir,
                         f"fig_{args.slug}_zone_clouds.{ext}")
        fig.savefig(p, dpi=200)
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
