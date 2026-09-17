"""Figure 5.16 -- the terms most distinctive to six classes, from all 10,000 captions.

Method (unchanged from the figure first drawn on 2026-09-11, whose code was never saved):
a caption is a set of lower-case words of four or more letters, minus a stop list of
generic and linking words. For a class, a term's share is the fraction of that class's
captions containing it, and its lift is that share divided by the share across all
captions. A term qualifies when it appears in at least max(4, 20%) of the class's
captions and has lift above 1.4; the five highest-lift terms are shown. The same test
over three latitude/longitude boxes (share >= 25%, lift > 1.6) feeds the region
paragraph in Section 5.7 and is printed, not drawn.

Redrawn 2026-09-17 for the captions regenerated without climate and demand, at the
thesis figure size (453.6 pt wide, placed 1:1), in Cmr10 with the thesis slate colour.

    python -m scripts.figures.fig_caption_terms
"""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "outputs" / "csv" / "pore_features.csv"
CAPTIONS = ROOT / "outputs" / "captions" / "pore_captions.json"
OUT = ROOT / "thesis" / "figures" / "fig_caption_terms.pdf"

WIDTH_IN = 453.6 / 72.0
INK, GRID, SLATE = "#333333", "#dddddd", "#2c3e50"

CLASSES = ["Mangrove + Industrial", "Industrial + Forest", "Coastal + Solar-Wind Hybrid",
           "Industrial + Arid", "Agricultural + Water", "Dense Urban"]
REGIONS = {"Gulf": (22, 32, 45, 58), "Nordic and Baltic": (58, 71, 4, 32),
           "Southeast Asian tropics": (-10, 20, 95, 130)}

GEN = set("""current component reflecting detected supplied option external strong largely virtually
overwhelmingly modest moderate significant relatively notable substantial limited minimal indicating
suggesting showing while although however therefore combined despite typical likely future potential
suitable available present absent reported value values area areas cell square metre data figure model
regression estimate estimates annual daily total mean about approximately there their this that with
from which rather than more most very high good people person year zero essentially accounting
would could should yields requirement requirements assets site sites development developments
worldcover characterized existing identified primarily needs profile intensity fraction footprint
network generate consumption versus bright""".split())
GEN |= set("""beyond basic nearly consistent effectively moderately implies imply implying indicates indicate
suggests suggest shows reflects trace traces""".split())

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8.5, "axes.edgecolor": INK, "axes.linewidth": 0.6,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.5, "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})


def key(lat, lon):
    return f"{round(float(lat), 6)},{round(float(lon), 6)}"


def tokens(text):
    text = text.lower().replace(chr(8209), "-")
    return {w for w in re.findall(r"[a-z][a-z-]{3,}", text) if w not in GEN}


def load():
    caps = json.load(open(CAPTIONS))
    docs = []
    with FEATURES.open() as fh:
        for r in csv.DictReader(fh):
            c = caps.get(key(r["lat"], r["lon"]))
            if c:
                docs.append((r["stratum_name"], float(r["lat"]), float(r["lon"]), tokens(c)))
    return docs


def distinctive(group, glob, n_all, min_share, min_abs, lift_min, top):
    n = len(group)
    c = Counter()
    for t in group:
        c.update(t)
    scored = [(k / n / (glob[w] / n_all), w, 100 * k / n)
              for w, k in c.items() if k >= max(min_abs, min_share * n) and glob[w] > 0]
    scored = [s for s in scored if s[0] > lift_min]
    scored.sort(reverse=True)
    return n, scored[:top]


def main():
    docs = load()
    glob = Counter()
    for _, _, _, t in docs:
        glob.update(t)
    n_all = len(docs)
    by_class = defaultdict(list)
    for cls, _, _, t in docs:
        by_class[cls].append(t)

    print(f"captions analysed: {n_all}")
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH_IN, 5.3))
    for ax, cls in zip(axes.ravel(), CLASSES):
        n, top = distinctive(by_class[cls], glob, n_all, 0.2, 4, 1.4, 5)
        print(f"  {cls} (n={n}): " + ", ".join(f"{w} {p:.0f}% (x{l:.1f})" for l, w, p in top))
        top = top[::-1]
        ax.barh([w for _, w, _ in top], [p for _, _, p in top], height=0.62, color=SLATE, linewidth=0)
        ax.set_title(f"{cls} (n = {n:,})", loc="left", pad=4, x=-0.36)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(True, axis="x")
        ax.tick_params(length=2, width=0.6)
    fig.supxlabel("% of the class's captions containing the term", fontsize=8.5, y=0.01)
    fig.tight_layout(pad=0.4, h_pad=1.2, w_pad=2.5, rect=(0, 0.03, 1, 1))
    fig.savefig(OUT)
    print(f"wrote {OUT}")

    print("regions (share >= 25%, lift > 1.6):")
    for name, (la1, la2, lo1, lo2) in REGIONS.items():
        group = [t for _, la, lo, t in docs if la1 <= la <= la2 and lo1 <= lo <= lo2]
        n, top = distinctive(group, glob, n_all, 0.25, 0, 1.6, 10)
        print(f"  {name} (n={n}): " + (", ".join(f"{w} {p:.0f}% (x{l:.1f})" for l, w, p in top) or "no term clears the bar"))


if __name__ == "__main__":
    main()
