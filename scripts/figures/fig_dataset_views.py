"""Section 5.5 figures -- the completed 10,000-coordinate dataset, one view per figure.

Five PDFs, each drawn 453.6 pt wide and placed at 0.85\textwidth, the same size,
fonts (Cmr10), palette and grid as the other validation figures. The sixth view,
the climate regime split, is fig_demand_regime.pdf and is not redrawn here.

    python -m scripts.figures.fig_dataset_views
"""
import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "outputs" / "csv" / "pore_features.csv"
FIGURES = ROOT / "thesis" / "figures"

WIDTH_IN = 453.6 / 72.0

INK   = "#333333"
GRID  = "#dddddd"
SLATE = "#2c3e50"
BROWN = "#8b5a3c"
BLUE  = "#8fa3b0"

# Tier colours as in fig_tier_allocation.pdf (Figure 3.3).
TIER_COLOUR = {"HIGH": SLATE, "MID": BROWN, "LOW": BLUE}

# Order and tiers as in Table 3.1 (tab:meth-tiers). The CSV's importance_tier column
# predates the tier reassignment and is deliberately not used.
TIERS = [
    ("HIGH", ["Suburban", "Dense Urban", "Informal + Urban", "Industrial"]),
    ("MID",  ["Suburban + Agricultural", "Industrial + Water", "Hydropower Reservoir",
              "Agriculture / Agrivoltaics", "Industrial + Arid", "Industrial + Forest",
              "Urban + Coastal"]),
    ("LOW",  ["Agricultural + Water", "Mangrove + Industrial", "Data Centre + Industrial",
              "Coastal + Solar-Wind Hybrid", "Coastal + Agricultural"]),
]

mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["cmr10"],
    "mathtext.fontset":   "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size":          11,
    "axes.titlesize":     11.5,
    "axes.labelsize":     11,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "axes.edgecolor":     INK,
    "axes.linewidth":     0.8,
    "xtick.color":        INK,
    "ytick.color":        INK,
    "text.color":         INK,
    "axes.labelcolor":    INK,
    "grid.color":         GRID,
    "grid.linewidth":     0.8,
    "figure.facecolor":   "white",
    "savefig.facecolor":  "white",
    "pdf.fonttype":       42,
})


def load():
    with FEATURES.open() as fh:
        return list(csv.DictReader(fh))


def house(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True)


def save(fig, name):
    out = FIGURES / name
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_classes(rows):
    counts = Counter(r["stratum_name"] for r in rows)
    names, vals, cols, groups = [], [], [], []
    for tier, members in TIERS:
        start = len(names)
        for m in members:
            names.append(m)
            vals.append(counts[m])
            cols.append(TIER_COLOUR[tier])
        groups.append((tier, start, len(names) - 1))
    y = np.arange(len(names))[::-1]

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 4.35))
    ax.barh(y, vals, height=0.7, color=cols, linewidth=0)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlim(0, 2450)
    ax.set_xlabel("coordinates")
    ax.set_title("Coordinates per class across the 10,000-coordinate sample")
    house(ax)
    ax.grid(False, axis="y")
    for tier, lo, hi in groups:
        val = vals[lo]
        ax.annotate(f"{tier}, {val:,} each", xy=(val + 50, (y[lo] + y[hi]) / 2),
                    va="center", ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_dataset_classes.pdf")


def fig_landcover(rows):
    counts = Counter(r["wc_dominant_class"] for r in rows)
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    names = [k.replace("_", " ") for k, _ in items]
    vals = [v for _, v in items]
    y = np.arange(len(names))[::-1]

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.35))
    ax.barh(y, vals, height=0.65, color=SLATE, linewidth=0)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 8400)
    ax.set_xlabel("coordinates")
    ax.set_title("Dominant land-cover class of each coordinate's tile")
    house(ax)
    ax.grid(False, axis="y")
    for val, yy in zip(vals, y):
        ax.annotate(f"{val:,}", xy=(val + 70, yy), va="center", ha="left",
                    fontsize=9.5, color=INK)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_dataset_landcover.pdf")


def fig_map(rows):
    lon = np.array([float(r["lon"]) for r in rows])
    lat = np.array([float(r["lat"]) for r in rows])

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.15))
    ax.scatter(lon, lat, s=2.0, c=SLATE, alpha=0.30, linewidths=0, rasterized=True)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 82)
    ax.set_aspect("equal")
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-40, 0, 40, 80])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Location of all 10,000 coordinates")
    house(ax)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_dataset_map.pdf")


def fig_builtup(rows):
    vals = np.array([float(r["wc_built_up_pct"]) for r in rows])
    above = 100 * float((vals >= 60).sum()) / vals.size

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.2))
    ax.hist(vals, bins=np.arange(0, 102.5, 2.5), color=SLATE, edgecolor="white",
            linewidth=0.6)
    ax.axvline(60, color=BROWN, linewidth=1.2, linestyle="--")
    ax.annotate(f"{above:.0f}% of tiles\nabove 60%", xy=(61.5, 0.93),
                xycoords=("data", "axes fraction"), ha="left", va="top",
                fontsize=9.5, color=INK, linespacing=1.3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("built-up share of the tile (%)")
    ax.set_ylabel("coordinates")
    ax.set_title("Built-up share of each coordinate's tile")
    house(ax)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_dataset_builtup.pdf")


def fig_demand(rows):
    def positive(col):
        return np.array([float(r[col]) for r in rows if r[col] and float(r[col]) > 0])

    heat, cool = positive("heating_MWh"), positive("cooling_MWh")
    lo = math.floor(math.log10(min(heat.min(), cool.min())))
    hi = math.ceil(math.log10(max(heat.max(), cool.max())))
    bins = np.logspace(lo, hi, 40)

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.2))
    for vals, colour, label in ((heat, SLATE, "heating"), (cool, BROWN, "cooling")):
        ax.hist(vals, bins=bins, histtype="stepfilled", color=colour, alpha=0.15,
                linewidth=0)
        ax.hist(vals, bins=bins, histtype="step", color=colour, linewidth=1.6,
                label=f"{label} ({vals.size:,} tiles)")
    ax.set_xscale("log")
    ax.set_xlabel("demand per tile (MWh per year, logarithmic axis)")
    ax.set_ylabel("coordinates")
    ax.set_title("Annual heating and cooling demand per tile")
    house(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_dataset_demand.pdf")


def main():
    rows = load()
    fig_classes(rows)
    fig_landcover(rows)
    fig_map(rows)
    fig_builtup(rows)
    fig_demand(rows)


if __name__ == "__main__":
    main()
