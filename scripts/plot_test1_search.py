#!/usr/bin/env python3
"""
Plot test 1 (search): recall@(n-1) for every ESM C layer, 86 species and 15 crops.

    x = layer 0-35
    y = recall@(n-1): fraction of a gene's n-1 nearest neighbours that are its family
    one line per dataset; dot + label on each peak; amber dot on the last layer

Reads:
    results/2026-10-24_search_test_layers/summary.csv     (86 species)
    combined_run/search_test_crops/summary.csv            (15 crops; skipped if not there yet)

    python scripts/plot_test1_search.py

Output: results/figures/test1_search_layers.png and .svg
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                  # no screen: write files only
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent
FILE_86 = CODE / "results" / "2026-10-24_search_test_layers" / "summary.csv"
FILE_15 = Path("/data2/bfernando/embedtree/combined_run/search_test_crops/summary.csv")
OUT = CODE / "results" / "figures" / "test1_search_layers"

FOREST, LEAF, AMBER = "#1F4E3D", "#5E9C6E", "#D98E1F"  # deck colours
INK, MUTED, GRID = "#222222", "#6B6B6B", "#E3E8E5"


# ===========================================================================
# STEP 1 - read one curve per dataset: {layer: recall@(n-1)}
# ===========================================================================

def read_curve(path, keep):
    curve = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            if row["bin"] == "all" and keep(row):
                curve[int(row["layer"])] = float(row["mean_recall_n"])
    return dict(sorted(curve.items()))


curves = [("All plants (86 species)", "-", FOREST,
           read_curve(FILE_86, lambda r: r["normalisation"] == "centre_pcs" and r["query_type"] == "every"))]
if FILE_15.exists():
    curves.append(("HPI crop panel (15 species)", "--", LEAF, read_curve(FILE_15, lambda r: True)))


# ===========================================================================
# STEP 2 - the plot
# ===========================================================================

fig, ax = plt.subplots(figsize=(9, 5))

MARKED = [32, 33, 34]                                  # layers around the peak, shown with their values
for layer in MARKED:
    ax.axvline(layer, color=MUTED, linewidth=0.8, linestyle=":", zorder=0)

for i, (name, style, colour, curve) in enumerate(curves):
    layers, values = list(curve), list(curve.values())
    ax.plot(layers, values, style, color=colour, linewidth=2.2)

    for layer in MARKED:                               # dot on each marked layer
        ax.plot(layer, curve[layer], "o", color=colour, markersize=6)
    # the marked layers' values, stacked in a small block left of the lines
    # (first dataset above its line, second below, so the blocks never overlap)
    values_text = "\n".join(f"layer {layer}   {curve[layer]:.3f}" for layer in MARKED)
    above = i == 0
    ax.text(MARKED[0] - 0.6, curve[MARKED[1]] + (0.03 if above else -0.075), values_text,
            ha="right", va="bottom" if above else "top", fontsize=10.5, color=colour,
            linespacing=1.4, family="monospace")

    last = layers[-1]                                  # the last layer (the convention)
    ax.plot(last, curve[last], "o", color=AMBER, markersize=9, zorder=5)
    ax.annotate(f"{curve[last]:.3f}", (last, curve[last]), xytext=(8, 4 if above else -12),
                textcoords="offset points", fontsize=11, color=INK)

    ax.annotate(name, (last, curve[last]), xytext=(42, 4 if above else -12), textcoords="offset points",
                fontsize=11, color=colour, fontweight="bold")   # label at the line's end


ax.set_xlim(-0.5, 35.5)
ax.set_ylim(0, 0.7)
ax.set_xticks(range(0, 36, 5))
ax.set_xlabel("ESM C layer", fontsize=12, color=INK)
ax.set_ylabel("recall@(n−1)\nfraction of nearest neighbours in its OG", fontsize=12, color=INK,
              linespacing=1.6)
ax.set_title("Test 1: are a gene's n−1 nearest neighbours in its OrthoFinder family (OG)?",
             fontsize=13, color=INK, loc="left")
ax.grid(axis="y", color=GRID, linewidth=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
ax.tick_params(colors=MUTED)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
print("saved", OUT.with_suffix(".png"), OUT.with_suffix(".svg"))
