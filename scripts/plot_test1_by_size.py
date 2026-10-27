#!/usr/bin/env python3
"""
Plot test 1 (search) by OG size: does the last layer drop in every family size?

Dumbbell chart, one panel per dataset (86 species | 15 crops):
    x = OG size bin (4-10, 11-25, 26-50, 51-100)
    y = recall@(n-1)
    per bin: layer 33 (green) joined to layer 35, the last layer (amber);
             layer 34 as a small grey dot; the drop written next to the line

Reads (same files as plot_test1_search.py):
    results/2026-10-24_search_test_layers/summary.csv + per_query.csv   (86 species)
    combined_run/search_test_crops/summary.csv + per_query.csv          (15 crops; skipped if not there yet)

    python scripts/plot_test1_by_size.py

Output: results/figures/test1_search_by_size.png and .svg
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                  # no screen: write files only
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent
DIR_86 = CODE / "results" / "2026-10-24_search_test_layers"
DIR_15 = Path("/data2/bfernando/embedtree/combined_run/search_test_crops")
OUT = CODE / "results" / "figures" / "test1_search_by_size"

FOREST, LEAF, AMBER = "#1F4E3D", "#5E9C6E", "#D98E1F"  # deck colours
INK, MUTED, GRID = "#222222", "#6B6B6B", "#E3E8E5"

BINS = ["4-10", "11-25", "26-50", "51-100"]
BEST, NEXT, LAST = 33, 34, 35


# ===========================================================================
# STEP 1 - per dataset: {(layer, bin): recall@(n-1)} and families per bin
# ===========================================================================

def read_dataset(folder, keep):
    recall = {}
    with open(folder / "summary.csv") as handle:
        for row in csv.DictReader(handle):
            if keep(row):
                recall[(int(row["layer"]), row["bin"])] = float(row["mean_recall_n"])

    families = defaultdict(set)                        # bin -> the families in it (from one layer)
    with open(folder / "per_query.csv") as handle:
        for row in csv.DictReader(handle):
            if int(row["layer"]) == BEST and keep(row):
                families[row["bin"]].add(row["family"])
    return recall, {b: len(f) for b, f in families.items()}


datasets = [("All plants (86 species)", FOREST,
             *read_dataset(DIR_86, lambda r: r.get("normalisation") == "centre_pcs"
                                             and r.get("query_type") == "every"))]
if (DIR_15 / "summary.csv").exists():
    datasets.append(("HPI crop panel (15 species)", LEAF, *read_dataset(DIR_15, lambda r: True)))


# ===========================================================================
# STEP 2 - the plot: one panel per dataset
# ===========================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

for ax, (name, colour, recall, families) in zip(axes, datasets):
    for x, b in enumerate(BINS):
        best, nxt, last = recall[(BEST, b)], recall[(NEXT, b)], recall[(LAST, b)]
        ax.plot([x, x], [last, best], color=MUTED, linewidth=2, zorder=1)        # the drop
        ax.plot(x, best, "o", color=colour, markersize=11, zorder=3)
        ax.plot(x + 0.12, nxt, "o", color="#A9B4AE", markersize=6, zorder=2)
        ax.plot(x, last, "o", color=AMBER, markersize=11, zorder=3)
        nudge = 0.015 if best - last < 0.03 else 0.0          # pull the two labels apart when the dots are close
        ax.text(x - 0.12, best + nudge, f"{best:.3f}", ha="right", va="center", fontsize=10, color=INK)
        ax.text(x - 0.12, last - nudge, f"{last:.3f}", ha="right", va="center", fontsize=10, color=INK)
        ax.text(x + 0.12, (best + last) / 2, f"−{best - last:.3f}", ha="left", va="center",
                fontsize=10.5, color=AMBER, fontweight="bold")

    ax.set_xticks(range(len(BINS)))
    ax.set_xticklabels([f"{b} genes\n({families.get(b, 0)} OGs)" for b in BINS], fontsize=10.5, color=INK)
    ax.set_xlim(-0.6, len(BINS) - 0.4)
    ax.set_title(name, fontsize=12.5, color=colour, fontweight="bold", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)

if len(datasets) == 1:                                 # crop run not finished yet
    axes[1].text(0.5, 0.5, "HPI crop panel (15 species)\nrun in progress", ha="center", va="center",
                 transform=axes[1].transAxes, fontsize=12, color=MUTED)
    axes[1].set_axis_off()

axes[0].set_ylim(0, 0.8)
axes[0].set_ylabel("recall@(n−1)\nfraction of nearest neighbours in its OG", fontsize=12, color=INK,
                   linespacing=1.6)
axes[0].set_xlabel("OG size", fontsize=12, color=INK)

# legend in words, under the title
for marker, colour, size, text in [("o", FOREST, 9, f"layer {BEST}"), ("o", "#A9B4AE", 6, f"layer {NEXT}"),
                                   ("o", AMBER, 9, f"layer {LAST} (last)")]:
    axes[0].plot([], [], marker, color=colour, markersize=size, label=text, linestyle="none")
axes[0].legend(loc="upper left", frameon=False, fontsize=10, ncol=3)

fig.suptitle("Test 1 by OG size: the last layer drops in every size", fontsize=13.5, color=INK,
             x=0.01, ha="left")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
print("saved", OUT.with_suffix(".png"), OUT.with_suffix(".svg"))
