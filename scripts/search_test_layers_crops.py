#!/usr/bin/env python3
"""
Test 1 (search) on the HPI crop panel, for every ESM C layer: are a gene's
n-1 nearest neighbours its family?

The same test as search_test_layers.py (86 species), on a second dataset:
    pool      full proteomes of Nolan's 15 Helixer species (658,005 proteins,
              every gene; other families and unassigned genes are distractors)
    families  the same 493 as layer_sweep_knn_trees.py hpi_crops_15
    queries   EVERY gene of each family
    answer    NOLAN'S OrthoFinder run (orthogroups)
    score     recall@(n-1) = fraction of the first n-1 hits that are family members
              (n = family size in the pool); perfect = all n-1 are members

No training: plain ESM C, centre + 2PCs (fitted on 20,000 random pool proteins).
CPU only, 8 threads.

    python scripts/search_test_layers_crops.py

Output: combined_run/search_test_crops/per_query.csv and summary.csv
"""

import os

MAX_THREADS = "8"                          # seabiscuit is shared: never take all 64 cores
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, MAX_THREADS)

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
torch.set_num_threads(int(MAX_THREADS))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embedtree.embeddings import fit_normalizer, read_orthogroups
from embedtree.newick import load_tree_collection


# ===========================================================================
# SETTINGS
# ===========================================================================

OUR_RUN = Path("/data2/bfernando/orthofinder_genetree_plants/orthofinder_out/Results_Sep28"
               "/WorkingDirectory/OrthoFinder/Results_Oct07")
NOLAN_RUN = Path("/data1/bfernando/vector_db/helixer_small.orthofinder")
EMBEDDING_DIR = Path("/data1/bfernando/embedtree/emb/esmc_600m")
RUN_DIR = Path("/data2/bfernando/embedtree/combined_run")
SPLIT_FILE = RUN_DIR / "split.json"
OUT_DIR = RUN_DIR / "search_test_crops"

CROP_SPECIES = [                              # our names of Nolan's 15 Helixer species
    "Arabidopsis_thaliana", "Brassica_napus", "Raphanus_sativus", "Thlaspi_arvense",
    "Gossypium_hirsutum", "Glycine_max", "Lotus_japonicus", "Trifolium_pratense",
    "Amborella_trichopoda", "Brachypodium_distachyon", "Oryza_sativa", "Setaria_italica",
    "Sorghum_bicolor", "Zea_mays", "Wolffia_australiana",
]

LAYERS = range(36)
NORMALISER_PROTEINS = 20_000
SEED = 0
SIZE_BINS = [(4, 10), (11, 25), (26, 50), (51, 100)]


# finds which size group a family is in
def size_bin(n):
    for lo, hi in SIZE_BINS:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return "other"


# Nolan's Orthogroups.tsv -> {protein: orthogroup}
def read_orthogroup_table(path):
    family = {}
    with open(path) as handle:
        next(handle)
        for line in handle:
            cells = line.rstrip("\n").split("\t")
            for cell in cells[1:]:
                for pid in cell.split(","):
                    if pid.strip():
                        family[pid.strip()] = cells[0]
    return family


# protein names stored in one embedding file, in file order
def read_ids(path):
    with h5py.File(path, "r") as handle:
        return [p.decode() if isinstance(p, bytes) else p for p in handle["protein_id"][:]]


# ===========================================================================
# STEP 1 - the pool: every protein of the 15 species, all 36 layers
# ===========================================================================

og_of, _ = read_orthogroups(OUR_RUN / "Orthogroups" / "Orthogroups.tsv")
split = json.loads(SPLIT_FILE.read_text())
family_of = read_orthogroup_table(NOLAN_RUN / "Orthogroups" / "Orthogroups.tsv")
trees = load_tree_collection(NOLAN_RUN / "Resolved_Gene_Trees")   # only to pick the same families
files = [EMBEDDING_DIR / f"{species}.h5" for species in CROP_SPECIES]

# First only the ids, so the array for all vectors can be made once (float16: ~55 GB)
ids = []
for path in files:
    ids += read_ids(path)

pool = np.zeros((len(ids), 36, 1152), dtype=np.float16)
start = 0
for path in files:
    with h5py.File(path, "r") as handle:
        block = handle["pooled"][:]
    pool[start:start + len(block)] = block
    start += len(block)
    print(f"  read {path.stem}: {len(block)} proteins", flush=True)
del block
print(f"pool of {len(ids)} proteins", flush=True)


# ===========================================================================
# STEP 2 - the families (same rule as layer_sweep_knn_trees.py) and the queries
# ===========================================================================

members = defaultdict(list)                       # family -> its rows in the pool
for row, pid in enumerate(ids):
    if pid in family_of:
        members[family_of[pid]].append(row)

seen_by_model = set(split["train"]) | set(split["validation"])
test_families = set(split["test"])
families = []
for fam, rows in members.items():
    ours = {og_of.get(ids[r]) for r in rows}      # our families of these genes
    if 4 <= len(rows) <= 100 and fam in trees and not ours & seen_by_model and ours & test_families:
        families.append(fam)

queries = []                                      # (family, query row): EVERY gene of each family
for fam in families:
    queries += [(fam, row) for row in members[fam]]
print(f"{len(families)} families, {len(queries)} queries", flush=True)

rng = np.random.default_rng(SEED)
fit_rows = np.sort(rng.choice(len(ids), size=min(NORMALISER_PROTEINS, len(ids)), replace=False))


# ===========================================================================
# STEP 3 - search, layer by layer
# ===========================================================================

query_rows = torch.tensor([row for _, row in queries])
n_hits = [len(members[fam]) - 1 for fam, _ in queries]      # n - 1 per query
results = []
for layer in LAYERS:
    normaliser = fit_normalizer(pool[fit_rows, layer, :].astype(np.float32), "centre_pcs")
    vectors = np.zeros((len(ids), 1152), dtype=np.float32)
    for s in range(0, len(ids), 50_000):                      # in blocks, to keep memory low
        vectors[s:s + 50_000] = normaliser.transform(pool[s:s + 50_000, layer, :])
    z = torch.from_numpy(vectors)

    for s in range(0, len(queries), 500):                     # 500 queries at a time
        q = query_rows[s:s + 500]
        similarity = z[q] @ z.T                               # cosine to every protein (same order as angular)
        similarity[torch.arange(len(q)), q] = -2.0            # never pick the query itself
        top = torch.topk(similarity, max(n_hits[s:s + 500]), dim=1).indices.numpy()

        for k in range(len(q)):
            fam, row = queries[s + k]
            hits = top[k, :n_hits[s + k]]                     # the first n - 1 hits
            in_family = [family_of.get(ids[h]) == fam for h in hits]
            recall = sum(in_family) / len(hits)
            results.append({"layer": layer, "family": fam, "size": len(members[fam]),
                            "bin": size_bin(len(members[fam])), "query": ids[row],
                            "recall_n": recall, "perfect": float(recall == 1.0)})
    print(f"layer {layer:>2} done", flush=True)


# ===========================================================================
# STEP 4 - CSV files and a short overview
# ===========================================================================

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / "per_query.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)

groups = defaultdict(list)
for r in results:
    groups[(r["layer"], "all")].append(r)
    groups[(r["layer"], r["bin"])].append(r)

summary = []
for (layer, b), rs in sorted(groups.items()):
    summary.append({"dataset": "hpi_crops_15", "layer": layer, "bin": b, "queries": len(rs),
                    "mean_recall_n": round(float(np.mean([r["recall_n"] for r in rs])), 4),
                    "pct_perfect": round(100 * float(np.mean([r["perfect"] for r in rs])), 2)})
with open(OUT_DIR / "summary.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

print(f"\n{'layer':>5}{'recall@n-1':>12}{'perfect':>9}")
for s in summary:
    if s["bin"] == "all":
        print(f"{s['layer']:>5}{s['mean_recall_n']:>12.3f}{s['pct_perfect']:>8.1f}%")
print("saved", OUT_DIR / "per_query.csv", OUT_DIR / "summary.csv")
