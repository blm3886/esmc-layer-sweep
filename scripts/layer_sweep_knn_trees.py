#!/usr/bin/env python3
"""
Layer sweep, search + tree together: 

From the ESM C find which layer finds the best gene's
family AND its OrthoFinder tree.

For every layer 0-35, for each query gene (selecting 5 genes per family randomly):
    1. SEARCH  take the query + its 2n nearest proteins (angular distance)
               from the whole pool; n = size of the query's orthogroup
    2. TREE    neighbour-joining on those 2n + 1 proteins (embedding distances only)
    3. CHECK   is OrthoFinder's tree of the family inside our tree?
                 found     = family members among the 2n + 1  /  n
                 clade     = 1 if the found members form ONE branch (no outsider inside)
                 topology  = our tree pruned to the found members vs OrthoFinder's
                             tree pruned to the same members (1 - nRF)
                 full      = 1 if found = 1, clade = 1 and topology = 1

Datasets:
( will run this script for two datasets)
    all_plants_86   the 80,056-protein pool
                        all 86 species	2,719,956	100%
                        in orthogroups	2,574,151	94.6%
                        this pool	80,056	~3% (these proteins are in Orthofgroups which have 4-100 nodes)
    
                    of the earlier sweeps (86 species),
                    542 validation families, answer key = OUR OrthoFinder run
    hpi_crops_15    full proteomes of Nolan's 15 Helixer species (~658k proteins),
                    490 families (same as panel_test), answer key = NOLAN'S run

No training: plain ESM C, centre + 2PCs (fitted on 20,000 random pool proteins).
CPU only, 8 threads.

    python scripts/layer_sweep_knn_trees.py all_plants_86
    python scripts/layer_sweep_knn_trees.py hpi_crops_15

Output: combined_run/layer_sweep_knn_trees/<dataset>/per_query.csv and summary.csv
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

from embedtree.embeddings import fit_normalizer, read_orthogroups, sample_families
from embedtree.newick import load_tree_collection, make_leaf_mapper
from embedtree.nj_trees import cuts, nj_groups, project
from train_distance import clades_of_tree


# ===========================================================================
# SETTINGS
# ===========================================================================

OUR_RUN = Path("/data2/bfernando/orthofinder_genetree_plants/orthofinder_out/Results_Sep28"
               "/WorkingDirectory/OrthoFinder/Results_Oct07")
NOLAN_RUN = Path("/data1/bfernando/vector_db/helixer_small.orthofinder")
EMBEDDING_DIR = Path("/data1/bfernando/embedtree/emb/esmc_600m")
RUN_DIR = Path("/data2/bfernando/embedtree/combined_run")
SPLIT_FILE = RUN_DIR / "split.json"

NOLAN_NAMES = {                               # our species name -> name in Nolan's run
    "Arabidopsis_thaliana": "Athaliana.Col-0.HPIv02", "Brassica_napus": "Bnapus.Westar.HPIv3",
    "Raphanus_sativus": "Rsativus.Okute-Sakurajima.HPIv3", "Thlaspi_arvense": "Tarvense.Ta1014.HPIv3",
    "Gossypium_hirsutum": "Ghirsutum.TM-1.HPIv3", "Glycine_max": "Gmax.wm82a5.HPIv3",
    "Lotus_japonicus": "Ljaponica.Gifu.HPIv3", "Trifolium_pratense": "Tpratense.Milvus_B.HPIv3",
    "Amborella_trichopoda": "Atrichopoda.UCSC.HPIv02", "Brachypodium_distachyon": "Bdistachyon.Bd21.HPIv3",
    "Oryza_sativa": "Osativa.NipponBare.HPIv3", "Setaria_italica": "Sitalica.Yugu1.HPIv3",
    "Sorghum_bicolor": "Sbicolor.BTx623.HPIv3", "Zea_mays": "Zmays.B73v5.HPIv3",
    "Wolffia_australiana": "Waustraliana.8730.HPIv1",
}

LAYERS = range(36)
QUERIES_PER_FAMILY = 5
NORMALISER_PROTEINS = 20_000
TARGET_PROTEINS = 80_000                      # the 80,056-protein pool of the earlier sweeps
SEED = 0
SIZE_BINS = [(4, 10), (11, 25), (26, 50), (51, 100)]

DATASET = sys.argv[1] if len(sys.argv) > 1 else "all_plants_86"
OUT_DIR = RUN_DIR / "layer_sweep_knn_trees" / DATASET

# finds which size group a family is in
def size_bin(n):
    for lo, hi in SIZE_BINS:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return "other"

# returns family every protein is in
# input file : 
#Orthogroup   Athaliana...           Zmays...
#OG0000001    AT1G01010, AT1G01020   Zm00001eb000010

# returns : 
#{"AT1G01010": "OG0000001", "AT1G01020": "OG0000001", "Zm00001eb000010": "OG0000001", ...}

def read_orthogroup_table(path):
    """Nolan's Orthogroups.tsv -> protein -> orthogroup."""
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

# return an array of protein names from emebdding files,
# ["Zm00001eb000010", "Zm00001eb000020",
def read_ids(path):
    with h5py.File(path, "r") as handle:
        return [p.decode() if isinstance(p, bytes) else p for p in handle["protein_id"][:]]


# ===========================================================================
# STEP 1 - the pool: protein ids, their family, all 36 layers
# ===========================================================================

og_of, sp_of = read_orthogroups(OUR_RUN / "Orthogroups" / "Orthogroups.tsv")
split = json.loads(SPLIT_FILE.read_text())

if DATASET == "all_plants_86":
    sampled = sample_families(og_of, target_proteins=TARGET_PROTEINS, seed=SEED)
    family_of = og_of
    trees = load_tree_collection(OUR_RUN / "Resolved_Gene_Trees" / "Resolved_Gene_Trees.txt")
    #cleans tree names
    strip = make_leaf_mapper(set(sp_of.values()))
    files = sorted(EMBEDDING_DIR.glob("*.h5"))
    #keep this protein if family was sampled
    keep_protein = lambda pid: og_of.get(pid) in sampled
elif DATASET == "hpi_crops_15":
    family_of = read_orthogroup_table(NOLAN_RUN / "Orthogroups" / "Orthogroups.tsv")
    trees = load_tree_collection(NOLAN_RUN / "Resolved_Gene_Trees")
    #cleans tree names
    strip = make_leaf_mapper([n.replace(".", "_") for n in NOLAN_NAMES.values()])
    files = [EMBEDDING_DIR / f"{species}.h5" for species in NOLAN_NAMES]
    #keep everything
    keep_protein = lambda pid: True               # full proteomes
else:
    sys.exit(f"unknown dataset {DATASET!r}: use all_plants_86 or hpi_crops_15")

# First only the ids, so the array for all vectors can be made once.
# makes a list of all protein ids we will be using
# ids = ["AT2", "AT4", "ZM1"]
# rows_in_file = [[1, 3], [0]] --> per species row indices
# which is like rows 1 and 3 from Arabidopsis.h5 and 0 from Zea_mays.h5
ids, rows_in_file = [], []
for path in files:
    stored = read_ids(path)
    wanted = [i for i, pid in enumerate(stored) if keep_protein(pid)]
    rows_in_file.append(wanted)
    ids += [stored[i] for i in wanted]

pool = np.zeros((len(ids), 36, 1152), dtype=np.float16) # [proteins × 36 layers × 1152 numbers] float16
start = 0
for path, wanted in zip(files, rows_in_file):
    if wanted:
        with h5py.File(path, "r") as handle:
            pool[start:start + len(wanted)] = handle["pooled"][np.array(wanted), :, :]
    start += len(wanted)
    print(f"  read {path.stem}: {len(wanted)} proteins", flush=True)
print(f"{DATASET}: pool of {len(ids)} proteins", flush=True)

#returns : 
# pool row 0 = AT2's vectors   ← ids[0] = "AT2"
# pool row 1 = AT4's vectors   ← ids[1] = "AT4"
# pool row 2 = ZM1's vectors   ← ids[2] = "ZM1"

# ===========================================================================
# STEP 2 - which families are tested, and which genes are the queries
# ===========================================================================

#ids       = ["AT2", "AT4", "ZM1", "AT7", "ZM5"]
#family_of = AT2→OG1, AT4→OG1, ZM1→OG1, AT7→OG2, ZM5→OG2
#members   = {"OG1": [0, 1, 2], "OG2": [3, 4]}
members = defaultdict(list)                       # family -> its rows in the pool
for row, pid in enumerate(ids):
    if pid in family_of:
        members[family_of[pid]].append(row)
#extracting required OGs
if DATASET == "all_plants_86":
    families = [fam for fam in split["validation"] if fam in members]
else:                                             # same 490 families as panel_test
    seen_by_model = set(split["train"]) | set(split["validation"])
    test_families = set(split["test"])
    families = []
    for fam, rows in members.items():
        ours = {og_of.get(ids[r]) for r in rows}  # our families of these genes
        if 4 <= len(rows) <= 100 and fam in trees and not ours & seen_by_model and ours & test_families:
            families.append(fam)

rng = np.random.default_rng(SEED)
queries = []                                      # (family, query row)
for fam in families:
    rows = members[fam]
    picked = rng.choice(rows, size=min(QUERIES_PER_FAMILY, len(rows)), replace=False)
    queries += [(fam, int(r)) for r in picked]
print(f"{len(families)} families, {len(queries)} queries", flush=True)

fit_rows = np.sort(rng.choice(len(ids), size=min(NORMALISER_PROTEINS, len(ids)), replace=False))


# ===========================================================================
# STEP 3 - one query at one layer: search, tree, check
# ===========================================================================

def check_one_query(fam, query_row, hits, vectors):
    """hits = the query's 2n nearest rows. Returns found / clade / topology / full."""
    n = len(members[fam])
    leaves = [query_row] + list(hits)                         # the query is leaf 0
    in_family = [family_of.get(ids[r]) == fam for r in leaves]
    member_positions = [i for i, yes in enumerate(in_family) if yes]
    outsider_positions = [i for i, yes in enumerate(in_family) if not yes]
    found = len(member_positions) / n

    # Our tree: NJ on the embeddings of the 2n + 1 leaves, as a set of cuts (branches)
    our_cuts = cuts(nj_groups(vectors[leaves]), len(leaves))

    # Too few members found to say anything about a branch or its shape
    if len(member_positions) < 4:
        return {"found": found, "clade": float("nan"), "topology": float("nan"), "full": 0.0}

    # Clade: the cut {outsiders | members} is one of our branches.
    # (cuts() writes each cut as the side without leaf 0; leaf 0 is the query, a member)
    clade = float(frozenset(outsider_positions) in our_cuts)

    # Topology: prune both trees to the found members, compare their branches
    keep = frozenset(member_positions)
    renumber = {p: i for i, p in enumerate(member_positions)}
    ours_on_members = cuts([frozenset(renumber[p] for p in side) for side in project(our_cuts, keep)],
                           len(member_positions))
    position_of = {ids[leaves[p]]: i for i, p in enumerate(member_positions)}
    truth = cuts(clades_of_tree(trees[fam], position_of, strip), len(member_positions))
    topology = len(truth & ours_on_members) / len(truth) if truth else float("nan")

    full = float(found == 1.0 and clade == 1.0 and topology == 1.0)
    return {"found": found, "clade": clade, "topology": topology, "full": full}


# ===========================================================================
# STEP 4 - every layer
# ===========================================================================

query_rows = torch.tensor([row for _, row in queries])
n_hits = [2 * len(members[fam]) for fam, _ in queries]
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
            scores = check_one_query(fam, row, top[k, :n_hits[s + k]], vectors)
            results.append({"layer": layer, "family": fam, "size": len(members[fam]),
                            "bin": size_bin(len(members[fam])), "query": ids[row], **scores})
    print(f"layer {layer:>2} done", flush=True)


# ===========================================================================
# STEP 5 - CSV files and a short overview
# ===========================================================================

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / "per_query.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)


def mean_of(rs, key):
    values = [r[key] for r in rs if r[key] == r[key]]         # skip NaN
    return round(float(np.mean(values)), 4) if values else float("nan")


groups = defaultdict(list)
for r in results:
    groups[(r["layer"], "all")].append(r)
    groups[(r["layer"], r["bin"])].append(r)

summary = []
for (layer, b), rs in sorted(groups.items()):
    summary.append({"dataset": DATASET, "layer": layer, "bin": b, "queries": len(rs),
                    "found": mean_of(rs, "found"), "clade": mean_of(rs, "clade"),
                    "topology": mean_of(rs, "topology"), "full": mean_of(rs, "full"),
                    "queries_with_4_found": sum(r["clade"] == r["clade"] for r in rs)})
with open(OUT_DIR / "summary.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

print(f"\n{'layer':>5}{'found':>8}{'clade':>8}{'topology':>10}{'full':>8}")
for s in summary:
    if s["bin"] == "all":
        print(f"{s['layer']:>5}{s['found']:>8.3f}{s['clade']:>8.3f}{s['topology']:>10.3f}{s['full']:>8.3f}")
print("saved", OUT_DIR / "per_query.csv", OUT_DIR / "summary.csv")
