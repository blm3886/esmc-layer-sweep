# ESM C layer sweep: which layer keeps OrthoFinder's information?

Scripts from the EmbedTree project (Todd Lab) testing every layer (0–35) of the
ESM C 600M protein language model against OrthoFinder gene families and gene trees.

| script | what it does |
|---|---|
| `scripts/search_test_layers_crops.py` | **Test 1, search:** for every layer, are a gene's n−1 nearest neighbours (angular distance) in its OrthoFinder family? Score: recall@(n−1). HPI crop panel, 15 species, full proteomes |
| `scripts/layer_sweep_knn_trees.py` | **Search + tree:** for every layer, take a gene + its 2n nearest neighbours, build a neighbour-joining tree, and check whether OrthoFinder's family is one branch (clade) with OrthoFinder's branching order (topology). Two datasets: 86 plant species, 15 crop species |
| `scripts/plot_test1_search.py` | Plot test 1: recall@(n−1) vs layer, one line per dataset |
| `scripts/plot_test1_by_size.py` | Plot test 1 by family size: layer 33 vs the last layer (dumbbell chart) |

## Main result

Search peaks at **layers 33–34** (recall@(n−1) 0.546 on 86 species, 0.526 on 15
crops); tree topology peaks at **layers 3–5**. The conventional last layer (35) is
best at neither.

## Note

These scripts are shared to read, not to run on their own. They import helpers
from the full EmbedTree code (`embedtree.embeddings`, `embedtree.newick`,
`embedtree.nj_trees`, `train_distance.clades_of_tree`), and they read ESM C
embeddings and OrthoFinder results from paths on our server. No data is included.
