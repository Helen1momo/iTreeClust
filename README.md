# iTreeClust

`iTreeClust` builds a fixed-threshold post-hoc explanation tree for biological sequence clustering results.


The intended workflow is:

1. cluster all input sequences into bottom-level flat clusters using a base similarity threshold;
2. use one representative from each bottom cluster as a leaf representative;
3. recluster representatives at user-specified upper-level thresholds;
4. export node summaries, representatives, and parent-child containment relations.


## Installation

The core code uses only the Python standard library. Installing `python-Levenshtein` is recommended for speed.

```bash
pip install -e .
pip install python-Levenshtein
```


External clustering backends are optional but recommended for real data:

- `vsearch` for DNA/RNA sequence clustering;
- `cd-hit-est` for CD-HIT-EST;
- `Rscript` with DECIPHER/Biostrings for Clusterize.

For a quick logic check without external tools, use `--base-method greedy`. 

## Command-line usage

Using the installable command:

```bash
itreeclust \
  --fasta examples/toy.fasta \
  --outdir outputs/toy_tree \
  --prefix toy \
  --base-method greedy \
  --base-threshold 0.97 \
  --n-levels 2 \
  --level-thresholds 0.90,0.75 \
  --taxonomy examples/toy_taxonomy.csv \
  --seq-id-col seq_id \
  --ranks class,order,family,genus,species
```

Without installation, run the wrapper directly:

```bash
python run_itreeclust.py \
  --fasta examples/toy.fasta \
  --outdir outputs/toy_tree \
  --prefix toy \
  --base-method greedy \
  --base-threshold 0.97 \
  --n-levels 2 \
  --level-thresholds 0.90,0.75
```

For a real dataset using VSEARCH:

```bash
itreeclust \
  --fasta /path/to/dataset.fasta \
  --outdir outputs/dataset_tree \
  --prefix dataset_name \
  --base-method vsearch \
  --base-threshold 0.99 \
  --n-levels 3 \
  --level-thresholds 0.95,0.86,0.78 \
  --threads 8 \
  --vsearch-exe vsearch
```

## Threshold semantics

- `--base-threshold` controls the bottom flat clustering. This becomes `level_0`, i.e. the tree leaves.
- `--level-thresholds` controls upper explanation levels. For example, `0.95,0.86,0.78` produces `level_1`, `level_2`, and `level_3`.
- `--n-levels` is optional but useful for reproducibility checks. It counts only upper explanation levels, not the bottom flat-cluster level.
- Thresholds are sorted from high to low internally. A warning is printed if the input order is not descending.

## Outputs

All outputs are written to `--outdir`.

| File | Meaning |
|---|---|
| `run_config.json` | Full command configuration and output descriptions. |
| `tree_level_summary.csv` | One row per level: threshold, number of nodes, node-size statistics. |
| `tree_nodes.csv` | One row per node: level, parent, representative sequence ID/index, children, size. |
| `tree_edges.csv` | Parent-child containment relations between adjacent levels. |
| `tree_level_assignments.csv` | Per-sequence node assignment at every tree level. |
| `tree_node_representatives.fasta` | Representative sequence of every tree node. |
| `node_taxonomy_summary.csv` | Optional dominant taxonomy and purity per node/rank if taxonomy CSV is provided. |

## Input formats

### FASTA

FASTA headers are parsed using the first whitespace-delimited token as the sequence ID.

```text
>seq_001
ACGT...
>seq_002
ACGT...
```


## Python API

```python
from itreeclust.base_algorithms import read_fasta, run_base_algorithm, norm_edit_distance
from itreeclust.posthoc_tree import PostHocTreeBuilder

seq_ids, sequences = read_fasta("examples/toy.fasta")
base = run_base_algorithm("greedy", sequences, threshold=0.97, distance_fn=norm_edit_distance)

builder = PostHocTreeBuilder(
    recluster_method="greedy",
    level_thresholds=[0.90, 0.75],
    distance_fn=norm_edit_distance,
)
tree = builder.build_from_base_result(sequences, base)

tree.export_level_assignments_csv("tree_level_assignments.csv", seq_ids)
tree.export_node_rep_map_csv("tree_nodes.csv", seq_ids)
tree.export_edges_csv("tree_edges.csv")
```
