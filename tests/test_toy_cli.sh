#!/usr/bin/env bash
set -euo pipefail
python run_itreeclust.py \
  --fasta examples/toy.fasta \
  --outdir outputs/toy_tree \
  --prefix toy \
  --base-method greedy \
  --base-threshold 0.97 \
  --n-levels 2 \
  --level-thresholds 0.90,0.75 \
  --taxonomy examples/toy_taxonomy.csv

test -s outputs/toy_tree/tree_level_summary.csv
test -s outputs/toy_tree/tree_nodes.csv
test -s outputs/toy_tree/tree_edges.csv
test -s outputs/toy_tree/tree_level_assignments.csv
test -s outputs/toy_tree/tree_node_representatives.fasta
