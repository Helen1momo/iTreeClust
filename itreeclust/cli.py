# -*- coding: utf-8 -*-
"""Command-line interface for building an iTreeClust explanation tree."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .base_algorithms import norm_edit_distance, read_fasta, run_base_algorithm
    from .posthoc_tree import HierarchyTree, PostHocTreeBuilder
except Exception:  # pragma: no cover - allows direct execution during development
    from base_algorithms import norm_edit_distance, read_fasta, run_base_algorithm  # type: ignore
    from posthoc_tree import HierarchyTree, PostHocTreeBuilder  # type: ignore


# -----------------------------------------------------------------------------
# Small IO helpers
# -----------------------------------------------------------------------------


def parse_thresholds(text: str) -> List[float]:
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("No thresholds were provided")
    return values


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# -----------------------------------------------------------------------------
# Tree summaries
# -----------------------------------------------------------------------------


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: Sequence[float]) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    return float(vals[mid]) if len(vals) % 2 else float((vals[mid - 1] + vals[mid]) / 2.0)


def level_summary_rows(tree: HierarchyTree) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_total = max((max(node.member_indices) for node in tree.nodes.values() if node.member_indices), default=-1) + 1
    for level in sorted(tree.levels):
        node_ids = tree.levels[level]
        sizes = [tree.nodes[nid].weight for nid in node_ids]
        threshold = tree.base_threshold if level == 0 else tree.selected_thresholds.get(level, "")
        rows.append({
            "level": level,
            "role": "base_flat_clusters" if level == 0 else f"explanation_level_{level}",
            "threshold": threshold,
            "n_nodes": len(node_ids),
            "n_sequences": n_total,
            "mean_node_size": mean(sizes),
            "median_node_size": median(sizes),
            "min_node_size": min(sizes) if sizes else 0,
            "max_node_size": max(sizes) if sizes else 0,
            "max_node_fraction": max(sizes) / n_total if sizes and n_total else 0.0,
            "n_singleton_nodes": sum(1 for x in sizes if x == 1),
            "compression_vs_base_level": (len(tree.levels[0]) / len(node_ids)) if node_ids else 0.0,
        })
    return rows


def edge_rows(tree: HierarchyTree) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for parent_id in sorted(tree.nodes, key=lambda nid: (tree.nodes[nid].level, nid)):
        parent = tree.nodes[parent_id]
        for child_id in parent.child_ids:
            child = tree.nodes[child_id]
            rows.append({
                "parent_level": parent.level,
                "parent_node_id": parent.node_id,
                "child_level": child.level,
                "child_node_id": child.node_id,
                "parent_threshold": parent.threshold,
                "child_threshold": child.threshold,
                "parent_n_sequences": parent.weight,
                "child_n_sequences": child.weight,
                "child_fraction_within_parent": child.weight / parent.weight if parent.weight else 0.0,
            })
    return rows


def node_rows(tree: HierarchyTree, seq_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for level in sorted(tree.levels):
        for node_id in tree.levels[level]:
            node = tree.nodes[node_id]
            rows.append({
                "level": node.level,
                "threshold": node.threshold,
                "node_id": node.node_id,
                "parent_id": node.parent_id or "",
                "n_sequences": node.weight,
                "n_children": len(node.child_ids),
                "representative_index": node.rep_index,
                "representative_seq_id": seq_ids[node.rep_index] if 0 <= node.rep_index < len(seq_ids) else "",
                "child_ids": ";".join(node.child_ids),
                "source_cluster_id": node.source_cluster_id if node.source_cluster_id is not None else "",
            })
    return rows


# -----------------------------------------------------------------------------
# Optional taxonomy summaries
# -----------------------------------------------------------------------------


def load_taxonomy(
    taxonomy_csv: Optional[Path],
    seq_ids: Sequence[str],
    seq_id_col: str,
    ranks: Sequence[str],
) -> Dict[str, List[str]]:
    if taxonomy_csv is None:
        return {}
    taxonomy_csv = Path(taxonomy_csv)
    if not taxonomy_csv.exists():
        raise FileNotFoundError(f"Taxonomy CSV not found: {taxonomy_csv}")
    by_id: Dict[str, Dict[str, str]] = {}
    with taxonomy_csv.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty taxonomy CSV: {taxonomy_csv}")
        if seq_id_col not in reader.fieldnames:
            raise ValueError(f"Taxonomy CSV must contain column {seq_id_col!r}; found {reader.fieldnames}")
        for row in reader:
            sid = str(row.get(seq_id_col, "")).strip()
            if sid:
                by_id[sid] = {k: str(v or "").strip() for k, v in row.items()}
    out: Dict[str, List[str]] = {}
    for rank in ranks:
        out[rank] = [by_id.get(seq_id, {}).get(rank, "UNK") or "UNK" for seq_id in seq_ids]
    return out


def dominant(values: Sequence[str]) -> Tuple[str, float, int]:
    vals = [str(v) if str(v).strip() else "UNK" for v in values]
    vals = [v for v in vals if v != "UNK"]
    if not vals:
        return "UNK", 0.0, 0
    label, count = Counter(vals).most_common(1)[0]
    return label, count / len(vals), count


def node_taxonomy_rows(tree: HierarchyTree, taxonomy_by_rank: Dict[str, Sequence[str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not taxonomy_by_rank:
        return rows
    for level in sorted(tree.levels):
        for node_id in tree.levels[level]:
            node = tree.nodes[node_id]
            for rank, labels in taxonomy_by_rank.items():
                node_labels = [labels[i] for i in node.member_indices if 0 <= i < len(labels)]
                label, purity, count = dominant(node_labels)
                rows.append({
                    "level": node.level,
                    "node_id": node.node_id,
                    "rank": rank,
                    "dominant_label": label,
                    "purity": purity,
                    "dominant_count": count,
                    "n_sequences_with_known_label": sum(1 for x in node_labels if str(x).strip() and str(x) != "UNK"),
                    "n_sequences": node.weight,
                })
    return rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="itreeclust",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build a fixed-threshold post-hoc explanation tree for biological sequence clusters.",
    )
    parser.add_argument("--fasta", type=Path, required=True, help="Input FASTA file.")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for output CSV/FASTA/JSON files.")
    parser.add_argument("--prefix", type=str, default="dataset", help="Dataset name used in run_config.json.")

    parser.add_argument("--base-method", type=str, default="vsearch", choices=["vsearch", "cdhit", "cd-hit", "clusterize", "greedy"],
                        help="Bottom-level and upper-level reclustering backend.")
    parser.add_argument("--base-threshold", type=float, required=True,
                        help="Similarity threshold for bottom flat clustering, i.e. level 0 leaves.")
    parser.add_argument("--level-thresholds", type=str, required=True,
                        help="Comma-separated thresholds for upper tree levels, fine to coarse, e.g. 0.95,0.86,0.78.")
    parser.add_argument("--n-levels", type=int, default=None,
                        help="Optional number of upper explanation levels. If provided, it must match --level-thresholds.")

    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--vsearch-exe", type=str, default="vsearch")
    parser.add_argument("--cdhit-exe", type=str, default="cd-hit-est")
    parser.add_argument("--rscript-exe", type=str, default="Rscript")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files produced by external tools.")

    parser.add_argument("--taxonomy", type=Path, default=None, help="Optional taxonomy CSV aligned by sequence ID.")
    parser.add_argument("--seq-id-col", type=str, default="seq_id", help="Sequence ID column in taxonomy CSV.")
    parser.add_argument("--ranks", type=str, default="class,order,family,genus,species",
                        help="Comma-separated taxonomy columns to summarize if --taxonomy is provided.")
    return parser


def make_base_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "vsearch_exe": args.vsearch_exe,
        "cdhit_exe": args.cdhit_exe,
        "rscript_exe": args.rscript_exe,
        "threads": args.threads,
        "processors": args.threads,
        "is_dna": True,
        "strand_both": True,
        "keep_files": bool(args.keep_temp),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    thresholds_raw = parse_thresholds(args.level_thresholds)
    thresholds = sorted(thresholds_raw, reverse=True)
    if thresholds != thresholds_raw:
        print(f"[warning] level thresholds were not descending; using {thresholds}")
    if args.n_levels is not None and int(args.n_levels) != len(thresholds):
        raise ValueError(f"--n-levels={args.n_levels} but {len(thresholds)} thresholds were provided")
    if any(th >= args.base_threshold for th in thresholds):
        print("[warning] an upper-level threshold is >= base-threshold; this is allowed but usually not intended.")

    if not args.fasta.exists():
        raise FileNotFoundError(f"FASTA not found: {args.fasta}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    seq_ids, sequences = read_fasta(args.fasta)
    ranks = [x.strip() for x in args.ranks.split(",") if x.strip()]
    taxonomy_by_rank = load_taxonomy(args.taxonomy, seq_ids, args.seq_id_col, ranks) if args.taxonomy else {}

    print("=" * 72)
    print("iTreeClust fixed-threshold explanation tree")
    print("=" * 72)
    print(f"Dataset          : {args.prefix}")
    print(f"Sequences        : {len(sequences)}")
    print(f"Base method      : {args.base_method}")
    print(f"Base threshold   : {args.base_threshold}")
    print(f"Upper thresholds : {thresholds}")
    print(f"Output directory : {args.outdir}")

    base_kwargs = make_base_kwargs(args)
    print("\n[1/3] Bottom flat clustering ...")
    base_result = run_base_algorithm(
        args.base_method,
        sequences,
        args.base_threshold,
        distance_fn=norm_edit_distance,
        **base_kwargs,
    )
    print(f"      clusters={base_result.n_clusters}, coverage={base_result.coverage:.4f}")

    print("[2/3] Building fixed-threshold post-hoc tree ...")
    builder = PostHocTreeBuilder(
        recluster_method=args.base_method,
        level_thresholds=thresholds,
        n_levels=len(thresholds),
        distance_fn=norm_edit_distance,
        recluster_kwargs=base_kwargs,
        verbose=True,
    )
    tree = builder.build_from_base_result(sequences, base_result)

    print("[3/3] Writing outputs ...")
    tree.export_level_assignments_csv(args.outdir / "tree_level_assignments.csv", seq_ids)
    tree.export_node_rep_map_csv(args.outdir / "tree_nodes.csv", seq_ids)
    tree.export_edges_csv(args.outdir / "tree_edges.csv")
    tree.export_representatives_fasta(args.outdir / "tree_node_representatives.fasta", seq_ids)
    write_csv_rows(args.outdir / "tree_level_summary.csv", level_summary_rows(tree))
    write_csv_rows(args.outdir / "node_taxonomy_summary.csv", node_taxonomy_rows(tree, taxonomy_by_rank))

    write_json(args.outdir / "run_config.json", {
        "dataset": args.prefix,
        "fasta": str(args.fasta),
        "taxonomy": str(args.taxonomy) if args.taxonomy else None,
        "base_method": args.base_method,
        "base_threshold": args.base_threshold,
        "upper_level_thresholds": thresholds,
        "n_upper_levels": len(thresholds),
        "total_levels_including_base": len(thresholds) + 1,
        "level_semantics": "level_0 is bottom flat clustering; level_i is built by reclustering representatives from level_i-1 at upper_level_thresholds[i-1]",
        "outputs": {
            "tree_level_summary.csv": "one row per level",
            "tree_nodes.csv": "one row per tree node, including representative and children",
            "tree_edges.csv": "parent-child containment relations",
            "tree_level_assignments.csv": "per-sequence assignment at every level",
            "tree_node_representatives.fasta": "representative sequence for every node",
            "node_taxonomy_summary.csv": "optional dominant taxonomy/purity per node if taxonomy CSV is provided",
        },
    })

    node_path = [len(tree.levels[level]) for level in sorted(tree.levels)]
    print(f"\nDONE. Node path level_0 -> level_{tree.max_level}: {node_path}")
    print(f"Outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()
