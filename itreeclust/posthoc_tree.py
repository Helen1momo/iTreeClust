# -*- coding: utf-8 -*-
"""Core post-hoc explanation tree construction for iTreeClust.

The tree is built after a bottom-level flat clustering step. Leaves are the
bottom clusters. Each upper level is produced by reclustering the representatives
of the previous level with a user-specified similarity threshold.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:  # package import
    from .base_algorithms import BaseClusteringResult, norm_edit_distance, run_base_algorithm
except Exception:  # direct script import
    from base_algorithms import BaseClusteringResult, norm_edit_distance, run_base_algorithm  # type: ignore


@dataclass
class TreeNode:
    """One node in the post-hoc explanation tree."""

    node_id: str
    level: int
    threshold: float
    member_indices: List[int]
    rep_index: int
    rep_seq: str
    child_ids: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    source_cluster_id: Optional[int] = None

    @property
    def weight(self) -> int:
        return len(self.member_indices)


class HierarchyTree:
    """Container and export utilities for the post-hoc tree."""

    def __init__(
        self,
        nodes: Dict[str, TreeNode],
        levels: Dict[int, List[str]],
        selected_thresholds: Dict[int, float],
        base_threshold: float,
    ) -> None:
        self.nodes = nodes
        self.levels = {int(k): list(v) for k, v in levels.items()}
        self.selected_thresholds = {int(k): float(v) for k, v in selected_thresholds.items()}
        self.base_threshold = float(base_threshold)

    @property
    def max_level(self) -> int:
        return max(self.levels) if self.levels else 0

    def level_labels(self, n_sequences: int, level: int, as_int: bool = False) -> List[Any]:
        """Return per-sequence labels for a given tree level."""
        labels: List[Any] = [-1] * n_sequences
        node_ids = self.levels.get(int(level), [])
        int_map = {node_id: i for i, node_id in enumerate(node_ids)}
        for node_id in node_ids:
            node = self.nodes[node_id]
            value: Any = int_map[node_id] if as_int else node_id
            for idx in node.member_indices:
                if 0 <= idx < n_sequences:
                    labels[idx] = value
        return labels

    def ancestor_at_level(self, node_id: str, target_level: int) -> Optional[str]:
        """Return the ancestor of ``node_id`` at ``target_level``."""
        current = self.nodes.get(node_id)
        if current is None:
            return None
        while current is not None and current.level < int(target_level):
            if current.parent_id is None:
                return None
            current = self.nodes.get(current.parent_id)
        if current is not None and current.level == int(target_level):
            return current.node_id
        return None

    def route_sequence(
        self,
        sequence: str,
        distance_fn: Callable[[str, str], float] = norm_edit_distance,
    ) -> List[Tuple[int, str, float]]:
        """Greedy top-down routing using node representatives.

        This helper is mainly for quick inspection. The command-line program
        does not rely on this for tree construction.
        """
        if not self.levels:
            return []
        current_candidates = list(self.levels[self.max_level])
        route: List[Tuple[int, str, float]] = []
        for level in range(self.max_level, -1, -1):
            candidates = [nid for nid in current_candidates if self.nodes[nid].level == level]
            if not candidates:
                break
            scored = [(float(distance_fn(sequence, self.nodes[nid].rep_seq)), nid) for nid in candidates]
            scored.sort(key=lambda x: (x[0], x[1]))
            best_d, best_id = scored[0]
            route.append((level, best_id, best_d))
            current_candidates = list(self.nodes[best_id].child_ids)
        return route

    def export_level_assignments_csv(self, path: Path, seq_ids: Sequence[str]) -> None:
        """Write per-sequence assignments at every tree level."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        level_numbers = sorted(self.levels)
        level_labels = {lv: self.level_labels(len(seq_ids), lv, as_int=False) for lv in level_numbers}
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            fieldnames = ["seq_id"] + [f"level_{lv}" for lv in level_numbers]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, seq_id in enumerate(seq_ids):
                row = {"seq_id": seq_id}
                for lv in level_numbers:
                    row[f"level_{lv}"] = level_labels[lv][i]
                writer.writerow(row)

    def export_node_rep_map_csv(self, path: Path, seq_ids: Optional[Sequence[str]] = None) -> None:
        """Write one row per node, including representative and containment metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "level",
            "threshold",
            "node_id",
            "parent_id",
            "n_sequences",
            "n_children",
            "child_ids",
            "representative_index",
            "representative_seq_id",
            "source_cluster_id",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for level in sorted(self.levels):
                for node_id in self.levels[level]:
                    node = self.nodes[node_id]
                    rep_seq_id = ""
                    if seq_ids is not None and 0 <= node.rep_index < len(seq_ids):
                        rep_seq_id = str(seq_ids[node.rep_index])
                    writer.writerow({
                        "level": node.level,
                        "threshold": node.threshold,
                        "node_id": node.node_id,
                        "parent_id": node.parent_id or "",
                        "n_sequences": node.weight,
                        "n_children": len(node.child_ids),
                        "child_ids": ";".join(node.child_ids),
                        "representative_index": node.rep_index,
                        "representative_seq_id": rep_seq_id,
                        "source_cluster_id": node.source_cluster_id if node.source_cluster_id is not None else "",
                    })

    def export_edges_csv(self, path: Path) -> None:
        """Write parent-child containment edges."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "parent_level",
            "parent_node_id",
            "child_level",
            "child_node_id",
            "parent_threshold",
            "child_threshold",
            "parent_n_sequences",
            "child_n_sequences",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for parent_id, parent in sorted(self.nodes.items(), key=lambda x: (x[1].level, x[0])):
                for child_id in parent.child_ids:
                    child = self.nodes[child_id]
                    writer.writerow({
                        "parent_level": parent.level,
                        "parent_node_id": parent.node_id,
                        "child_level": child.level,
                        "child_node_id": child.node_id,
                        "parent_threshold": parent.threshold,
                        "child_threshold": child.threshold,
                        "parent_n_sequences": parent.weight,
                        "child_n_sequences": child.weight,
                    })

    def export_representatives_fasta(self, path: Path, seq_ids: Optional[Sequence[str]] = None) -> None:
        """Write representative sequence of every node as FASTA."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for level in sorted(self.levels):
                for node_id in self.levels[level]:
                    node = self.nodes[node_id]
                    rep_seq_id = ""
                    if seq_ids is not None and 0 <= node.rep_index < len(seq_ids):
                        rep_seq_id = str(seq_ids[node.rep_index])
                    header = (
                        f">{node.node_id} level={node.level} threshold={node.threshold} "
                        f"size={node.weight} rep_index={node.rep_index} rep_seq_id={rep_seq_id}"
                    )
                    f.write(header + "\n")
                    seq = str(node.rep_seq)
                    for start in range(0, len(seq), 80):
                        f.write(seq[start:start + 80] + "\n")


class PostHocTreeBuilder:
    """Build a fixed-threshold post-hoc tree from a flat clustering result."""

    def __init__(
        self,
        recluster_method: str = "vsearch",
        level_thresholds: Optional[Sequence[float]] = None,
        threshold_grid: Optional[Sequence[float]] = None,
        n_levels: Optional[int] = None,
        top_k: int = 3,
        distance_fn: Callable[[str, str], float] = norm_edit_distance,
        recluster_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> None:
        # ``threshold_grid``, ``n_levels`` and ``top_k`` are accepted for
        # compatibility with older experiment scripts. Fixed-threshold mode uses
        # ``level_thresholds`` directly.
        thresholds = list(level_thresholds if level_thresholds is not None else (threshold_grid or []))
        if n_levels is not None and thresholds and int(n_levels) != len(thresholds):
            raise ValueError(f"n_levels={n_levels} but {len(thresholds)} thresholds were provided")
        if not thresholds:
            raise ValueError("At least one upper-level threshold is required")
        self.recluster_method = recluster_method
        self.level_thresholds = [float(x) for x in thresholds]
        self.n_levels = len(self.level_thresholds)
        self.top_k = top_k
        self.distance_fn = distance_fn
        self.recluster_kwargs = dict(recluster_kwargs or {})
        self.verbose = verbose

    @staticmethod
    def _members_from_labels(labels: Sequence[int]) -> Dict[int, List[int]]:
        members: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            if int(label) != -1:
                members[int(label)].append(idx)
        return dict(members)

    def build_from_base_result(
        self,
        sequences: Sequence[str],
        base_result: BaseClusteringResult,
    ) -> HierarchyTree:
        """Build the tree. Level 0 is the bottom flat clustering."""
        nodes: Dict[str, TreeNode] = {}
        levels: Dict[int, List[str]] = {}
        selected_thresholds: Dict[int, float] = {}

        base_members = self._members_from_labels(base_result.labels)
        if not base_members:
            raise ValueError("Bottom clustering produced no assigned clusters")

        level0_ids: List[str] = []
        for order, cid in enumerate(sorted(base_members), start=1):
            members = sorted(base_members[cid])
            rep_index = int(base_result.prototypes.get(cid, members[0]))
            if rep_index not in members:
                rep_index = members[0]
            node_id = f"L0_C{order:06d}"
            nodes[node_id] = TreeNode(
                node_id=node_id,
                level=0,
                threshold=float(base_result.threshold),
                member_indices=members,
                rep_index=rep_index,
                rep_seq=str(sequences[rep_index]),
                child_ids=[],
                parent_id=None,
                source_cluster_id=int(cid),
            )
            level0_ids.append(node_id)
        levels[0] = level0_ids

        previous_ids = level0_ids
        for level, threshold in enumerate(self.level_thresholds, start=1):
            selected_thresholds[level] = float(threshold)
            if self.verbose:
                print(f"[tree] building level {level} at threshold={threshold} from {len(previous_ids)} nodes")

            if len(previous_ids) == 1:
                labels = [0]
                prototypes = {0: 0}
            else:
                previous_rep_sequences = [nodes[nid].rep_seq for nid in previous_ids]
                result = run_base_algorithm(
                    self.recluster_method,
                    previous_rep_sequences,
                    float(threshold),
                    distance_fn=self.distance_fn,
                    **self.recluster_kwargs,
                )
                labels = [int(x) for x in result.labels]
                prototypes = {int(k): int(v) for k, v in result.prototypes.items()}

            child_groups: Dict[int, List[int]] = defaultdict(list)
            for prev_pos, label in enumerate(labels):
                if int(label) != -1:
                    child_groups[int(label)].append(prev_pos)
            if not child_groups:
                # Defensive fallback: keep all previous nodes under one parent.
                child_groups[0] = list(range(len(previous_ids)))
                prototypes = {0: 0}

            current_ids: List[str] = []
            for order, raw_cid in enumerate(sorted(child_groups), start=1):
                child_positions = child_groups[raw_cid]
                child_ids = [previous_ids[pos] for pos in child_positions]
                member_indices = sorted({idx for child_id in child_ids for idx in nodes[child_id].member_indices})

                proto_pos = prototypes.get(raw_cid, child_positions[0])
                if proto_pos not in child_positions:
                    proto_pos = child_positions[0]
                proto_child_id = previous_ids[proto_pos]
                rep_index = nodes[proto_child_id].rep_index

                node_id = f"L{level}_N{order:06d}"
                nodes[node_id] = TreeNode(
                    node_id=node_id,
                    level=level,
                    threshold=float(threshold),
                    member_indices=member_indices,
                    rep_index=rep_index,
                    rep_seq=str(sequences[rep_index]),
                    child_ids=child_ids,
                    parent_id=None,
                    source_cluster_id=int(raw_cid),
                )
                for child_id in child_ids:
                    nodes[child_id].parent_id = node_id
                current_ids.append(node_id)

            levels[level] = current_ids
            previous_ids = current_ids

        return HierarchyTree(
            nodes=nodes,
            levels=levels,
            selected_thresholds=selected_thresholds,
            base_threshold=float(base_result.threshold),
        )
