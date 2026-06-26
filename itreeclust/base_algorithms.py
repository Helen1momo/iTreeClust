# -*- coding: utf-8 -*-
"""
base_algorithms.py
==================

Reusable wrappers for bottom-level biological sequence clustering algorithms.

This file is deliberately dataset-independent.  For every algorithm, the output
is normalized to the same object:

    BaseClusteringResult(labels, prototypes, method, threshold, runtime_sec)

where
    labels[i] = cluster id of sequence i, or -1 if unassigned;
    prototypes[cid] = sequence index used as the cluster representative.

Supported methods
-----------------
- vsearch       : external VSEARCH executable
- cdhit/cd-hit  : external CD-HIT / CD-HIT-EST executable
- clusterize    : R + DECIPHER::Clusterize
- greedy        : pure-Python debug fallback, not recommended for final experiments

The code uses temporary directories and does not leave intermediate FASTA/UC/R
files unless keep_files=True.
"""

from __future__ import annotations

import collections
import os
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple



# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------

@dataclass
class BaseClusteringResult:
    labels: List[int]
    prototypes: Dict[int, int]
    method: str
    threshold: float
    runtime_sec: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_sequences(self) -> int:
        return len(self.labels)

    @property
    def n_clusters(self) -> int:
        return len({x for x in self.labels if x != -1})

    @property
    def coverage(self) -> float:
        if not self.labels:
            return 0.0
        return sum(1 for x in self.labels if x != -1) / len(self.labels)


# -----------------------------------------------------------------------------
# Distance and FASTA helpers
# -----------------------------------------------------------------------------

def _levenshtein_fallback(a: str, b: str) -> int:
    """Small pure-Python Levenshtein distance fallback for debugging."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def norm_edit_distance(a: str, b: str) -> float:
    """Normalized Levenshtein distance: edit_distance / max(lengths)."""
    denom = max(len(a), len(b), 1)
    try:
        import Levenshtein  # type: ignore
        d = Levenshtein.distance(a, b)
    except Exception:
        d = _levenshtein_fallback(a, b)
    return float(d) / float(denom)


def write_fasta(records: Sequence[Tuple[str, str]], path: Path, line_width: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sid, seq in records:
            f.write(f">{sid}\n")
            seq = str(seq).replace("\n", "").replace("\r", "").upper()
            for i in range(0, len(seq), line_width):
                f.write(seq[i:i + line_width] + "\n")


def read_fasta(path: Path) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    seqs: List[str] = []
    current_id: Optional[str] = None
    parts: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    ids.append(current_id)
                    seqs.append("".join(parts).upper())
                current_id = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
    if current_id is not None:
        ids.append(current_id)
        seqs.append("".join(parts).upper())
    return ids, seqs


def _seq_label(i: int) -> str:
    return f"Seq_{i}"


def _idx_from_seq_label(label: str) -> int:
    label = label.split()[0]
    if not label.startswith("Seq_"):
        raise ValueError(f"Unexpected temporary sequence label: {label!r}")
    return int(label.split("_", 1)[1])


def _remap_labels_and_prototypes(
    raw_labels: Sequence[int],
    raw_prototypes: Dict[int, int],
) -> Tuple[List[int], Dict[int, int]]:
    raw_cids = sorted({int(x) for x in raw_labels if x != -1})
    cid_map = {old: new for new, old in enumerate(raw_cids)}
    labels = [cid_map[int(x)] if int(x) in cid_map else -1 for x in raw_labels]

    prototypes: Dict[int, int] = {}
    for old, proto_idx in raw_prototypes.items():
        if int(old) in cid_map:
            prototypes[cid_map[int(old)]] = int(proto_idx)

    # Safety fallback: if a prototype was not detected, use the first member.
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        if cid not in prototypes:
            prototypes[cid] = labels.index(cid)
    return labels, prototypes


def _copy_keep_dir(tmp_obj: tempfile.TemporaryDirectory, keep_files: bool) -> Optional[str]:
    """Return the tmpdir path when keep_files=True; otherwise cleanup by context."""
    if keep_files:
        return tmp_obj.name
    return None


# -----------------------------------------------------------------------------
# Prototype fallback
# -----------------------------------------------------------------------------

def compute_medoids_for_clusters(
    sequences: Sequence[str],
    labels: Sequence[int],
    distance_fn: Callable[[str, str], float] = norm_edit_distance,
    candidate_cap: Optional[int] = 200,
    random_state: int = 0,
) -> Dict[int, int]:
    """Compute one medoid-like representative index for each predicted cluster."""
    rng = random.Random(random_state)

    clusters: Dict[int, List[int]] = collections.defaultdict(list)
    for i, c in enumerate(labels):
        if int(c) != -1:
            clusters[int(c)].append(i)

    prototypes: Dict[int, int] = {}
    for cid, members in clusters.items():
        if len(members) == 1:
            prototypes[cid] = members[0]
            continue

        candidates = list(members)
        if candidate_cap is not None and len(candidates) > candidate_cap:
            candidates = rng.sample(candidates, candidate_cap)

        best_idx = candidates[0]
        best_cost = float("inf")
        for i in candidates:
            cost = 0.0
            for j in members:
                cost += distance_fn(sequences[i], sequences[j])
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        prototypes[cid] = int(best_idx)
    return prototypes


def compute_longest_seq_for_clusters(sequences: Sequence[str], labels: Sequence[int]) -> Dict[int, int]:
    """Representative fallback used for Clusterize-style longest-seed reporting."""
    clusters: Dict[int, List[int]] = collections.defaultdict(list)
    for i, c in enumerate(labels):
        if int(c) != -1:
            clusters[int(c)].append(i)
    return {cid: max(members, key=lambda idx: len(sequences[idx])) for cid, members in clusters.items()}


# -----------------------------------------------------------------------------
# VSEARCH
# -----------------------------------------------------------------------------

def run_vsearch(
    sequences: Sequence[str],
    threshold: float = 0.97,
    vsearch_exe: str = "vsearch",
    threads: int = 1,
    strand_both: bool = True,
    iddef: int = 1,
    qmask: str = "none",
    keep_files: bool = False,
    work_dir: Optional[Path] = None,
) -> BaseClusteringResult:
    """Run VSEARCH --cluster_fast and parse UC assignments."""
    t0 = time.time()
    tmp_parent = Path(work_dir) if work_dir is not None else None
    with tempfile.TemporaryDirectory(prefix="posthoc_vsearch_", dir=tmp_parent) as tmp:
        tmpdir = Path(tmp)
        input_fasta = tmpdir / "input.fasta"
        output_uc = tmpdir / "clusters.uc"
        centroids = tmpdir / "centroids.fasta"
        write_fasta([(_seq_label(i), seq) for i, seq in enumerate(sequences)], input_fasta)

        cmd = [
            vsearch_exe,
            "--cluster_fast", str(input_fasta),
            "--id", f"{float(threshold):.6f}",
            "--iddef", str(iddef),
            "--uc", str(output_uc),
            "--centroids", str(centroids),
            "--qmask", qmask,
            "--threads", str(threads),
        ]
        if strand_both:
            cmd += ["--strand", "both"]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                "VSEARCH failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{res.stdout}\n"
                f"stderr:\n{res.stderr}\n"
            )

        raw_labels = [-1] * len(sequences)
        raw_prototypes: Dict[int, int] = {}
        if output_uc.exists():
            with output_uc.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip() or line.startswith("#"):
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 9:
                        continue
                    rec_type = parts[0]
                    raw_cid = int(parts[1])
                    query = parts[8].split()[0]
                    q_idx = _idx_from_seq_label(query)
                    if rec_type == "S":
                        raw_labels[q_idx] = raw_cid
                        raw_prototypes[raw_cid] = q_idx
                    elif rec_type == "H":
                        raw_labels[q_idx] = raw_cid

        labels, prototypes = _remap_labels_and_prototypes(raw_labels, raw_prototypes)
        keep_path = None
        if keep_files:
            keep_path = str(tmpdir)
            # Prevent TemporaryDirectory cleanup by copying to a stable folder.
            stable = Path.cwd() / f"kept_vsearch_{int(time.time())}"
            shutil.copytree(tmpdir, stable, dirs_exist_ok=True)
            keep_path = str(stable)

    return BaseClusteringResult(
        labels=labels,
        prototypes=prototypes,
        method="vsearch",
        threshold=float(threshold),
        runtime_sec=time.time() - t0,
        extra={"kept_files": keep_path},
    )


# -----------------------------------------------------------------------------
# CD-HIT / CD-HIT-EST
# -----------------------------------------------------------------------------

def _cdhit_word_size(threshold: float, is_dna: bool) -> int:
    if not is_dna:
        if threshold >= 0.70:
            return 5
        if threshold >= 0.60:
            return 4
        return 3
    # CD-HIT-EST practical mapping. For very low identities, VSEARCH is safer.
    if threshold >= 0.95:
        return 10
    if threshold >= 0.90:
        return 8
    if threshold >= 0.88:
        return 7
    if threshold >= 0.85:
        return 6
    if threshold >= 0.80:
        return 5
    return 4


def run_cdhit(
    sequences: Sequence[str],
    threshold: float = 0.97,
    cdhit_exe: Optional[str] = None,
    is_dna: bool = True,
    threads: int = 1,
    memory_mb: int = 0,
    keep_files: bool = False,
    work_dir: Optional[Path] = None,
) -> BaseClusteringResult:
    """Run CD-HIT/CD-HIT-EST and parse .clstr assignments."""
    t0 = time.time()
    program = cdhit_exe or ("cd-hit-est" if is_dna else "cd-hit")
    tmp_parent = Path(work_dir) if work_dir is not None else None
    with tempfile.TemporaryDirectory(prefix="posthoc_cdhit_", dir=tmp_parent) as tmp:
        tmpdir = Path(tmp)
        input_fasta = tmpdir / "input.fasta"
        output_prefix = tmpdir / "cdhit_out"
        write_fasta([(_seq_label(i), seq) for i, seq in enumerate(sequences)], input_fasta)

        n_word = _cdhit_word_size(float(threshold), is_dna=is_dna)
        cmd = [
            program,
            "-i", str(input_fasta),
            "-o", str(output_prefix),
            "-c", str(float(threshold)),
            "-n", str(n_word),
            "-M", str(memory_mb),
            "-T", str(threads),
            "-d", "0",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                "CD-HIT failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{res.stdout}\n"
                f"stderr:\n{res.stderr}\n"
            )

        raw_labels = [-1] * len(sequences)
        raw_prototypes: Dict[int, int] = {}
        clstr = Path(str(output_prefix) + ".clstr")
        current_cid = -1
        if clstr.exists():
            with clstr.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(">Cluster"):
                        current_cid += 1
                        continue
                    if ">Seq_" not in line:
                        continue
                    idx_text = line.split(">Seq_", 1)[1].split("...", 1)[0]
                    seq_idx = int(idx_text)
                    raw_labels[seq_idx] = current_cid
                    if "*" in line:
                        raw_prototypes[current_cid] = seq_idx

        labels, prototypes = _remap_labels_and_prototypes(raw_labels, raw_prototypes)
        keep_path = None
        if keep_files:
            stable = Path.cwd() / f"kept_cdhit_{int(time.time())}"
            shutil.copytree(tmpdir, stable, dirs_exist_ok=True)
            keep_path = str(stable)

    return BaseClusteringResult(
        labels=labels,
        prototypes=prototypes,
        method="cdhit" if not is_dna else "cdhit-est",
        threshold=float(threshold),
        runtime_sec=time.time() - t0,
        extra={"word_size": n_word, "kept_files": keep_path},
    )


# -----------------------------------------------------------------------------
# Clusterize via R DECIPHER
# -----------------------------------------------------------------------------

def run_clusterize(
    sequences: Sequence[str],
    threshold: float = 0.97,
    rscript_exe: str = "Rscript",
    processors: int = 1,
    representative: str = "longest",
    keep_files: bool = False,
    work_dir: Optional[Path] = None,
) -> BaseClusteringResult:
    """Run DECIPHER::Clusterize through a small temporary R script."""
    t0 = time.time()
    tmp_parent = Path(work_dir) if work_dir is not None else None
    with tempfile.TemporaryDirectory(prefix="posthoc_clusterize_", dir=tmp_parent) as tmp:
        tmpdir = Path(tmp)
        input_fasta = tmpdir / "input.fasta"
        output_csv = tmpdir / "clusters.csv"
        r_script = tmpdir / "run_clusterize.R"
        write_fasta([(_seq_label(i), seq) for i, seq in enumerate(sequences)], input_fasta)

        distance_cutoff = 1.0 - float(threshold)
        r_code = f"""
args <- commandArgs(trailingOnly = TRUE)
suppressMessages(library(DECIPHER))
suppressMessages(library(Biostrings))
seqs <- readDNAStringSet(args[1])
clusters <- Clusterize(seqs, cutoff={distance_cutoff:.8f}, processors={int(processors)})
out_df <- data.frame(SequenceID = rownames(clusters), ClusterID = clusters[, 1])
write.csv(out_df, file=args[2], row.names=FALSE, quote=FALSE)
"""
        r_script.write_text(r_code, encoding="utf-8")
        res = subprocess.run([rscript_exe, str(r_script), str(input_fasta), str(output_csv)], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                "Clusterize failed. Make sure R, DECIPHER and Biostrings are installed.\n"
                f"stdout:\n{res.stdout}\n"
                f"stderr:\n{res.stderr}\n"
            )

        raw_labels = [-1] * len(sequences)
        if output_csv.exists():
            lines = output_csv.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines[1:]:
                if not line.strip():
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                seq_id = parts[0].strip().strip('"')
                cid = int(float(parts[1].strip().strip('"')))
                raw_labels[_idx_from_seq_label(seq_id)] = cid

        labels, _ = _remap_labels_and_prototypes(raw_labels, {})
        if representative.lower() == "medoid":
            prototypes = compute_medoids_for_clusters(sequences, labels)
        else:
            prototypes = compute_longest_seq_for_clusters(sequences, labels)

        keep_path = None
        if keep_files:
            stable = Path.cwd() / f"kept_clusterize_{int(time.time())}"
            shutil.copytree(tmpdir, stable, dirs_exist_ok=True)
            keep_path = str(stable)

    return BaseClusteringResult(
        labels=labels,
        prototypes=prototypes,
        method="clusterize",
        threshold=float(threshold),
        runtime_sec=time.time() - t0,
        extra={"distance_cutoff": distance_cutoff, "kept_files": keep_path},
    )


# -----------------------------------------------------------------------------
# Pure-Python greedy debug clustering
# -----------------------------------------------------------------------------

def run_greedy(
    sequences: Sequence[str],
    threshold: float = 0.97,
    distance_fn: Callable[[str, str], float] = norm_edit_distance,
) -> BaseClusteringResult:
    """
    Simple sequential representative clustering.

    This is only a debug fallback for checking pipeline logic when external tools
    are unavailable. It should not be used as the main baseline in the paper.
    """
    t0 = time.time()
    labels: List[int] = [-1] * len(sequences)
    prototypes: Dict[int, int] = {}
    centers: List[int] = []

    for i, seq in enumerate(sequences):
        best_cid: Optional[int] = None
        best_d = float("inf")
        for cid, center_idx in enumerate(centers):
            d = distance_fn(seq, sequences[center_idx])
            if 1.0 - d >= float(threshold) and d < best_d:
                best_cid = cid
                best_d = d
        if best_cid is None:
            cid = len(centers)
            centers.append(i)
            labels[i] = cid
            prototypes[cid] = i
        else:
            labels[i] = best_cid

    return BaseClusteringResult(
        labels=labels,
        prototypes=prototypes,
        method="greedy",
        threshold=float(threshold),
        runtime_sec=time.time() - t0,
        extra={},
    )


# -----------------------------------------------------------------------------
# Unified dispatcher
# -----------------------------------------------------------------------------

def run_base_algorithm(
    method: str,
    sequences: Sequence[str],
    threshold: float,
    **kwargs: Any,
) -> BaseClusteringResult:
    """Unified entry for all base algorithms."""
    m = method.lower().replace("_", "-")
    if m in {"vsearch", "uclust"}:
        allowed = {"vsearch_exe", "threads", "strand_both", "iddef", "qmask", "keep_files", "work_dir"}
        return run_vsearch(sequences, threshold=threshold, **{k: v for k, v in kwargs.items() if k in allowed})
    if m in {"cdhit", "cd-hit", "cdhit-est", "cd-hit-est"}:
        allowed = {"cdhit_exe", "is_dna", "threads", "memory_mb", "keep_files", "work_dir"}
        return run_cdhit(sequences, threshold=threshold, **{k: v for k, v in kwargs.items() if k in allowed})
    if m in {"clusterize", "decipher"}:
        allowed = {"rscript_exe", "processors", "representative", "keep_files", "work_dir"}
        return run_clusterize(sequences, threshold=threshold, **{k: v for k, v in kwargs.items() if k in allowed})
    if m in {"greedy", "debug"}:
        allowed = {"distance_fn"}
        return run_greedy(sequences, threshold=threshold, **{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(f"Unsupported base clustering method: {method!r}")


def scan_thresholds(
    method: str,
    sequences: Sequence[str],
    thresholds: Sequence[float],
    **kwargs: Any,
) -> List[BaseClusteringResult]:
    """Run the same clustering method over a threshold grid."""
    results: List[BaseClusteringResult] = []
    for thr in thresholds:
        print(f"[scan] {method} threshold={thr}")
        results.append(run_base_algorithm(method, sequences, float(thr), **kwargs))
    return results
