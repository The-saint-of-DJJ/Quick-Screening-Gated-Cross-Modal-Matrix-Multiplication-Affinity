"""Coarse matrix screening and MC-dropout pair scoring."""
from __future__ import annotations

import csv
import gzip
import io
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .features import AA, DEFAULT_HASH_DIM, chemistry_features, peptide_smiles, sequence_features
from .model import device, load_checkpoint

SCORE_FIELDS = [
    "rank", "peptide_id", "peptide_sequence", "peptide_smiles", "target_gene",
    "target_accession", "target_length", "model_status", "predicted_pKd",
    "predicted_Kd_nM", "affinity_uncertainty_pKd", "proxy_score", "diversity_score",
    "screen_status",
]


def _write_tsv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        binary = gzip.GzipFile(filename=str(path), mode="wb", mtime=0)
        handle = io.TextIOWrapper(binary, encoding="utf-8", newline="")
    else:
        handle = path.open("w", encoding="utf-8", newline="")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _proxy_scores(pep: np.ndarray, target: np.ndarray, chem: np.ndarray) -> np.ndarray:
    """Transparent sequence prior; this branch is explicitly not affinity."""
    pep_norm = pep / np.maximum(np.linalg.norm(pep, axis=1, keepdims=True), 1e-8)
    target_norm = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-8)
    similarity = pep_norm @ target_norm.T
    charge = pep[:, pep.shape[1] - 5].reshape(-1, 1) if pep.shape[1] > 28 else 0.0
    target_charge = target[:, target.shape[1] - 5].reshape(1, -1) if target.shape[1] > 28 else 0.0
    chemistry_bonus = chem[:, 0:1] * 0.08 - chem[:, 1:2] * 0.05
    return similarity + 0.12 * np.tanh(charge * target_charge) + chemistry_bonus


def _row(model_status, calibration_status, checkpoint, value, sd, peptide, target, rank, diversity):
    pred_kd = 10 ** (9.0 - value) if checkpoint and math.isfinite(value) else float("nan")
    return {
        "rank": rank,
        "peptide_id": peptide["peptide_id"],
        "peptide_sequence": peptide["sequence"],
        "peptide_smiles": peptide["peptide_smiles"],
        "target_gene": target["gene_symbol"],
        "target_accession": target["accession"],
        "target_length": target["length"],
        "model_status": model_status,
        "predicted_pKd": f"{value:.6f}" if checkpoint else "NA",
        "predicted_Kd_nM": f"{pred_kd:.6g}" if checkpoint else "NA",
        "affinity_uncertainty_pKd": f"{sd:.6f}" if checkpoint and math.isfinite(sd) else "NA",
        "proxy_score": f"{value:.6f}" if not checkpoint else "NA",
        "diversity_score": f"{diversity:.1f}",
        "screen_status": (
            "PASS_CALIBRATED_MODEL" if checkpoint and calibration_status == "PASS_EXTERNAL_CALIBRATION"
            else "PASS_UNCALIBRATED_MODEL" if checkpoint else "PASS_PROXY_ONLY"
        ),
    }


def screen(
    peptides: list[dict[str, str]],
    targets: list[dict[str, str]],
    output: Path,
    checkpoint: Path | None,
    device_name: str | None,
    top_k: int,
    mc_samples: int,
    batch_size: int,
    hash_dim: int = DEFAULT_HASH_DIM,
    all_pairs: bool = False,
    top_output: Path | None = None,
) -> dict[str, object]:
    torch = __import__("torch")
    torch.manual_seed(20260823)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260823)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive")
    target_rows = [
        row for row in targets
        if row.get("sequence_status", "available") == "available"
        and row.get("sequence") and set(row["sequence"]).issubset(AA)
    ]
    backend = device(device_name)
    if not target_rows:
        _write_tsv(output, [], SCORE_FIELDS)
        if top_output:
            _write_tsv(top_output, [], SCORE_FIELDS)
        return {
            "status": "PASS_SCREEN_EMPTY_SEQUENCE_READY_TARGET_SET",
            "algorithm": "GCMM-AF",
            "model_name": "QS-GCMM-AF",
            "model_status": "NO_SCORE_SEQUENCE_MISSING",
            "device": str(backend),
            "cuda": bool(torch.cuda.is_available()),
            "peptide_count": len(peptides), "target_count": 0, "pair_count": 0,
            "reported_pairs": 0, "sequence_ready_target_count": 0,
            "all_pairs": all_pairs, "output": str(output),
        }
    pep_np = sequence_features([row["sequence"] for row in peptides], hash_dim)
    target_np = sequence_features([row["sequence"] for row in target_rows], hash_dim)
    chem_np = chemistry_features([row["peptide_smiles"] for row in peptides])
    model = None
    model_status = "PROXY_NOT_AFFINITY"
    calibration_status = "NOT_APPLICABLE"
    if checkpoint:
        model, payload, backend = load_checkpoint(checkpoint, pep_np.shape[1], device_name)
        calibration_status = str(payload.get("calibration_status", "NOT_EXTERNAL_CALIBRATED"))
        model_status = f"CHECKPOINT_{payload.get('task', 'pKd')}_{calibration_status}"
    pep = torch.as_tensor(pep_np, dtype=torch.float32, device=backend)
    target = torch.as_tensor(target_np, dtype=torch.float32, device=backend)
    chem = torch.as_tensor(chem_np, dtype=torch.float32, device=backend)
    if model is None:
        matrix = _proxy_scores(pep_np, target_np, chem_np)
        uncertainty = np.full_like(matrix, np.nan)
    else:
        model.eval()
        with torch.inference_mode():
            p_all = torch.nn.functional.normalize(model.module["pep"](torch.cat([pep, chem], dim=-1)), dim=-1)
            t_all = torch.nn.functional.normalize(model.module["target"](target), dim=-1)
            gate_matrix = p_all @ t_all.T
            if all_pairs:
                candidate_indices = np.arange(gate_matrix.numel(), dtype=np.int64)
            else:
                flat = torch.topk(gate_matrix.reshape(-1), k=min(gate_matrix.numel(), max(top_k * 8, top_k))).indices
                candidate_indices = flat.detach().cpu().numpy()
        matrix = np.full((len(peptides), len(target_rows)), np.nan, dtype=np.float32)
        uncertainty = np.full_like(matrix, np.nan)
        model.train(True)
        for start in range(0, len(candidate_indices), max(1, batch_size)):
            batch_indices = candidate_indices[start : start + max(1, batch_size)]
            pis = np.asarray(batch_indices // len(target_rows), dtype=np.int64)
            tis = np.asarray(batch_indices % len(target_rows), dtype=np.int64)
            pidx = torch.as_tensor(pis, dtype=torch.long, device=backend)
            tidx = torch.as_tensor(tis, dtype=torch.long, device=backend)
            values = [model.forward_pairs(pep[pidx], target[tidx], chem[pidx]).detach() for _ in range(max(1, mc_samples))]
            stacked = torch.stack(values, dim=0)
            matrix[pis, tis] = stacked.mean(dim=0).cpu().numpy()
            uncertainty[pis, tis] = stacked.std(dim=0, unbiased=False).cpu().numpy()
        model.eval()
    flat_matrix = matrix.reshape(-1)
    order = np.argsort(np.nan_to_num(flat_matrix, nan=-1e9))[::-1]
    selected: list[int] = []
    selected_prefixes: set[str] = set()
    diversity_by_index: dict[int, float] = {}
    for flat_index in order:
        if not np.isfinite(flat_matrix[flat_index]):
            continue
        pi, _ = divmod(int(flat_index), len(target_rows))
        prefix = peptides[pi]["sequence"][:3]
        novelty = 1.0 if prefix not in selected_prefixes else 0.0
        selected_prefixes.add(prefix)
        diversity_by_index[int(flat_index)] = novelty
        selected.append(int(flat_index))
        if len(selected) >= top_k:
            break
    selected.sort(key=lambda index: float(flat_matrix[index]), reverse=True)
    rows = []
    for rank, flat_index in enumerate(selected, start=1):
        pi, ti = divmod(flat_index, len(target_rows))
        sd = float(uncertainty[pi, ti]) if np.isfinite(uncertainty[pi, ti]) else float("nan")
        rows.append(_row(model_status, calibration_status, checkpoint, float(matrix[pi, ti]), sd, peptides[pi], target_rows[ti], rank, diversity_by_index.get(flat_index, 0.0)))
    if all_pairs:
        def full_rows():
            for rank, flat_index in enumerate((int(index) for index in order if np.isfinite(flat_matrix[index])), start=1):
                pi, ti = divmod(flat_index, len(target_rows))
                sd = float(uncertainty[pi, ti]) if np.isfinite(uncertainty[pi, ti]) else float("nan")
                yield _row(model_status, calibration_status, checkpoint, float(matrix[pi, ti]), sd, peptides[pi], target_rows[ti], rank, diversity_by_index.get(flat_index, 0.0))
        _write_tsv(output, full_rows(), SCORE_FIELDS)
        if top_output:
            _write_tsv(top_output, rows, SCORE_FIELDS)
        reported_pairs = int(np.isfinite(flat_matrix).sum())
    else:
        _write_tsv(output, rows, SCORE_FIELDS)
        reported_pairs = len(rows)
    return {
        "status": "PASS_SCREEN", "algorithm": "GCMM-AF", "model_name": "QS-GCMM-AF",
        "model_status": model_status, "device": str(backend), "cuda": bool(torch.cuda.is_available()),
        "peptide_count": len(peptides), "target_count": len(target_rows),
        "pair_count": len(peptides) * len(target_rows), "reported_pairs": reported_pairs,
        "sequence_ready_target_count": len(target_rows), "all_pairs": all_pairs,
        "top_output": str(top_output) if top_output else "", "checkpoint": str(checkpoint) if checkpoint else "",
        "output": str(output),
        "warning": (
            "Proxy scores are not affinity and must not be interpreted as Kd." if not checkpoint
            else "Independent external calibration remains required." if calibration_status != "PASS_EXTERNAL_CALIBRATION"
            else "Calibrated model output is still a computational hypothesis."
        ),
    }
