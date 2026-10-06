"""Grouped evaluation and checkpoint training for QS-GCMM-AF."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .features import chemistry_features, clean_sequence, peptide_smiles, sequence_features
from .metrics import metrics
from .model import GCMMAffinity, device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"peptide_sequence", "target_sequence", "pKd"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"labels must contain {', '.join(sorted(required))}")
    valid: list[dict[str, str]] = []
    for row in rows:
        peptide = clean_sequence(row.get("peptide_sequence", ""))
        target = clean_sequence(row.get("target_sequence", ""))
        try:
            pkd = float(row.get("pKd", ""))
        except (TypeError, ValueError):
            continue
        if not peptide or not target or not math.isfinite(pkd):
            continue
        cleaned = dict(row)
        cleaned["peptide_sequence"] = peptide
        cleaned["target_sequence"] = target
        cleaned["pKd"] = str(pkd)
        valid.append(cleaned)
    if len(valid) < 8:
        raise RuntimeError(f"only {len(valid)} valid labels remain; grouped evaluation needs at least 8")
    return valid


def _group_value(row: dict[str, str], field: str) -> str:
    if field == "peptide_sequence":
        return row["peptide_sequence"]
    if field == "target_sequence_sha256":
        return row.get(field) or hashlib.sha256(row["target_sequence"].encode("ascii")).hexdigest()
    if field == "target_family_id":
        return row.get(field) or _group_value(row, "target_sequence_sha256")
    return row.get(field) or row["target_sequence"]


def split_indices(rows: list[dict[str, str]], field: str, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    groups = sorted({_group_value(row, field) for row in rows})
    if len(groups) < 2:
        raise RuntimeError(f"cannot split {field}: only {len(groups)} groups")
    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n_test = min(max(1, int(round(len(groups) * fraction))), len(groups) - 1)
    test_groups = set(shuffled[:n_test])
    test = np.asarray([_group_value(row, field) in test_groups for row in rows], dtype=bool)
    return np.flatnonzero(~test), np.flatnonzero(test), len(groups) - n_test, n_test


def _features(rows: list[dict[str, str]], hash_dim: int):
    peptides = [row["peptide_sequence"] for row in rows]
    targets = [row["target_sequence"] for row in rows]
    return (
        sequence_features(peptides, hash_dim),
        sequence_features(targets, hash_dim),
        chemistry_features([peptide_smiles(sequence) for sequence in peptides]),
        np.asarray([float(row["pKd"]) for row in rows], dtype=np.float32),
    )


def fit(
    rows: list[dict[str, str]],
    hash_dim: int,
    epochs: int,
    batch_size: int,
    device_name: str | None,
    seed: int,
    hidden_dim: int = 128,
    dropout: float = 0.12,
):
    torch = __import__("torch")
    import torch.nn.functional as F

    pep_np, target_np, chem_np, truth_np = _features(rows, hash_dim)
    backend = device(device_name)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = GCMMAffinity(pep_np.shape[1], hidden_dim, dropout).to(backend)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    pep = torch.as_tensor(pep_np, dtype=torch.float32, device=backend)
    target = torch.as_tensor(target_np, dtype=torch.float32, device=backend)
    chem = torch.as_tensor(chem_np, dtype=torch.float32, device=backend)
    truth = torch.as_tensor(truth_np, dtype=torch.float32, device=backend)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(max(1, epochs)):
        model.train(True)
        permutation = torch.randperm(len(rows), generator=generator)
        for start in range(0, len(rows), max(1, batch_size)):
            index = permutation[start : start + max(1, batch_size)].to(backend)
            prediction = model.forward_pairs(pep[index], target[index], chem[index])
            regression = F.smooth_l1_loss(prediction, truth[index])
            if len(index) > 1:
                pair_delta = prediction[:, None] - prediction[None, :]
                truth_delta = truth[index][:, None] - truth[index][None, :]
                mask = truth_delta != 0
                ranking = F.softplus(-(pair_delta * torch.sign(truth_delta))).masked_select(mask).mean() if mask.any() else regression * 0
                loss = regression + 0.15 * ranking
            else:
                loss = regression
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model, backend, pep_np, target_np, chem_np, truth_np


def predict(model, backend, pep_np, target_np, chem_np) -> np.ndarray:
    torch = __import__("torch")
    model.eval()
    with torch.inference_mode():
        return model.forward_pairs(
            torch.as_tensor(pep_np, dtype=torch.float32, device=backend),
            torch.as_tensor(target_np, dtype=torch.float32, device=backend),
            torch.as_tensor(chem_np, dtype=torch.float32, device=backend),
        ).detach().cpu().numpy()


def train_and_evaluate(
    labels_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    epochs: int,
    batch_size: int,
    hash_dim: int,
    device_name: str | None,
    test_fraction: float,
) -> dict[str, object]:
    torch = __import__("torch")
    rows = read_rows(labels_path)
    split_reports: dict[str, object] = {}
    split_specs = (
        ("target_family_disjoint", "target_family_id", 20260822),
        ("target_sequence_disjoint", "target_sequence_sha256", 20260823),
        ("peptide_disjoint", "peptide_sequence", 20260824),
    )
    for split_name, field, seed in split_specs:
        train_idx, test_idx, train_groups, test_groups = split_indices(rows, field, test_fraction, seed)
        train_rows = [rows[index] for index in train_idx]
        test_rows = [rows[index] for index in test_idx]
        model, backend, *_ = fit(train_rows, hash_dim, epochs, batch_size, device_name, seed)
        test_pep, test_target, test_chem, test_truth = _features(test_rows, hash_dim)
        prediction = predict(model, backend, test_pep, test_target, test_chem)
        split_reports[split_name] = {
            "group_field": field,
            "train_rows": len(train_rows), "test_rows": len(test_rows),
            "train_groups": train_groups, "test_groups": test_groups,
            "metrics": metrics(test_truth, prediction),
        }
    model, backend, pep_np, target_np, chem_np, _ = fit(rows, hash_dim, epochs, batch_size, device_name, 20260825)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    human_kd = all(
        row.get("affinity_types", "").upper() == "KD"
        and row.get("target_organism", "") == "Homo sapiens"
        and row.get("target_relationship", "") == "SINGLE PROTEIN"
        and row.get("target_component_count", "") == "1"
        for row in rows
    )
    task = "human single-protein thermodynamic Kd expressed as pKd" if human_kd else "heterogeneous affinity pKd proxy"
    report = {
        "status": "PASS_HUMAN_KD_DEVELOPMENT_MODEL_TRAINED" if human_kd else "PASS_GROUPED_MODEL_TRAINED",
        "algorithm": "GCMM-AF", "model_name": "QS-GCMM-AF", "task": task,
        "measurement_contract": "HUMAN_SINGLE_PROTEIN_EXACT_KD" if human_kd else "HETEROGENEOUS_AFFINITY_PROXY",
        "labels": str(labels_path), "labels_sha256": sha256_file(labels_path), "rows": len(rows),
        "hash_dim": hash_dim, "device": str(backend), "cuda": bool(torch.cuda.is_available()),
        "grouped_evaluation": split_reports, "calibration_status": "NOT_EXTERNAL_CALIBRATED",
        "claim_boundary": "grouped metrics are internal development evidence; the model is not independently calibrated and predictions are not experimental Kd",
        "created_at": now(),
    }
    torch.save(
        {
            "format": "GCMM-AF-v1", "model_name": "QS-GCMM-AF", "task": task,
            "measurement_contract": report["measurement_contract"], "hash_dim": hash_dim,
            "input_dim": int(pep_np.shape[1]), "hidden_dim": model.hidden_dim, "dropout": 0.12,
            "calibration_status": "NOT_EXTERNAL_CALIBRATED", "state_dict": model.state_dict(),
            "labels_sha256": sha256_file(labels_path), "trained_at": now(), "n_rows": len(rows),
            "training_report": str(report_path),
        },
        checkpoint_path,
    )
    report["checkpoint"] = str(checkpoint_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    return report
