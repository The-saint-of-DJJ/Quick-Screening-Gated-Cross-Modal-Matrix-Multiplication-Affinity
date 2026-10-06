"""Small dependency-light regression/ranking metrics used by training gates."""
from __future__ import annotations

import numpy as np


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        tied = np.flatnonzero(values == value)
        if len(tied) > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.std(truth) == 0 or np.std(prediction) == 0:
        return float("nan")
    return float(np.corrcoef(rank(truth), rank(prediction))[0, 1])


def auroc(truth: np.ndarray, prediction: np.ndarray, threshold: float = 6.0) -> float:
    labels = truth >= threshold
    positives = prediction[labels]
    negatives = prediction[~labels]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    comparisons = (positives[:, None] > negatives[None, :]).mean()
    ties = (positives[:, None] == negatives[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = prediction - truth
    return {
        "n": int(len(truth)),
        "rmse_pKd": float(np.sqrt(np.mean(error * error))),
        "mae_pKd": float(np.mean(np.abs(error))),
        "spearman": spearman(truth, prediction),
        "auroc_pKd_ge_6": auroc(truth, prediction),
    }
