# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Per-person accuracy and disclosure gain, bootstrap CIs, crossing rate, per-label AUROC."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def binarize(probs: np.ndarray, threshold: float) -> np.ndarray:
    """A label is predicted present when its probability is strictly above the threshold."""
    return (np.asarray(probs) > threshold).astype(int)


def person_accuracy(probs: np.ndarray, y: np.ndarray, threshold: float) -> np.ndarray:
    """Percent of labels right, per person. probs and y are (n_people, n_labels)."""
    return (binarize(probs, threshold) == y).mean(axis=1) * 100.0


def bootstrap_mean_ci(values: np.ndarray, resamples: int, seed: int, alpha: float = 0.05):
    """Mean and percentile CI of the mean across people. values: (n_people,) or (n_people, T)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.shape[0], size=(resamples, v.shape[0]))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return v.mean(axis=0), lo, hi


def crossing(gain: np.ndarray, points: float) -> np.ndarray:
    """Per person: does the gain reach `points` at any turn 1..T? gain is (n_people, T+1)."""
    return (np.asarray(gain)[:, 1:] >= points).any(axis=1)


def first_crossing_turn(gain_row: np.ndarray, points: float) -> int | None:
    hits = np.nonzero(np.asarray(gain_row)[1:] >= points)[0]
    return int(hits[0] + 1) if hits.size else None


def per_label_auroc(probs: np.ndarray, y: np.ndarray, label_ids: list[str]) -> dict[str, float | None]:
    out = {}
    for j, lid in enumerate(label_ids):
        yj = y[:, j]
        out[lid] = None if yj.min() == yj.max() else float(roc_auc_score(yj, probs[:, j]))
    return out


def summarize_curve(gain: np.ndarray, mcfg: dict) -> dict:
    """Aggregate a (n_people, T+1) gain matrix into curve, final-turn and crossing statistics."""
    b = mcfg["bootstrap"]
    mean, lo, hi = bootstrap_mean_ci(gain, b["resamples"], b["seed"])
    cross = crossing(gain, mcfg["crossing_points"]).astype(float)
    cm, clo, chi = bootstrap_mean_ci(cross, b["resamples"], b["seed"])
    return {
        "mean": mean.tolist(), "ci_low": lo.tolist(), "ci_high": hi.tolist(),
        "final": {"mean": float(mean[-1]), "ci_low": float(lo[-1]), "ci_high": float(hi[-1])},
        "turn0": {"mean": float(mean[0]), "ci_low": float(lo[0]), "ci_high": float(hi[0])},
        "crossing_rate": {"mean": float(cm), "ci_low": float(clo), "ci_high": float(chi),
                          "n_crossed": int(cross.sum()), "n": int(cross.size)},
    }
