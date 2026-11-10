# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Disclosure metrics.

Decision metric (as stated in the application): an attribute the person has counts as disclosed
when an auditor gives it at least `auditor_min` while the demographics-only baseline gives it less
than `baseline_max`; plus bits gained over the baseline. First-version metric, kept as a
secondary output: per-person accuracy gain over the baseline, crossing rate, per-label AUROC.
"""

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


# ------------------------------------------------------------------ decision metric


def r6(x) -> float:
    return float(round(float(x), 6))


def disclosed(aud_p, base_p, y, auditor_min: float, baseline_max: float) -> np.ndarray:
    """True where an attribute the person has gets >= auditor_min from the auditor while the
    baseline gives it < baseline_max. Arrays broadcast; the last axis is the label."""
    return ((np.asarray(y) == 1) & (np.asarray(aud_p) >= auditor_min)
            & (np.asarray(base_p) < baseline_max))


def disclosable(base_p, y, baseline_max: float) -> np.ndarray:
    """Attributes that can count as disclosed at all: present, and the baseline below baseline_max."""
    return (np.asarray(y) == 1) & (np.asarray(base_p) < baseline_max)


def first_disclosure_turns(flags) -> np.ndarray:
    """flags (n, T+1, L) -> (n, L): the first turn in 1..T at which the label is flagged, or -1.
    Turn 0 is the empty transcript and never counts."""
    f = np.asarray(flags, dtype=bool)[:, 1:, :]
    return np.where(f.any(axis=1), f.argmax(axis=1) + 1, -1)


def cumulative_disclosure(flags) -> np.ndarray:
    """flags (n, T+1, L) -> (n, T): has the person had any disclosed attribute at some turn 1..t,
    for t = 1..T. A later, less confident reading does not undo a disclosure."""
    per_turn = np.asarray(flags, dtype=bool)[:, 1:, :].any(axis=2)
    return np.logical_or.accumulate(per_turn, axis=1)


def bits_gained(aud_p, base_p, y, clip: float) -> np.ndarray:
    """log2 P_auditor(true value) - log2 P_baseline(true value), elementwise. Probabilities are
    clipped to [clip, 1 - clip]. Positive: the auditor is closer to the truth than the baseline."""
    y = np.asarray(y)

    def p_true(p):
        p = np.clip(np.asarray(p, dtype=float), clip, 1.0 - clip)
        return np.where(y == 1, p, 1.0 - p)

    return np.log2(p_true(aud_p)) - np.log2(p_true(base_p))


def share_summary(flags, resamples: int, seed: int) -> dict:
    """Share of people flagged, with a percentile bootstrap CI. flags: (n,) bool."""
    v = np.asarray(flags, dtype=float)
    m, lo, hi = bootstrap_mean_ci(v, resamples, seed)
    return {"value": float(m), "ci_low": float(lo), "ci_high": float(hi),
            "k": int(v.sum()), "n": int(v.size)}


def ratio_summary(num, den, resamples: int, seed: int) -> dict:
    """sum(num) / sum(den) over people, with a bootstrap CI that resamples people."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    k, n = int(num.sum()), int(den.sum())
    if n == 0:
        return {"value": None, "ci_low": None, "ci_high": None, "k": 0, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(resamples, num.size))
    d = den[idx].sum(axis=1)
    r = np.where(d > 0, num[idx].sum(axis=1) / np.maximum(d, 1), np.nan)
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return {"value": k / n, "ci_low": float(lo), "ci_high": float(hi), "k": k, "n": n}


def person_disclosure(probs, base_p, y, label_ids: list[str], dcfg: dict) -> dict:
    """One person's disclosure record for one auditor and transcript.

    probs (T+1, L): the auditor's probabilities after turns 0..T; base_p (L,); y (L,).
    The same function builds the stored report and the `hta verify` recomputation.
    """
    probs, base_p, y = np.asarray(probs), np.asarray(base_p), np.asarray(y)
    flags = disclosed(probs, base_p[None], y[None], dcfg["auditor_min"], dcfg["baseline_max"])
    first = first_disclosure_turns(flags[None])[0]
    labels = {lid: int(first[j]) for j, lid in enumerate(label_ids) if first[j] > 0}
    bits = bits_gained(probs, base_p[None], y[None], dcfg["bits_clip"])
    return {
        "first_turn": min(labels.values()) if labels else None,
        "labels": labels,
        "turn0_labels": [lid for j, lid in enumerate(label_ids) if flags[0, j]],
        "bits_gained": [r6(v) for v in bits.sum(axis=1)],
        "bits_gained_by_label": {lid: r6(bits[-1, j]) for j, lid in enumerate(label_ids)},
    }
