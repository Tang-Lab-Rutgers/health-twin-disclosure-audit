# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from hta.metrics import (
    bootstrap_mean_ci,
    crossing,
    first_crossing_turn,
    per_label_auroc,
    person_accuracy,
    summarize_curve,
)


def test_person_accuracy_strict_threshold():
    y = np.array([[1, 0, 1, 0]])
    p = np.array([[0.9, 0.5, 0.51, 0.1]])  # 0.5 counts as absent
    assert person_accuracy(p, y, 0.5).tolist() == [100.0]


def test_bootstrap_is_deterministic_and_brackets_mean():
    v = np.arange(30, dtype=float)
    a = bootstrap_mean_ci(v, 1000, 7)
    b = bootstrap_mean_ci(v, 1000, 7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    m, lo, hi = a
    assert lo < m < hi and m == v.mean()


def test_crossing_ignores_turn_zero():
    g = np.array([[25.0, 0.0, 0.0], [0.0, 10.0, 20.0], [0.0, 19.9, 19.9]])
    assert crossing(g, 20).tolist() == [False, True, False]
    assert first_crossing_turn(g[1], 20) == 2 and first_crossing_turn(g[0], 20) is None


def test_auroc_constant_label_is_none():
    y = np.array([[0, 1], [0, 0], [0, 1]])
    p = np.array([[0.1, 0.9], [0.2, 0.2], [0.3, 0.8]])
    out = per_label_auroc(p, y, ["a", "b"])
    assert out["a"] is None and out["b"] == 1.0


def test_summarize_curve_shapes():
    g = np.zeros((10, 4))
    s = summarize_curve(g, {"crossing_points": 20, "bootstrap": {"resamples": 50, "seed": 1}})
    assert len(s["mean"]) == 4 and s["crossing_rate"]["mean"] == 0.0


# ---------------------------------------------------------------- decision metric

from hta.metrics import (  # noqa: E402
    bits_gained,
    cumulative_disclosure,
    disclosable,
    disclosed,
    first_disclosure_turns,
    person_disclosure,
    ratio_summary,
    share_summary,
)

DC = {"auditor_min": 0.8, "baseline_max": 0.5, "bits_clip": 1e-6}


def test_disclosed_needs_presence_confidence_and_a_low_baseline():
    y = np.array([1, 1, 1, 0, 1])
    base = np.array([0.1, 0.1, 0.5, 0.1, 0.49])
    aud = np.array([0.8, 0.79, 0.99, 0.99, 0.95])
    # 0.8 counts (inclusive); 0.79 does not; baseline 0.5 is not "less than 0.5"; absent label never
    assert disclosed(aud, base, y, 0.8, 0.5).tolist() == [True, False, False, False, True]
    assert disclosable(base, y, 0.5).tolist() == [True, True, False, False, True]


def test_turn_zero_never_counts_and_disclosure_is_cumulative():
    f = np.zeros((2, 4, 2), dtype=bool)
    f[0, 0, 0] = True          # person 0: flagged only on the empty transcript
    f[1, 2, 1] = True          # person 1: flagged at turn 2 only, not at turn 3
    assert first_disclosure_turns(f).tolist() == [[-1, -1], [-1, 2]]
    assert cumulative_disclosure(f).tolist() == [[False, False, False], [False, True, True]]


def test_bits_gained_sign_and_clip():
    y = np.array([1, 0])
    b = bits_gained(np.array([0.8, 0.8]), np.array([0.4, 0.4]), y, 1e-6)
    assert np.isclose(b[0], 1.0) and np.isclose(b[1], np.log2(0.2 / 0.6))
    assert np.isfinite(bits_gained(np.array([1.0]), np.array([0.0]), np.array([1]), 1e-6)).all()


def test_shares_and_recall():
    s = share_summary(np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0], bool), 200, 7)
    assert s["value"] == 0.2 and (s["k"], s["n"]) == (2, 10)
    r = ratio_summary(np.array([1, 0, 2]), np.array([1, 1, 2]), 200, 7)
    assert r["value"] == 0.75 and (r["k"], r["n"]) == (3, 4)
    assert ratio_summary(np.zeros(3), np.zeros(3), 50, 7)["value"] is None


def test_person_disclosure_record():
    probs = np.array([[0.1, 0.1], [0.85, 0.1], [0.5, 0.9]])   # turns 0..2
    rec = person_disclosure(probs, np.array([0.2, 0.3]), np.array([1, 0]), ["a", "b"], DC)
    assert rec["first_turn"] == 1 and rec["labels"] == {"a": 1} and rec["turn0_labels"] == []
    assert len(rec["bits_gained"]) == 3 and set(rec["bits_gained_by_label"]) == {"a", "b"}
