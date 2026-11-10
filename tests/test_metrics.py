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
