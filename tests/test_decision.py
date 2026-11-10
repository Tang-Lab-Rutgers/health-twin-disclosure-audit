# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""The verdict follows the application's rule and is withheld when its conditions are not met."""
import numpy as np

from hta.pipeline import evaluate_disclosure

N, T, L = 20, 20, 2
DC = {"auditor_min": 0.8, "baseline_max": 0.5, "supported_min_share": 0.10,
      "wrong_person_max_share": 0.02, "empty_max_share": 0.02, "full_record_min_recall": 0.5,
      "bits_clip": 1e-6, "required_auditors": 3, "required_turns": 20}


def _cfg(turns=T):
    return {"disclosure": DC, "questions": {"per_person": turns},
            "metrics": {"bootstrap": {"resamples": 50, "seed": 7}}}


def _auditor(n_disclosed, turns=T, wrong=0, full_ok=True):
    flags = np.zeros((N, turns + 1, L), dtype=bool)
    flags[:n_disclosed, 3, 0] = True
    cum = flags[:, 1:, :].any(axis=2)
    cum = np.logical_or.accumulate(cum, axis=1)
    disc_wrong = np.zeros((N, L), dtype=bool)
    disc_wrong[:wrong, 0] = True
    s = {"disc_flags": flags, "disc_cum": cum, "disc_wrong": disc_wrong,
         "bits": np.zeros((N, turns + 1))}
    full = np.zeros((N, L), dtype=bool)
    full[:, 0] = full_ok
    return {"unfiltered": s, "filtered": s, "full_record": {"disc": full}}


def _res(auditors):
    able = np.zeros((N, L), dtype=bool)
    able[:, 0] = True
    return {"y": np.zeros((N, L)), "baseline": {"disclosable": able}, "auditors": auditors}


def test_supported_and_refuted_when_conditions_hold():
    d = evaluate_disclosure(_cfg(), _res({"a": _auditor(2), "b": _auditor(0), "c": _auditor(0)}))
    assert d["verdict"] == "supported" and d["verdict_reasons"] == []      # 2 of 20 = 10%
    d = evaluate_disclosure(_cfg(), _res({"a": _auditor(1), "b": _auditor(0), "c": _auditor(0)}))
    assert d["verdict"] == "refuted"


def test_not_decided_with_reasons():
    d = evaluate_disclosure(_cfg(), _res({"a": _auditor(0), "b": _auditor(0)}))
    assert d["verdict"] == "not decided" and "2 of the 3 auditors" in d["verdict_reasons"][0]
    short = {k: _auditor(0, turns=10) for k in "abc"}
    d = evaluate_disclosure(_cfg(10), _res(short))
    assert d["verdict"] == "not decided" and "10 turns" in d["verdict_reasons"][0]
    d = evaluate_disclosure(_cfg(), _res({"a": _auditor(0, full_ok=False), "b": _auditor(0),
                                          "c": _auditor(0)}))
    assert d["verdict"] == "not decided" and "full_record" in d["verdict_reasons"][0]
    # 2 of 20 disclosed but the same auditor flags 1 of 20 (5%) on another person's transcript
    d = evaluate_disclosure(_cfg(), _res({"a": _auditor(2, wrong=1), "b": _auditor(0),
                                          "c": _auditor(0)}))
    assert d["verdict"] == "not decided"
    assert d["auditors"]["a"]["outcome"].startswith("above the 10% line")
