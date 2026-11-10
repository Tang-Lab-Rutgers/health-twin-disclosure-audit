# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np

from hta.auditors.baseline import BaselineAuditor, fit_baseline
from hta.auditors.lexical import LexicalModel, fit_lexical
from hta.auditors.llm import parse_answer
from hta.pipeline import assign_donors

LCFG = {"ngram_range": [1, 2], "min_df": 1, "max_features": 100, "C": 1.0}


def _people(n=40):
    out = []
    for i in range(n):
        y = int(i % 2 == 0)
        out.append({"person_id": f"p{i:03d}",
                    "demographics": {"age_band": "50-59" if i < n // 2 else "60-69",
                                     "sex": "female" if i % 4 < 2 else "male"},
                    "labels": {"a": y, "b": int(i < n // 2)}})
    return out


def _rec(p, text):
    return {"person_id": p["person_id"], "age_band": p["demographics"]["age_band"],
            "sex": p["demographics"]["sex"], "text": text}


def test_baseline_fallback():
    ppl = _people()
    base = BaselineAuditor(fit_baseline(ppl, ["a", "b"], min_cell=15))
    # cells have 10 people < 15, so the sex-level prevalence is used
    assert base.prior("50-59", "female") == base.p["by_sex"]["female"]["prevalence"]
    assert base.prior("90+", "unknown") == base.p["overall"]["prevalence"]


def test_lexical_json_roundtrip_is_exact_and_learns_text():
    ppl = _people()
    bp = fit_baseline(ppl, ["a", "b"], min_cell=5)
    recs = [_rec(p, "salad and walking" if p["labels"]["a"] else "coffee and stairs") for p in ppl]
    y = np.array([[p["labels"]["a"], p["labels"]["b"]] for p in ppl])
    params = fit_lexical(recs, y, ["a", "b"], LCFG, "test", bp)
    m1 = LexicalModel(params)
    m2 = LexicalModel(json.loads(json.dumps(params)))
    assert np.array_equal(m1.predict(recs), m2.predict(recs))
    p = m1.predict(recs)
    assert ((p[:, 0] > 0.5) == y[:, 0]).all()


def test_lexical_empty_text_uses_only_the_prior():
    ppl = _people()
    bp = fit_baseline(ppl, ["a", "b"], min_cell=5)
    y = np.array([[p["labels"]["a"], p["labels"]["b"]] for p in ppl])
    params = fit_lexical([_rec(p, "tea toast") for p in ppl], y, ["a", "b"], LCFG, "t", bp)
    m = LexicalModel(params)
    a = m.predict([_rec(ppl[0], "")])
    b = m.predict([_rec(ppl[1], "")])   # same demographics, so same prediction
    assert np.array_equal(a, b)


def test_parse_answer():
    keys = ["C1", "C2", "M1"]
    assert parse_answer('{"C1": 0.2, "C2": 0.9, "M1": 0}', keys) == ({"C1": 0.2, "C2": 0.9, "M1": 0.0}, "ok")
    vals, st = parse_answer('{"C1": 0.2, "C2": "high"', keys)
    assert st == "partial" and vals == {"C1": 0.2}
    assert parse_answer("I cannot tell.", keys) == ({}, "failed")
    assert parse_answer('{"C1": 80, "C2": 1.5, "M1": -1}', keys)[0] == {"C1": 0.8, "C2": 0.015}


def test_donors_never_self_and_match_strata():
    ppl = _people(13)
    d = assign_donors(ppl)
    assert set(d) == {p["person_id"] for p in ppl}
    assert all(v["donor"] != k for k, v in d.items())
    demo = {p["person_id"]: p["demographics"] for p in ppl}
    assert all(demo[k] == demo[v["donor"]] for k, v in d.items() if v["matched"])
