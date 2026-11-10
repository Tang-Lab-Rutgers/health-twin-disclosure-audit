# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Demographics-only baseline: label prevalence by age band and sex in the reference population."""

from __future__ import annotations

import numpy as np


def fit_baseline(reference: list[dict], label_ids: list[str], min_cell: int) -> dict:
    """Prevalence tables. A cell with fewer than `min_cell` people falls back to the sex-level
    prevalence, and that to the overall prevalence."""
    def prev(people):
        y = np.array([[p["labels"][k] for k in label_ids] for p in people], dtype=float)
        return y.mean(axis=0).tolist(), len(people)

    overall, n = prev(reference)
    by_sex, by_cell = {}, {}
    for s in sorted({p["demographics"]["sex"] for p in reference}):
        grp = [p for p in reference if p["demographics"]["sex"] == s]
        by_sex[s] = {"prevalence": prev(grp)[0], "n": len(grp)}
    for key in sorted({(p["demographics"]["age_band"], p["demographics"]["sex"]) for p in reference}):
        grp = [p for p in reference if (p["demographics"]["age_band"], p["demographics"]["sex"]) == key]
        by_cell[f"{key[0]}|{key[1]}"] = {"prevalence": prev(grp)[0], "n": len(grp)}
    return {"kind": "baseline", "label_ids": label_ids, "min_cell": min_cell,
            "overall": {"prevalence": overall, "n": n}, "by_sex": by_sex, "by_cell": by_cell}


class BaselineAuditor:
    name = "baseline"

    def __init__(self, params: dict):
        self.p = params

    def prior(self, age_band: str, sex: str) -> list[float]:
        cell = self.p["by_cell"].get(f"{age_band}|{sex}")
        if cell and cell["n"] >= self.p["min_cell"]:
            return cell["prevalence"]
        s = self.p["by_sex"].get(sex)
        if s and s["n"] >= self.p["min_cell"]:
            return s["prevalence"]
        return self.p["overall"]["prevalence"]

    def predict(self, records: list[dict]) -> np.ndarray:
        return np.array([self.prior(r["age_band"], r["sex"]) for r in records], dtype=float)
