# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Compare a fresh run with the committed expected outputs (results/expected).

Tier A must match exactly everywhere: Synthea files, profiles and labels, question order, the
filter applied to the run's own unfiltered answers, and the lexical/baseline scores recomputed
from the run's own transcripts (checked by `hta verify` on every person).

Tier B is the LLM-generated text. With the same Docker image on the same CPU architecture it is
expected to match exactly. Across CPUs, floating-point differences can flip a greedy token, after
which the answer diverges; the run is then compared on the headline numbers within the stated
tolerance instead.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .config import read_json, read_jsonl, sha256_file

TOLERANCE = {"gain_points": 3.0, "crossing_rate": 0.10}


def _summary(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return {(r["section"], r["auditor"], r["setting"], r["quantity"]): r for r in csv.DictReader(f)}


def compare(expected: str, actual: str) -> int:
    e, a = Path(expected), Path(actual)
    hard_fail = False
    print(f"Comparing {a} against {e}")

    def line(tag: str, msg: str) -> None:
        print(f"  [{tag}] {msg}")

    # ---- tier A
    se, sa = read_json(e / "synthea_sha256.json"), read_json(a / "synthea_sha256.json")
    n = sum(len(v) for v in se.values())
    same = sum(se[p].get(k) == sa.get(p, {}).get(k) for p in se for k in se[p])
    ok = same == n and se == sa
    hard_fail |= not ok
    line("OK" if ok else "FAIL", f"Synthea FHIR files identical: {same}/{n}")
    for pop in ("target", "reference"):
        pe, pa = e / "profiles" / f"{pop}.jsonl", a / "profiles" / f"{pop}.jsonl"
        ok = sha256_file(pe) == sha256_file(pa)
        hard_fail |= not ok
        line("OK" if ok else "FAIL", f"{pop} profiles and labels identical")

    # ---- tier B: transcripts
    totals = {}
    for s in ("unfiltered", "filtered"):
        files = sorted((e / "transcripts" / s).glob("*.jsonl"))
        n_same_file = n_same_ans = n_ans = 0
        q_order_ok = True
        for fe in files:
            fa = a / "transcripts" / s / fe.name
            re_, ra = read_jsonl(fe), read_jsonl(fa)
            q_order_ok &= [r["question_id"] for r in re_] == [r["question_id"] for r in ra]
            n_same_file += sha256_file(fe) == sha256_file(fa)
            for x, y in zip(re_, ra, strict=True):
                n_ans += 1
                n_same_ans += x["answer"] == y["answer"]
        hard_fail |= not q_order_ok
        if s == "unfiltered":
            line("OK" if q_order_ok else "FAIL", "question order identical for every person")
        totals[s] = (n_same_ans, n_ans)
        tag = "OK" if n_same_file == len(files) else "DRIFT"
        line(tag, f"{s} transcripts identical: {n_same_file}/{len(files)} files, "
                  f"{n_same_ans}/{n_ans} answers")
    exact_text = all(x == y for x, y in totals.values())

    # ---- headline numbers
    me, ma = _summary(e / "summary.csv"), _summary(a / "summary.csv")
    worst = 0.0
    for key, re_ in me.items():
        if key not in ma or re_["value"] in ("", None):
            continue
        ve, va = float(re_["value"]), float(ma[key]["value"])
        if exact_text and key[1] in ("lexical", "baseline", ""):
            ok = ve == va
            hard_fail |= not ok
            if not ok:
                line("FAIL", f"{key}: expected {ve}, got {va}")
            continue
        tol = TOLERANCE["crossing_rate"] if "rate" in key[3] else TOLERANCE["gain_points"]
        d = abs(ve - va)
        worst = max(worst, d / tol)
        if d > tol:
            hard_fail = True
            line("FAIL", f"{key}: expected {ve:.3f}, got {va:.3f} (tolerance {tol})")
    if exact_text:
        line("OK" if not hard_fail else "FAIL",
             "all summary numbers identical" if not hard_fail else "summary numbers differ")
    else:
        line("OK" if not hard_fail else "FAIL",
             f"summary numbers within tolerance {TOLERANCE} (worst {worst:.0%} of tolerance)")
    print("RESULT:", "FAIL" if hard_fail else ("MATCH" if exact_text else "MATCH WITHIN TOLERANCE"))
    return 1 if hard_fail else 0
