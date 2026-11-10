# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Choose the LLM auditor before any target-population result exists.

Runs each candidate model on the raw profile text of the first 20 REFERENCE people (sorted by
id; the target population is never touched) and reports the same full-record criterion the main
control uses: mean gain over the demographics-only baseline as a share of the mean headroom.

    python scripts/auditor_precheck.py --profiles results/demo/profiles/reference.jsonl \
        --device cuda --out docs/auditor_precheck.json

The rule, fixed before running: take the smallest candidate whose share is >= 0.5 and whose
unusable-answer rate is < 20% (json: answers with no usable number; yes_no: labels whose Yes+No
probability mass is below min_mass). First run: --method json (all four failed, see
docs/auditor_precheck_json.json). Second run: --method yes_no.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hta.auditors.base import audit_record
from hta.auditors.baseline import BaselineAuditor, fit_baseline
from hta.config import load_config, read_jsonl
from hta.pipeline import make_llm_auditor
from hta.profiles import label_ids, render_profile

CANDIDATES = [
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951", 0.36),
    ("allenai/OLMo-2-0425-1B-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e", 1.48),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674", 1.71),
    ("allenai/OLMo-2-1124-7B-Instruct", "470b1fba1ae01581f270116362ee4aa1b97f4c84", 7.30),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="yes_no", choices=["yes_no", "json"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    ref = read_jsonl(args.profiles)
    L = label_ids(cfg["labels"])
    base = BaselineAuditor(fit_baseline(ref, L, cfg["auditors"]["baseline"]["min_cell"]))
    sample = sorted(ref, key=lambda p: p["person_id"])[: args.n]
    recs = [audit_record(p, render_profile(p)) for p in sample]
    y = np.array([[p["labels"][k] for k in L] for p in sample])
    thr = cfg["metrics"]["threshold"]
    bacc = ((base.predict(recs) > thr) == y).mean(1) * 100
    results = []
    for model, rev, size_b in CANDIDATES:
        acfg = dict(cfg["auditors"]["llm"], model=model, revision=rev, device=args.device,
                    batch_size=args.batch_size, method=args.method)
        cache = Path(args.out).with_suffix(f".{model.split('/')[1]}.jsonl")
        aud = make_llm_auditor(acfg, cfg["labels"], base, cache)
        t0 = time.time()
        probs, det = aud.predict(recs, return_details=True)
        acc = ((probs > thr) == y).mean(1) * 100
        gain, head = float((acc - bacc).mean()), float((100 - bacc).mean())
        r = {"model": model, "revision": rev, "params_billion": size_b,
             "seconds_per_prompt": round((time.time() - t0) / len(recs), 2), "device": args.device,
             "mean_gain": round(gain, 2), "mean_headroom": round(head, 2),
             "share_of_headroom": round(gain / head, 3),
             "method": args.method, "parse": aud.stats, "example_raw": det[0]["raw"][:300]}
        if args.method == "json":
            unusable = aud.stats["failed"] / len(recs)
        else:
            unusable = aud.stats["labels_fallback"] / max(aud.stats["labels_total"], 1)
        r["unusable_rate"] = round(unusable, 3)
        r["passes"] = r["share_of_headroom"] >= 0.5 and unusable < 0.2
        print(json.dumps(r), flush=True)
        results.append(r)
    chosen = next((r["model"] for r in results if r["passes"]), None)
    Path(args.out).write_text(json.dumps({"n_reference_people": len(sample), "method": args.method, "rule":
        "smallest candidate with share_of_headroom >= 0.5 and unusable rate < 0.2",
        "chosen": chosen, "results": results}, indent=2) + "\n")
    print("chosen:", chosen)


if __name__ == "__main__":
    main()
