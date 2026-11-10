# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Re-compute one person's disclosure score from a transcript and compare it with the report.

    python -m hta verify --person results/demo/per_person/<id>.json \
                         --transcript results/demo/transcripts/filtered/<id>.jsonl

What is checked, in order:
  1. the transcript's SHA256 is one the report says it was computed from (this picks the setting);
  2. the run config and the auditor model files are the ones the report names (SHA256);
  3. for every turn, the baseline and lexical-auditor accuracy and gain, recomputed from the
     transcript, equal the stored values exactly;
  4. the LLM auditor's stored scores (Yes/No probabilities, or raw JSON answers) are turned into
     probabilities again and its gain recomputed (exact);
  5. optionally (--rerun-llm), the LLM auditor is run again on the transcript. This is reported,
     not required to match: a different CPU can change floating-point sums slightly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .auditors.base import audit_record, render_transcript
from .auditors.baseline import BaselineAuditor
from .auditors.lexical import LexicalModel
from .config import read_json, read_jsonl, sha256_file
from .metrics import person_accuracy


def _r(v) -> float:
    return float(round(float(v), 6))


def verify(person_file: str, transcript_file: str, rerun_llm: bool = False) -> int:
    person_path = Path(person_file)
    rep = read_json(person_path)
    rd = person_path.resolve().parent.parent
    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"  [{'OK' if passed else 'MISMATCH'}] {name}{(': ' + detail) if detail else ''}")

    print(f"Verifying person {rep['person_id']} (run '{rep['run_name']}')")
    tsha = sha256_file(transcript_file)
    setting = next((s for s, e in rep["settings"].items() if e["transcript_sha256"] == tsha), None)
    check("transcript SHA256 matches a stored setting", setting is not None,
          f"{tsha[:16]}... -> {setting or 'none'}")
    if setting is None:
        print("RESULT: MISMATCH (this is not the transcript the score was computed from)")
        return 1

    cfg = read_json(rd / "config.resolved.json")
    check("run config hash", cfg["_config_hash"] == rep["config_hash"], rep["config_hash"][:16])
    entry = rep["settings"][setting]
    rows = read_jsonl(transcript_file)
    person = {"person_id": rep["person_id"], "demographics": rep["demographics"]}
    # Label order comes from the frozen label file, never from the report's dict (JSON key order
    # is not meaningful), and must equal the order the model was trained with.
    label_ids = [c["id"] for c in cfg["labels"]["conditions"]] + \
        [m["id"] for m in cfg["labels"]["medication_classes"]]
    y = np.array([[rep["labels"][k] for k in label_ids]])
    thr = rep["threshold"]
    T = len(rows)
    records = [audit_record(person, render_transcript(rows, t)) for t in range(T + 1)]

    bfile = rd / rep["baseline"]["model_file"]
    check("baseline model SHA256", sha256_file(bfile) == rep["baseline"]["model_sha256"])
    base = BaselineAuditor(read_json(bfile))
    base_p = base.predict(records[:1])
    base_acc = float(person_accuracy(base_p, y, thr)[0])
    check("baseline accuracy", _r(base_acc) == rep["baseline"]["accuracy"],
          f"{base_acc:.2f} vs stored {rep['baseline']['accuracy']:.2f}")

    lex = entry["auditors"]["lexical"]
    lfile = rd / lex["model_file"]
    check("lexical model SHA256", sha256_file(lfile) == lex["model_sha256"])
    lparams = read_json(lfile)
    check("label order of lexical model equals the frozen label order",
          lparams["label_ids"] == label_ids)
    lp = LexicalModel(lparams).predict(records)
    acc = person_accuracy(lp, np.repeat(y, T + 1, axis=0), thr)
    gain = [_r(v) for v in acc - base_acc]
    check(f"lexical gain at every turn 0..{T}", gain == lex["gain"],
          "recomputed " + " ".join(f"{g:+.1f}" for g in gain))

    llm = entry["auditors"].get("llm")
    if llm:
        from .pipeline import make_llm_auditor

        cache = {r["key"]: r for r in read_jsonl(rd / llm["cache_file"])}
        probs = []
        if llm.get("method", "yes_no") == "yes_no":
            from .auditors.yesno import YesNoAuditor

            yn = YesNoAuditor.__new__(YesNoAuditor)  # only the stored-score path, no model
            yn.baseline, yn.min_mass = base, float(cfg["auditors"]["llm"].get("min_mass", 0.5))
            for key, rec in zip(llm["cache_keys"], records, strict=True):
                probs.append(yn.to_probs(rec, cache[key])[0])
        else:
            from .auditors.llm import label_keys, parse_answer

            short = [k for _, k, _ in label_keys(cfg["labels"])]
            prior = base.prior(person["demographics"]["age_band"], person["demographics"]["sex"])
            for key in llm["cache_keys"]:
                vals, _ = parse_answer("{" + cache[key]["text"], short)
                probs.append([vals.get(k, prior[j]) for j, k in enumerate(short)])
        acc = person_accuracy(np.array(probs), np.repeat(y, T + 1, axis=0), thr)
        g = [_r(v) for v in acc - base_acc]
        check(f"LLM gain from stored raw answers, turns 0..{T}", g == llm["gain"],
              "recomputed " + " ".join(f"{v:+.1f}" for v in g))
        if rerun_llm:
            fresh = rd / "cache" / "verify_rerun_llm.jsonl"
            fresh.unlink(missing_ok=True)
            aud = make_llm_auditor(cfg["auditors"]["llm"], cfg["labels"], base, fresh)
            p2 = aud.predict(records)
            d = float(np.abs(p2 - np.array(probs)).max())
            g2 = [_r(v) for v in person_accuracy(p2, np.repeat(y, T + 1, axis=0), thr) - base_acc]
            print(f"  [INFO] re-ran the LLM auditor: max probability difference {d:.2e}; "
                  f"gain identical at {sum(a == b for a, b in zip(g2, g, strict=True))} of {T + 1} turns")

    stored = entry["auditors"]["lexical"]
    print(f"Lexical auditor, {setting}: final gain {stored['gain'][-1]:+.1f} points, "
          f"crossed {rep['crossing_points']} points: {stored['crossed']}")
    print(f"RESULT: {'MATCH' if ok else 'MISMATCH'}")
    return 0 if ok else 1
