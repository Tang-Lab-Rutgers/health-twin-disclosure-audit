# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""The main run: population -> twin transcripts -> filter -> auditors -> metrics -> report."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import synthea
from .auditors.api import APIAuditor, api_enabled
from .auditors.base import audit_record, render_transcript
from .auditors.baseline import BaselineAuditor, fit_baseline
from .auditors.lexical import LexicalModel, fit_lexical
from .auditors.llm import LLMAuditor
from .config import (
    read_json,
    run_dir,
    sha256_file,
    work_dir,
    write_json,
    write_jsonl,
)
from .filters import apply_filter, protected_terms
from .generation import CachedGenerator
from .metrics import per_label_auroc, person_accuracy
from .profiles import label_ids, load_population, render_profile
from .questions import questions_for
from .twin import twin_messages

SETTINGS = ["unfiltered", "filtered"]


# ------------------------------------------------------------------ helpers


def transcript_path(rd: Path, population: str, setting: str, pid: str) -> Path:
    base = rd / "transcripts" if population == "target" else rd / "reference" / "transcripts"
    return base / setting / f"{pid}.jsonl"


def assign_donors(people: list[dict]) -> dict[str, dict]:
    """Wrong-person control: give each person another person's transcript.

    Donors come from the same age band and sex where that stratum has at least two people
    (rotate by one within the stratum, sorted by id), so the donor transcript is plausible for
    the demographics the auditor is told. People in singleton strata are rotated among
    themselves. Nobody is ever their own donor.
    """
    strata: dict[tuple, list[str]] = {}
    for p in people:
        strata.setdefault((p["demographics"]["age_band"], p["demographics"]["sex"]), []).append(
            p["person_id"])
    out: dict[str, dict] = {}
    leftovers = []
    for key in sorted(strata):
        ids = sorted(strata[key])
        if len(ids) < 2:
            leftovers += ids
            continue
        for i, pid in enumerate(ids):
            out[pid] = {"donor": ids[(i + 1) % len(ids)], "matched": True}
    leftovers.sort()
    if len(leftovers) == 1:
        # A single unmatched person borrows from the first matched stratum member.
        pid = leftovers[0]
        donor = next(q for q in sorted(out) if q != pid)
        out[pid] = {"donor": donor, "matched": False}
    else:
        for i, pid in enumerate(leftovers):
            out[pid] = {"donor": leftovers[(i + 1) % len(leftovers)], "matched": False}
    assert all(v["donor"] != k for k, v in out.items()), "a person was paired with themselves"
    return out


# ------------------------------------------------------------------ stages


def stage_population(cfg: dict, timings: dict):
    t0 = time.time()
    wd = work_dir(cfg)
    scfg = cfg["synthea"]
    dirs = {pop: wd / "synthea" / pop for pop in ("target", "reference")}
    bundles = {pop: synthea.generate(scfg, pop, d) for pop, d in dirs.items()}
    pops = {pop: load_population(b, scfg["reference_date"], cfg["labels"])
            for pop, b in bundles.items()}
    overlap = {p["person_id"] for p in pops["target"]} & {p["person_id"] for p in pops["reference"]}
    if overlap:
        raise RuntimeError(f"target and reference populations share {len(overlap)} people")
    timings["population_s"] = round(time.time() - t0, 1)
    digests = {pop: synthea.population_digest(d) for pop, d in dirs.items()}
    hashes = {pop: synthea.bundle_hashes(d) for pop, d in dirs.items()}
    return pops["target"], pops["reference"], digests, hashes


def stage_transcripts(cfg: dict, people: list[dict], population: str, gen: CachedGenerator,
                      rd: Path) -> dict:
    """Generate (or load from cache) the twin's answers, then filter them. Returns stats."""
    k = cfg["questions"]["per_person"]
    items = []
    for p in people:
        for q in questions_for(p["person_id"], cfg["question_templates"], k, cfg["seed"]):
            items.append((p, q, gen.chat_prompt(twin_messages(p, q["question"]))))
    outs = gen.generate([it[2] for it in items])
    stats = {"answers": 0, "refused": 0, "redacted_answers": 0, "redactions": 0, "truncated": 0}
    by_person: dict[str, list] = {}
    for p, q, prompt in items:
        by_person.setdefault(p["person_id"], []).append((p, q, outs[prompt]))
    for pid, rows in by_person.items():
        p = rows[0][0]
        terms = protected_terms(p, cfg["synonyms"])
        unf, filt, log = [], [], []
        for _, q, o in rows:
            unf.append({**q, "answer": o["text"], "n_new_tokens": o["n_new_tokens"],
                        "finished": o["finished"]})
            f = apply_filter(o["text"], p, terms, cfg["filter"])
            filt.append({**q, "answer": f["answer"]})
            log.append({"turn": q["turn"], "refused": f["refused"], "redactions": f["redactions"],
                        "matched_terms": f["matched_terms"]})
            stats["answers"] += 1
            stats["refused"] += int(f["refused"])
            stats["redacted_answers"] += int(f["redactions"] > 0)
            stats["redactions"] += f["redactions"]
            stats["truncated"] += int(not o["finished"])
        write_jsonl(transcript_path(rd, population, "unfiltered", pid), unf)
        write_jsonl(transcript_path(rd, population, "filtered", pid), filt)
        # The filter log names the matched protected terms, so it is kept out of the released
        # transcript. It exists for auditing the filter itself.
        base = rd / ("filter_log" if population == "target" else "reference/filter_log")
        write_jsonl(base / f"{pid}.jsonl", log)
    return stats


def load_transcript(rd: Path, population: str, setting: str, pid: str) -> list[dict]:
    from .config import read_jsonl

    return read_jsonl(transcript_path(rd, population, setting, pid))


def fit_models(cfg: dict, reference: list[dict], rd: Path, T: int) -> dict:
    L = label_ids(cfg["labels"])
    mdir = rd / "models"
    base = fit_baseline(reference, L, cfg["auditors"]["baseline"]["min_cell"])
    write_json(mdir / "baseline.json", base)
    y_ref = np.array([[p["labels"][k] for k in L] for p in reference])
    for s in SETTINGS:
        recs, ys = [], []
        for i, p in enumerate(reference):
            rows = load_transcript(rd, "reference", s, p["person_id"])
            for t in range(T + 1):  # train on every prefix, as the auditor is used on every prefix
                recs.append(audit_record(p, render_transcript(rows, t)))
                ys.append(y_ref[i])
        write_json(mdir / f"lexical_{s}.json",
                   fit_lexical(recs, np.array(ys), L, cfg["auditors"]["lexical"], s, base), indent=None)
    recs = [audit_record(p, render_profile(p)) for p in reference]
    write_json(mdir / "lexical_full_record.json",
               fit_lexical(recs, y_ref, L, cfg["auditors"]["lexical"], "full_record", base), indent=None)
    return {f.name: sha256_file(f) for f in sorted(mdir.glob("*.json"))}


def build_tasks(target: list[dict], rd: Path, T: int, donors: dict) -> list[dict]:
    """Every (person, setting, condition, turn) the auditors score, with its input record."""
    tasks = []
    tx = {(s, p["person_id"]): load_transcript(rd, "target", s, p["person_id"])
          for s in SETTINGS for p in target}
    for p in target:
        pid = p["person_id"]
        for s in SETTINGS:
            for t in range(T + 1):
                tasks.append({"pid": pid, "setting": s, "condition": "main", "turn": t,
                              "record": audit_record(p, render_transcript(tx[(s, pid)], t))})
            donor = donors[pid]["donor"]
            tasks.append({"pid": pid, "setting": s, "condition": "wrong_person", "turn": T,
                          "donor": donor,
                          "record": audit_record(p, render_transcript(tx[(s, donor)], T))})
        tasks.append({"pid": pid, "setting": "full_record", "condition": "full_record", "turn": None,
                      "record": audit_record(p, render_profile(p))})
    return tasks


def score_tasks(cfg: dict, tasks: list[dict], rd: Path, models: dict) -> tuple[dict, dict]:
    """Probabilities for every task from every enabled auditor. Returns (probs, auditor_info)."""
    baseline = BaselineAuditor(read_json(rd / "models" / "baseline.json"))
    lex = {s: LexicalModel(read_json(rd / "models" / f"lexical_{s}.json"))
           for s in [*SETTINGS, "full_record"]}
    recs = [t["record"] for t in tasks]
    probs = {"baseline": baseline.predict(recs)}
    info = {"baseline": {"model_file": "models/baseline.json",
                         "model_sha256": models["baseline.json"]}}
    lp = np.zeros_like(probs["baseline"])
    for s, model in lex.items():
        idx = [i for i, t in enumerate(tasks) if t["setting"] == s]
        if idx:
            lp[idx] = model.predict([recs[i] for i in idx])
    probs["lexical"] = lp
    info["lexical"] = {"model_files": {s: f"models/lexical_{s}.json" for s in lex},
                       "model_sha256": {s: models[f"lexical_{s}.json"] for s in lex}}
    acfg = cfg["auditors"]["llm"]
    if acfg.get("enabled"):
        aud = LLMAuditor(acfg, cfg["labels"], baseline, rd / "cache" / "llm_auditor.jsonl")
        p, details = aud.predict(recs, return_details=True)
        probs["llm"] = p
        for t, d in zip(tasks, details, strict=True):
            t["llm"] = {"cache_key": d["key"], "parse_status": d["status"]}
        info["llm"] = {"model": acfg["model"], "revision": acfg["revision"],
                       "backend": acfg.get("backend", "transformers"),
                       "parse": aud.stats}
    if api_enabled(cfg["auditors"]["api"]):
        probs["api"] = APIAuditor(cfg["auditors"]["api"], cfg["labels"], baseline).predict(recs)
        info["api"] = {"provider": cfg["auditors"]["api"]["provider"],
                       "model": cfg["auditors"]["api"]["model"]}
    return probs, info


def collect(cfg: dict, target: list[dict], tasks: list[dict], probs: dict, T: int) -> dict:
    """Reshape task-level probabilities into per-person accuracy and gain arrays."""
    L = label_ids(cfg["labels"])
    thr = cfg["metrics"]["threshold"]
    pidx = {p["person_id"]: i for i, p in enumerate(target)}
    n, nl = len(target), len(L)
    y = np.array([[p["labels"][k] for k in L] for p in target])
    res: dict = {"y": y, "label_ids": L, "auditors": {}}
    base_rows = [i for i, t in enumerate(tasks) if t["condition"] == "full_record"]
    base_p = np.zeros((n, nl))
    for i in base_rows:
        base_p[pidx[tasks[i]["pid"]]] = probs["baseline"][i]
    base_acc = person_accuracy(base_p, y, thr)
    res["baseline"] = {"probs": base_p, "acc": base_acc, "headroom": 100.0 - base_acc}
    for a in probs:
        if a == "baseline":
            continue
        out = {}
        for s in SETTINGS:
            main = np.zeros((n, T + 1, nl))
            wrong = np.zeros((n, nl))
            for i, t in enumerate(tasks):
                if t["setting"] != s:
                    continue
                if t["condition"] == "main":
                    main[pidx[t["pid"]], t["turn"]] = probs[a][i]
                elif t["condition"] == "wrong_person":
                    wrong[pidx[t["pid"]]] = probs[a][i]
            acc = np.stack([person_accuracy(main[:, t], y, thr) for t in range(T + 1)], axis=1)
            out[s] = {"probs": main, "acc": acc, "gain": acc - base_acc[:, None],
                      "wrong_probs": wrong,
                      "wrong_gain": person_accuracy(wrong, y, thr) - base_acc,
                      "auroc_final": per_label_auroc(main[:, T], y, L)}
        full = np.zeros((n, nl))
        for i in base_rows:
            full[pidx[tasks[i]["pid"]]] = probs[a][i]
        out["full_record"] = {"probs": full, "gain": person_accuracy(full, y, thr) - base_acc}
        res["auditors"][a] = out
    return res


def evaluate_controls(cfg: dict, res: dict) -> list[dict]:
    from .metrics import bootstrap_mean_ci

    c, b = cfg["controls"], cfg["metrics"]["bootstrap"]
    head = float(res["baseline"]["headroom"].mean())
    rows = []
    for a, out in res["auditors"].items():
        for s in SETTINGS:
            m, lo, hi = bootstrap_mean_ci(out[s]["gain"][:, 0], b["resamples"], b["seed"])
            rows.append({"control": "empty_transcript", "auditor": a, "setting": s,
                         "mean_gain": float(m), "ci_low": float(lo), "ci_high": float(hi),
                         "criterion": f"|mean gain| <= {c['empty_abs_max']}",
                         "pass": bool(abs(m) <= c["empty_abs_max"])})
            m, lo, hi = bootstrap_mean_ci(out[s]["wrong_gain"], b["resamples"], b["seed"])
            rows.append({"control": "wrong_person", "auditor": a, "setting": s,
                         "mean_gain": float(m), "ci_low": float(lo), "ci_high": float(hi),
                         "criterion": f"mean gain <= {c['wrong_person_max']}",
                         "pass": bool(m <= c["wrong_person_max"])})
        m, lo, hi = bootstrap_mean_ci(out["full_record"]["gain"], b["resamples"], b["seed"])
        need = c["full_record_min_frac_headroom"] * head
        rows.append({"control": "full_record", "auditor": a, "setting": "full_record",
                     "mean_gain": float(m), "ci_low": float(lo), "ci_high": float(hi),
                     "criterion": f"mean gain >= {c['full_record_min_frac_headroom']} x mean "
                                  f"headroom ({need:.1f})",
                     "pass": bool(m >= need)})
    return rows


def run(config_path: str, stop_after: str | None = None) -> int:
    from . import report
    from .config import load_config

    t_all = time.time()
    cfg = load_config(config_path)
    if cfg["labels"] is None:
        raise RuntimeError(f"{cfg['labels_file']} is missing; run `python -m hta freeze-labels` once")
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    write_json(rd / "config.resolved.json", {k: v for k, v in cfg.items() if not k.startswith("_")}
               | {"_config_hash": cfg["_config_hash"], "_config_path": cfg["_config_path"]})
    timings: dict = {}
    T = cfg["questions"]["per_person"]

    target, reference, digests, hashes = stage_population(cfg, timings)
    write_jsonl(rd / "profiles" / "target.jsonl", target)
    write_jsonl(rd / "profiles" / "reference.jsonl", reference)
    write_json(rd / "synthea_sha256.json", hashes)
    print(f"[run] target {len(target)} people, reference {len(reference)} people, {T} turns")

    t0 = time.time()
    gen = CachedGenerator(cfg["twin"], rd / "cache" / "twin.jsonl", "twin")
    filter_stats = {"reference": stage_transcripts(cfg, reference, "reference", gen, rd),
                    "target": stage_transcripts(cfg, target, "target", gen, rd)}
    timings["twin_s"] = round(time.time() - t0, 1)
    print(f"[run] filter: {filter_stats}")
    if stop_after == "twin":
        print("[run] stopping after the twin stage (--stop-after twin)")
        return 0

    t0 = time.time()
    models = fit_models(cfg, reference, rd, T)
    donors = assign_donors(target)
    tasks = build_tasks(target, rd, T, donors)
    probs, info = score_tasks(cfg, tasks, rd, models)
    timings["audit_s"] = round(time.time() - t0, 1)

    res = collect(cfg, target, tasks, probs, T)
    controls = evaluate_controls(cfg, res)
    timings["total_s"] = round(time.time() - t_all, 1)
    report.write_all(cfg, rd, target, tasks, res, controls, info, donors, filter_stats, digests,
                     models, timings)
    failed = [c for c in controls if not c["pass"]]
    print(f"[run] done in {timings['total_s'] / 60:.1f} min; results in {rd}")
    if failed:
        print("[run] CONTROL FAILURES:")
        for c in failed:
            print(f"   {c['control']} / {c['auditor']} / {c['setting']}: mean gain "
                  f"{c['mean_gain']:.2f} (criterion {c['criterion']})")
    return 0
