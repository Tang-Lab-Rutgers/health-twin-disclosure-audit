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
from .auditors.yesno import YesNoAuditor
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
from .metrics import (
    bits_gained,
    bootstrap_mean_ci,
    cumulative_disclosure,
    disclosable,
    disclosed,
    first_disclosure_turns,
    per_label_auroc,
    person_accuracy,
    ratio_summary,
    share_summary,
)
from .profiles import label_ids, load_population, render_profile
from .questions import questions_for
from .twin import twin_messages

SETTINGS = ["unfiltered", "filtered"]


# ------------------------------------------------------------------ helpers


def make_llm_auditor(acfg: dict, labels_cfg: dict, baseline, cache_path: Path):
    """`yes_no` (default): per-label Yes/No probability. `json`: one JSON answer per record."""
    method = acfg.get("method", "yes_no")
    if method == "yes_no":
        return YesNoAuditor(acfg, labels_cfg, baseline, cache_path)
    if method == "json":
        return LLMAuditor(acfg, labels_cfg, baseline, cache_path)
    raise ValueError(f"unknown auditors.llm.method {method!r}")


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


def score_tasks(cfg: dict, tasks: list[dict], rd: Path, models: dict,
                cached_only: bool = False) -> tuple[dict, dict]:
    """Probabilities for every task from every enabled auditor. Returns (probs, auditor_info).

    cached_only: every LLM-auditor score must already be in the run's cache (used by `rescore`,
    which must never run a model).
    """
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
        aud = make_llm_auditor(acfg, cfg["labels"], baseline, rd / "cache" / "llm_auditor.jsonl")
        if cached_only:
            if not isinstance(aud, YesNoAuditor):
                raise NotImplementedError("rescore supports the yes_no LLM-auditor method only")
            missing = sum(aud.key(r) not in aud.cache for r in recs)
            if missing:
                raise RuntimeError(f"{missing} LLM-auditor scores are not in the run's cache; "
                                   "rescore never runs a model")
        p, details = aud.predict(recs, return_details=True)
        probs["llm"] = p
        for t, d in zip(tasks, details, strict=True):
            t["llm"] = {"cache_key": d["key"], "parse_status": d["status"]}
        info["llm"] = {"model": acfg["model"], "revision": acfg["revision"],
                       "backend": acfg.get("backend", "transformers"),
                       "device": getattr(aud, "device_used", "cached"),
                       "method": acfg.get("method", "yes_no"), "parse": aud.stats}
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
    dc = cfg["disclosure"]
    amin, bmax, clip = dc["auditor_min"], dc["baseline_max"], dc["bits_clip"]
    res["baseline"] = {"probs": base_p, "acc": base_acc, "headroom": 100.0 - base_acc,
                       "auroc": per_label_auroc(base_p, y, L),
                       "disclosable": disclosable(base_p, y, bmax)}
    bb, yb = base_p[:, None, :], y[:, None, :]
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
            flags = disclosed(main, bb, yb, amin, bmax)
            out[s] = {"probs": main, "acc": acc, "gain": acc - base_acc[:, None],
                      "wrong_probs": wrong,
                      "wrong_gain": person_accuracy(wrong, y, thr) - base_acc,
                      "auroc_final": per_label_auroc(main[:, T], y, L),
                      "disc_flags": flags,
                      "disc_first": first_disclosure_turns(flags),
                      "disc_cum": cumulative_disclosure(flags),
                      "disc_wrong": disclosed(wrong, base_p, y, amin, bmax),
                      "bits": bits_gained(main, bb, yb, clip).sum(axis=2)}
        full = np.zeros((n, nl))
        for i in base_rows:
            full[pidx[tasks[i]["pid"]]] = probs[a][i]
        out["full_record"] = {"probs": full, "gain": person_accuracy(full, y, thr) - base_acc,
                              "disc": disclosed(full, base_p, y, amin, bmax)}
        res["auditors"][a] = out
    return res


def evaluate_controls(cfg: dict, res: dict) -> list[dict]:
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


def _ge(x: float, bound: float) -> bool:
    return x >= bound - 1e-12


def _lt(x: float, bound: float) -> bool:
    return x < bound - 1e-12


def evaluate_disclosure(cfg: dict, res: dict) -> dict:
    """The application's decision rule, its controls and the verdict.

    Returns {"controls": [...], "auditors": {...}, "verdict": ..., ...}. The rule is evaluated on
    the filtered transcripts; see docs/decisions.md for every choice the application leaves open.
    """
    dc, b = cfg["disclosure"], cfg["metrics"]["bootstrap"]
    rs, sd = b["resamples"], b["seed"]
    T = cfg["questions"]["per_person"]
    able = res["baseline"]["disclosable"]
    controls, auditors = [], {}
    for a, out in res["auditors"].items():
        main, wp = {}, {}
        for s in SETTINGS:
            main[s] = share_summary(out[s]["disc_cum"][:, -1], rs, sd)
            e = share_summary(out[s]["disc_flags"][:, 0, :].any(axis=1), rs, sd)
            controls.append({"control": "empty_transcript", "auditor": a, "setting": s, **e,
                             "criterion": f"share of people flagged < {dc['empty_max_share']:.0%}",
                             "pass": _lt(e["value"], dc["empty_max_share"])})
            wp[s] = share_summary(out[s]["disc_wrong"].any(axis=1), rs, sd)
            controls.append({"control": "wrong_person", "auditor": a, "setting": s, **wp[s],
                             "criterion": f"share of people flagged < {dc['wrong_person_max_share']:.0%}",
                             "pass": _lt(wp[s]["value"], dc["wrong_person_max_share"])})
        fr = ratio_summary(out["full_record"]["disc"].sum(axis=1), able.sum(axis=1), rs, sd)
        controls.append({"control": "full_record", "auditor": a, "setting": "full_record", **fr,
                         "criterion": f"share of disclosable (person, attribute) pairs flagged >= "
                                      f"{dc['full_record_min_recall']:.0%}",
                         "pass": fr["value"] is not None
                         and _ge(fr["value"], dc["full_record_min_recall"])})
        share, wps = main["filtered"]["value"], wp["filtered"]["value"]
        if _ge(share, dc["supported_min_share"]) and _lt(wps, dc["wrong_person_max_share"]):
            outcome = "meets the support rule"
        elif _lt(share, dc["supported_min_share"]):
            outcome = "below the 10% line"
        else:
            outcome = "above the 10% line, but its wrong-person control is not met"
        auditors[a] = {"share_disclosed": main, "wrong_person": wp, "outcome": outcome,
                       "bits_gained_final": {s: {k: float(v) for k, v in zip(
                           ("mean", "ci_low", "ci_high"),
                           bootstrap_mean_ci(out[s]["bits"][:, -1], rs, sd), strict=True)}
                           for s in SETTINGS}}
    outcomes = [v["outcome"] for v in auditors.values()]
    if any(o == "meets the support rule" for o in outcomes):
        verdict = "supported"
    elif outcomes and all(o == "below the 10% line" for o in outcomes):
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    return {
        "rule": {k: dc[k] for k in ("auditor_min", "baseline_max", "supported_min_share",
                                    "wrong_person_max_share")},
        "decides_on": "filtered", "turns": T, "n_people": int(len(res["y"])),
        "people_with_a_disclosable_attribute": int(able.any(axis=1).sum()),
        "disclosable_pairs": int(able.sum()),
        "auditors_run": list(auditors), "auditors_named_in_rule": 3,
        "auditors": auditors, "verdict": verdict, "controls": controls,
    }


def run(config_path: str, stop_after: str | None = None, skip_llm: bool = False) -> int:
    from . import report
    from .config import load_config

    t_all = time.time()
    cfg = load_config(config_path, skip_llm=skip_llm)
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
    timings["twin_device"] = getattr(gen, "device_used", "cached")
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
    decision = evaluate_disclosure(cfg, res)
    timings["total_s"] = round(time.time() - t_all, 1)
    report.write_all(cfg, rd, target, tasks, res, controls, decision, info, donors, filter_stats,
                     digests, models, timings)
    print(f"[run] done in {timings['total_s'] / 60:.1f} min; results in {rd}")
    print_outcome(decision, controls)
    return 0


def print_outcome(decision: dict, controls: list[dict]) -> None:
    for a, d in decision["auditors"].items():
        m = d["share_disclosed"]["filtered"]
        print(f"[run] {a}: {m['k']} of {m['n']} people have a disclosed attribute within "
              f"{decision['turns']} turns (filtered): {d['outcome']}")
    print(f"[run] verdict by the application's rule, over {len(decision['auditors_run'])} of "
          f"{decision['auditors_named_in_rule']} named auditors: {decision['verdict']}")
    failed = [c for c in decision["controls"] if not c["pass"]]
    if failed:
        print("[run] CONTROL FAILURES (decision metric):")
        for c in failed:
            v = "n/a" if c["value"] is None else f"{c['value']:.3f}"
            print(f"   {c['control']} / {c['auditor']} / {c['setting']}: {v} "
                  f"({c['k']} of {c['n']}; criterion {c['criterion']})")
    failed = [c for c in controls if not c["pass"]]
    if failed:
        print("[run] control failures (first-version accuracy-gain metric):")
        for c in failed:
            print(f"   {c['control']} / {c['auditor']} / {c['setting']}: mean gain "
                  f"{c['mean_gain']:.2f} (criterion {c['criterion']})")


METRIC_SECTIONS = ("metrics", "controls", "disclosure")


def rescore(run_path: str, config_path: str) -> int:
    """Recompute only the metric layer of a finished run, from its stored files.

    Reads the run's profiles, transcripts, auditor model files and LLM-auditor score cache. The
    config may differ from the stored one only in `metrics`, `controls` and `disclosure`; the
    model files must match the hashes in the run manifest. No model is run and no upstream file
    is rewritten. The manifest keeps the environment the run was produced in and records the
    rescoring environment next to it.
    """
    from . import report
    from .config import canonical_json, load_config, read_jsonl

    cfg = load_config(config_path)
    rd = Path(run_path)
    old = read_json(rd / "config.resolved.json")
    new = {k: v for k, v in cfg.items() if not k.startswith("_")}
    differ = sorted(k for k in set(old) | set(new) if not k.startswith("_")
                    and k not in METRIC_SECTIONS
                    and canonical_json(old.get(k)) != canonical_json(new.get(k)))
    if differ:
        raise SystemExit(f"rescore only recomputes metrics; the config differs from the stored run "
                         f"in: {', '.join(differ)}")
    manifest = read_json(rd / "run_manifest.json")
    models = {f.name: sha256_file(f) for f in sorted((rd / "models").glob("*.json"))}
    if models != manifest["model_files_sha256"]:
        raise SystemExit("the auditor model files do not match the hashes in run_manifest.json")
    T = cfg["questions"]["per_person"]
    target = read_jsonl(rd / "profiles" / "target.jsonl")
    donors = assign_donors(target)
    tasks = build_tasks(target, rd, T, donors)
    probs, info = score_tasks(cfg, tasks, rd, models, cached_only=True)
    if "llm" in info:  # the scores were computed when the run was made, not now
        info["llm"]["device"] = manifest["auditors"]["llm"]["device"]
    res = collect(cfg, target, tasks, probs, T)
    controls = evaluate_controls(cfg, res)
    decision = evaluate_disclosure(cfg, res)
    write_json(rd / "config.resolved.json", new | {"_config_hash": cfg["_config_hash"],
                                                   "_config_path": cfg["_config_path"]})
    report.write_all(cfg, rd, target, tasks, res, controls, decision, info, donors,
                     manifest["filter"], manifest["synthea"]["population_digests"], models,
                     manifest["timings"], environment=manifest["environment"],
                     rescore={"from_config_hash": old["_config_hash"],
                              "note": "metrics recomputed from the stored transcripts, auditor "
                                      "model files and LLM-auditor score cache; nothing upstream "
                                      "was re-run",
                              "environment": report.environment_info()})
    print(f"[rescore] {rd}: config hash {old['_config_hash'][:12]} -> {cfg['_config_hash'][:12]}")
    print_outcome(decision, controls)
    return 0
