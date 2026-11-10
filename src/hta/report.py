# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Figures, summary CSV, per-person JSON reports, run manifest and a plain-text report."""

from __future__ import annotations

import csv
import importlib.metadata as md
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import REPO_ROOT, sha256_file, write_json
from .metrics import first_crossing_turn, summarize_curve

SETTINGS = ["unfiltered", "filtered"]
AUDITOR_LABEL = {"lexical": "Lexical (TF-IDF + LR)", "llm": "Open-weight LLM", "api": "API LLM"}
# Categorical slots 1-3 of the reference palette (validated all-pairs, light surface).
AUDITOR_COLOR = {"lexical": "#2a78d6", "llm": "#eb6834", "api": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"


def _f(x: float) -> float:
    return float(round(float(x), 6))


# ------------------------------------------------------------------ figure


def plot_curves(cfg: dict, res: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = cfg["questions"]["per_person"]
    thr = cfg["metrics"]["crossing_points"]
    turns = np.arange(T + 1)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=SURFACE)
    lo_all, hi_all = [0.0, float(thr)], []
    for ax, s in zip(axes, SETTINGS, strict=True):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=GRID, lw=1, zorder=1)
        ax.axhline(thr, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(0.2, thr + 0.6, f"{thr}-point line", color=INK_2, fontsize=8.5, va="bottom")
        for a, out in res["auditors"].items():
            summ = summarize_curve(out[s]["gain"], cfg["metrics"])
            m, lo, hi = map(np.array, (summ["mean"], summ["ci_low"], summ["ci_high"]))
            lo_all.append(lo.min())
            hi_all.append(hi.max())
            col = AUDITOR_COLOR.get(a, INK)
            ax.fill_between(turns, lo, hi, color=col, alpha=0.14, lw=0, zorder=2)
            ax.plot(turns, m, color=col, lw=2, marker="o", ms=4.5, zorder=3,
                    label=AUDITOR_LABEL.get(a, a))
            ax.annotate(f"{m[-1]:+.1f}", (turns[-1], m[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK)
        ax.set_title(f"{s.capitalize()} answers", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Turn (answers seen by the app)")
        ax.set_xticks(turns)
        ax.set_xlim(-0.3, T + 1.2)
        ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Disclosure gain over baseline (accuracy points)")
    axes[0].set_ylim(min(lo_all) - 3, max([*hi_all, thr]) + 4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.01), labelcolor=INK)
    n = len(res["y"])
    fig.suptitle(f"Mean disclosure gain per turn, {n} synthetic people, 95% bootstrap CI",
                 x=0.01, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out_dir / "curves.png", dpi=200, facecolor=SURFACE)
    fig.savefig(out_dir / "curves.pdf", facecolor=SURFACE, metadata={"CreationDate": None})
    plt.close(fig)


# ------------------------------------------------------------------ manifest


def environment_info() -> dict:
    pkgs = ["torch", "transformers", "tokenizers", "safetensors", "huggingface-hub",
            "scikit-learn", "scipy", "numpy", "matplotlib", "PyYAML", "vllm"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    git = {}
    try:
        git["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       capture_output=True, text=True).stdout.strip()
        git["dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "configs",
                                            "data"], cwd=REPO_ROOT, capture_output=True,
                                           text=True).stdout.strip())
    except OSError:
        pass
    if not git.get("commit") and os.environ.get("HTA_GIT_COMMIT"):  # run from an exported copy
        git = {"commit": os.environ["HTA_GIT_COMMIT"], "dirty": False, "source": "HTA_GIT_COMMIT"}
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": sys.version.split()[0], "platform": platform.platform(), "gpu": gpu,
        "machine": platform.machine(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "packages": versions, "git": git,
    }


# ------------------------------------------------------------------ main writer


def write_all(cfg, rd: Path, target, tasks, res, controls, info, donors, filter_stats, digests,
              models, timings) -> None:
    T = cfg["questions"]["per_person"]
    L = res["label_ids"]
    mcfg = cfg["metrics"]
    pts = mcfg["crossing_points"]
    plot_curves(cfg, res, rd)

    # ---------------------------------------------------------- summary.csv
    rows = []

    def add(section, auditor, setting, quantity, value, lo=None, hi=None, n=None, ok=None, note=""):
        rows.append({"section": section, "auditor": auditor, "setting": setting,
                     "quantity": quantity, "value": None if value is None else _f(value),
                     "ci_low": None if lo is None else _f(lo), "ci_high": None if hi is None else _f(hi),
                     "n": n, "pass": "" if ok is None else ("PASS" if ok else "FAIL"), "note": note})

    n = len(target)
    head = res["baseline"]["headroom"]
    add("baseline", "baseline", "", "mean_accuracy", res["baseline"]["acc"].mean(), n=n)
    add("baseline", "baseline", "", "mean_headroom_points", head.mean(), n=n,
        note="100 minus baseline accuracy: the largest gain possible for that person")
    add("baseline", "baseline", "", f"share_with_headroom_ge_{pts}", (head >= pts).mean(), n=n,
        note=f"only these people can cross {pts} points at all")
    summaries = {}
    for a, out in res["auditors"].items():
        for s in SETTINGS:
            sm = summarize_curve(out[s]["gain"], mcfg)
            summaries[(a, s)] = sm
            add("main", a, s, f"gain_turn_{T}", sm["final"]["mean"], sm["final"]["ci_low"],
                sm["final"]["ci_high"], n)
            cr = sm["crossing_rate"]
            add("main", a, s, f"crossing_rate_{pts}pt", cr["mean"], cr["ci_low"], cr["ci_high"], n,
                note=f"{cr['n_crossed']} of {cr['n']} people")
    for c in controls:
        add("control", c["auditor"], c["setting"], c["control"], c["mean_gain"], c["ci_low"],
            c["ci_high"], n, c["pass"], c["criterion"])
    for pop, st in filter_stats.items():
        add("filter", "", pop, "refusal_rate", st["refused"] / max(st["answers"], 1), n=st["answers"])
        add("filter", "", pop, "redacted_answer_rate", st["redacted_answers"] / max(st["answers"], 1),
            n=st["answers"])
        add("filter", "", pop, "truncated_answer_rate", st["truncated"] / max(st["answers"], 1),
            n=st["answers"], note=f"answers that hit max_new_tokens={cfg['twin']['max_new_tokens']}")
    if "llm" in info:
        ps = info["llm"]["parse"]
        if "calls" in ps:  # json method
            add("llm_parse", "llm", "", "failed_answer_rate", ps["failed"] / max(ps["calls"], 1),
                n=ps["calls"], note="no usable number at all; every label fell back to the baseline")
            add("llm_parse", "llm", "", "partial_answer_rate", ps["partial"] / max(ps["calls"], 1),
                n=ps["calls"])
        add("llm_parse", "llm", "", "label_fallback_rate",
            ps["labels_fallback"] / max(ps["labels_total"], 1), n=ps["labels_total"],
            note="labels that fell back to the baseline value")
    with open(rd / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---------------------------------------------------------- per-person reports
    pdir = rd / "per_person"
    pdir.mkdir(exist_ok=True)
    task_llm = {}
    for t in tasks:
        if "llm" in t:
            task_llm[(t["pid"], t["setting"], t["condition"], t["turn"])] = t["llm"]
    for i, p in enumerate(target):
        pid = p["person_id"]
        rep = {
            "person_id": pid, "run_name": cfg["run_name"], "config_hash": cfg["_config_hash"],
            "demographics": {"age_band": p["demographics"]["age_band"],
                             "sex": p["demographics"]["sex"]},
            "labels": {k: int(res["y"][i, j]) for j, k in enumerate(L)},
            "threshold": mcfg["threshold"], "crossing_points": pts,
            "baseline": {"accuracy": _f(res["baseline"]["acc"][i]),
                         "headroom": _f(res["baseline"]["headroom"][i]),
                         "probs": [_f(v) for v in res["baseline"]["probs"][i]],
                         **info["baseline"]},
            "twin": {"model": cfg["twin"]["model"], "revision": cfg["twin"]["revision"]},
            "settings": {}, "controls": {"wrong_person": {**donors[pid], "gain": {}},
                                         "full_record": {"gain": {}}},
        }
        for s in SETTINGS:
            tpath = rd / "transcripts" / s / f"{pid}.jsonl"
            entry = {"transcript_file": f"transcripts/{s}/{pid}.jsonl",
                     "transcript_sha256": sha256_file(tpath), "auditors": {}}
            for a, out in res["auditors"].items():
                g = out[s]["gain"][i]
                ae = {"accuracy": [_f(v) for v in out[s]["acc"][i]], "gain": [_f(v) for v in g],
                      "probs": [[_f(v) for v in row] for row in out[s]["probs"][i]],
                      "crossed": bool((g[1:] >= pts).any()),
                      "first_crossing_turn": first_crossing_turn(g, pts)}
                if a == "lexical":
                    ae["model_file"] = info["lexical"]["model_files"][s]
                    ae["model_sha256"] = info["lexical"]["model_sha256"][s]
                elif a == "llm":
                    ae.update({k: info["llm"][k] for k in ("model", "revision", "backend", "method")})
                    ae["parse_status"] = [task_llm[(pid, s, "main", t)]["parse_status"]
                                          for t in range(T + 1)]
                    ae["cache_keys"] = [task_llm[(pid, s, "main", t)]["cache_key"]
                                        for t in range(T + 1)]
                    ae["cache_file"] = "cache/llm_auditor.jsonl"
                entry["auditors"][a] = ae
                rep["controls"]["wrong_person"]["gain"].setdefault(a, {})[s] = _f(out[s]["wrong_gain"][i])
            rep["settings"][s] = entry
        for a, out in res["auditors"].items():
            rep["controls"]["full_record"]["gain"][a] = _f(out["full_record"]["gain"][i])
        write_json(pdir / f"{pid}.json", rep)

    # ---------------------------------------------------------- manifest
    manifest = {
        "run_name": cfg["run_name"], "config": cfg["_config_path"],
        "config_hash": cfg["_config_hash"], "seed": cfg["seed"],
        "synthea": {"version": cfg["synthea"]["version"], "sha256": cfg["synthea"]["sha256"],
                    "reference_date": cfg["synthea"]["reference_date"],
                    "target": cfg["synthea"]["target"], "reference": cfg["synthea"]["reference"],
                    "population_digests": digests},
        "twin": {**{k: cfg["twin"][k] for k in ("backend", "model", "revision", "license",
                                                 "max_new_tokens", "batch_size")},
                 "device": timings.get("twin_device")},
        "auditors": info, "model_files_sha256": models,
        "bootstrap": mcfg["bootstrap"], "filter": filter_stats, "timings": timings,
        "environment": environment_info(),
    }
    write_json(rd / "run_manifest.json", manifest)
    write_report_md(cfg, rd, res, controls, summaries, filter_stats, info)


def write_report_md(cfg, rd, res, controls, summaries, filter_stats, info) -> None:
    T = cfg["questions"]["per_person"]
    pts = cfg["metrics"]["crossing_points"]
    n = len(res["y"])
    head = res["baseline"]["headroom"]
    lines = [
        f"# Run report: {cfg['run_name']}",
        "",
        f"{n} target people, {T} turns, labels: {len(res['label_ids'])} "
        f"({len(cfg['labels']['conditions'])} conditions, "
        f"{len(cfg['labels']['medication_classes'])} medication classes).",
        f"Baseline (age band and sex only) mean accuracy {res['baseline']['acc'].mean():.1f}%; "
        f"mean headroom {head.mean():.1f} points; {int((head >= pts).sum())} of {n} people have "
        f"at least {pts} points of headroom, so only they can cross the {pts}-point line.",
        "",
        "## Main result",
        "",
        f"| Auditor | Setting | Gain at turn {T} (95% CI) | Crossing rate ({pts} pt) |",
        "|---|---|---|---|",
    ]
    for (a, s), sm in summaries.items():
        f_, cr = sm["final"], sm["crossing_rate"]
        lines.append(f"| {a} | {s} | {f_['mean']:+.1f} ({f_['ci_low']:+.1f} to {f_['ci_high']:+.1f}) "
                     f"| {cr['mean']:.2f} ({cr['n_crossed']}/{cr['n']}) |")
    lines += ["", "## Controls", "",
              "| Control | Auditor | Setting | Mean gain (95% CI) | Criterion | Result |",
              "|---|---|---|---|---|---|"]
    for c in controls:
        lines.append(f"| {c['control']} | {c['auditor']} | {c['setting']} | {c['mean_gain']:+.1f} "
                     f"({c['ci_low']:+.1f} to {c['ci_high']:+.1f}) | {c['criterion']} | "
                     f"{'PASS' if c['pass'] else '**FAIL**'} |")
    failed = [c for c in controls if not c["pass"]]
    lines += ["", ("**Some controls failed.** The curves for those auditors should not be read as "
                   "disclosure measurements until the failure is understood." if failed
                   else "All controls passed."), ""]
    lines += ["## Filter", ""]
    for pop, st in filter_stats.items():
        a = max(st["answers"], 1)
        lines.append(f"- {pop}: {st['answers']} answers, {st['refused'] / a:.1%} refused, "
                     f"{st['redacted_answers'] / a:.1%} had an identifier redacted, "
                     f"{st['truncated'] / a:.1%} hit the token limit.")
    if "llm" in info:
        ps = info["llm"]["parse"]
        lines += ["", "## LLM auditor", "",
                  f"Model `{info['llm']['model']}` (revision `{info['llm']['revision'][:12]}`), method "
                  f"`{info['llm']['method']}`. {ps['labels_fallback']} of {ps['labels_total']} label "
                  "values fell back to the baseline (unanswered or unparseable)."]
    lines += ["", f"## Per-label AUROC at turn {T} (filtered setting)", "",
              "Both auditors include the demographics-only prior, so compare each with the baseline.",
              "",
              "| Label | baseline | " + " | ".join(res["auditors"]) + " |",
              "|---|---|" + "---|" * len(res["auditors"])]
    for lid in res["label_ids"]:
        b = res["baseline"]["auroc"][lid]
        vals = ["n/a" if b is None else f"{b:.2f}"]
        for a in res["auditors"]:
            v = res["auditors"][a]["filtered"]["auroc_final"][lid]
            vals.append("n/a" if v is None else f"{v:.2f}")
        lines.append(f"| {lid} | " + " | ".join(vals) + " |")
    (rd / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
