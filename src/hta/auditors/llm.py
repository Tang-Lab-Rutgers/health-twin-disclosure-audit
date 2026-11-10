# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Open-weight LLM auditor: prompt with the released text and the label list, parse JSON.

Labels are listed under short keys (C1..C10, M1..M6) to keep the JSON answer short. A label the
model does not return a usable number for falls back to the demographics-only baseline value;
both whole-answer and per-label parse failures are counted and reported.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..generation import CachedGenerator
from .baseline import BaselineAuditor

SYSTEM = ("You audit what a wellness app could infer about a person from text it has seen. "
          "Answer with a JSON object only.")

USER = """Known about the person: age band {age_band}, sex {sex}.

Text the app has seen:
<<<
{text}
>>>

For each item, give the probability (a number from 0 to 1) that this person has it.
{items}

Reply with one JSON object that has exactly these keys: {keys}."""


def label_keys(labels_cfg: dict) -> list[tuple[str, str, str]]:
    """(label_id, short_key, description) in the frozen label order."""
    out = []
    for i, c in enumerate(labels_cfg["conditions"], 1):
        out.append((c["id"], f"C{i}", f"has the condition: {c['name']}"))
    for i, m in enumerate(labels_cfg["medication_classes"], 1):
        out.append((m["id"], f"M{i}", f"currently takes a {m['name']}"))
    return out


def build_user_prompt(record: dict, labels_cfg: dict) -> str:
    keys = label_keys(labels_cfg)
    items = "\n".join(f"{k}: {d}" for _, k, d in keys)
    text = record["text"].strip() or "(nothing yet)"
    return USER.format(age_band=record["age_band"], sex=record["sex"], text=text, items=items,
                       keys=", ".join(k for _, k, _ in keys))


def parse_answer(raw: str, keys: list[str]) -> tuple[dict[str, float], str]:
    """Return ({key: prob}, status). status is 'ok', 'partial' or 'failed'."""
    found: dict[str, float] = {}
    m = re.search(r"\{.*?\}", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in keys and isinstance(v, int | float) and not isinstance(v, bool):
                        found[k] = float(v)
        except json.JSONDecodeError:
            pass
    if len(found) < len(keys):  # lenient fallback: "C3": 0.4 pairs anywhere in the text
        for k, v in re.findall(r'"?\b([CM]\d{1,2})\b"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', raw):
            if k in keys and k not in found:
                found[k] = float(v)
    clean = {}
    for k, v in found.items():
        if 0.0 <= v <= 1.0:
            clean[k] = v
        elif 1.0 < v <= 100.0:  # answered in percent
            clean[k] = v / 100.0
    status = "ok" if len(clean) == len(keys) else ("failed" if not clean else "partial")
    return clean, status


class LLMAuditor:
    name = "llm"

    def __init__(self, acfg: dict, labels_cfg: dict, baseline: BaselineAuditor, cache_path: Path):
        self.acfg = acfg
        self.labels_cfg = labels_cfg
        self.baseline = baseline
        self.keys = label_keys(labels_cfg)
        self.gen = CachedGenerator(acfg, cache_path, "llm-auditor")
        self.stats = {"calls": 0, "ok": 0, "partial": 0, "failed": 0, "labels_fallback": 0,
                      "labels_total": 0}

    def prompt(self, record: dict) -> str:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user_prompt(record, self.labels_cfg)}]
        return self.gen.chat_prompt(msgs, prefill="{")

    def predict(self, records: list[dict], return_details: bool = False):
        prompts = [self.prompt(r) for r in records]
        outs = self.gen.generate(prompts)
        short = [k for _, k, _ in self.keys]
        probs = np.zeros((len(records), len(self.keys)))
        details = []
        for i, (rec, p) in enumerate(zip(records, prompts, strict=True)):
            raw = "{" + outs[p]["text"]
            vals, status = parse_answer(raw, short)
            prior = self.baseline.prior(rec["age_band"], rec["sex"])
            for j, k in enumerate(short):
                probs[i, j] = vals.get(k, prior[j])
            self.stats["calls"] += 1
            self.stats[status] += 1
            self.stats["labels_total"] += len(short)
            self.stats["labels_fallback"] += len(short) - len(vals)
            details.append({"key": self.gen.key(p), "status": status, "raw": raw})
        return (probs, details) if return_details else probs
