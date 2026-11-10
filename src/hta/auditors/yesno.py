# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Yes/no scoring for the open-weight LLM auditor (method `yes_no`, the default).

For every label the model is asked one question, "does this person have X? Answer Yes or No",
and the auditor reads P(Yes) / (P(Yes) + P(No)) from the next-token distribution. Nothing is
sampled and nothing is parsed. A label whose Yes+No probability mass is below `min_mass` is
counted as unanswered and falls back to the baseline value.

The released text comes first in the prompt and the label question last, so the text is encoded
once per record and its key/value cache is reused by the 16 label questions, one after another
(the cache is cropped back to the text after each). Everything runs at batch size one, so the
score of a record never depends on which other records are scored, and memory stays small.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from ..config import canonical_json, sha256_text
from .baseline import BaselineAuditor

SYSTEM = ("You audit what a wellness app could infer about a person from text it has seen. "
          "Answer each question with Yes or No.")

PREFIX = """Known about the person: age band {age_band}, sex {sex}.

Text the app has seen:
<<<
{text}
>>>

"""

ANSWER_WORDS = {"yes": ["Yes", " Yes", "yes", " yes", "YES"], "no": ["No", " No", "no", " no", "NO"]}


def label_questions(labels_cfg: dict) -> list[tuple[str, str]]:
    out = [(c["id"], f"Question: does this person have the condition: {c['name']}? Answer Yes or No.")
           for c in labels_cfg["conditions"]]
    out += [(m["id"], f"Question: does this person currently take any {m['name']}? Answer Yes or No.")
            for m in labels_cfg["medication_classes"]]
    return out


class YesNoAuditor:
    name = "llm"

    def __init__(self, acfg: dict, labels_cfg: dict, baseline: BaselineAuditor, cache_path: Path):
        self.acfg = acfg
        self.baseline = baseline
        self.questions = label_questions(labels_cfg)
        self.min_mass = float(acfg.get("min_mass", 0.5))
        self.cache_path = Path(cache_path)
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self.cache[row["key"]] = row
                except (json.JSONDecodeError, KeyError):
                    pass  # a line cut off by a killed run
        self._tok = self._model = None
        self.stats = {"records": 0, "labels_total": 0, "labels_fallback": 0}

    # ------------------------------------------------------------------ model
    def tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.acfg["model"], revision=self.acfg["revision"])
        return self._tok

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM

        torch.set_num_threads(int(os.environ.get("HTA_THREADS", self.acfg.get("torch_threads", 4))))
        from ..config import resolve_device

        device = resolve_device(self.acfg.get("device", "auto"))
        self.device_used = device
        self._model = AutoModelForCausalLM.from_pretrained(
            self.acfg["model"], revision=self.acfg["revision"],
            dtype=getattr(torch, self.acfg.get("dtype", "float32")),
            attn_implementation="sdpa").eval().to(device)
        tok = self.tokenizer()
        self._ids = {k: sorted({tok(w, add_special_tokens=False).input_ids[0] for w in ws})
                     for k, ws in ANSWER_WORDS.items()}

    # ------------------------------------------------------------------ prompts
    def prompt_parts(self, record: dict) -> tuple[str, list[str]]:
        """(shared prefix, one suffix per label). prefix + suffix is a complete chat prompt."""
        tok = self.tokenizer()
        text = record["text"].strip() or "(nothing yet)"
        user_prefix = PREFIX.format(age_band=record["age_band"], sex=record["sex"], text=text)
        marker = "\x00QUESTION\x00"
        full = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prefix + marker}],
            tokenize=False, add_generation_prompt=True)
        head, tail = full.split(marker)
        return head, [q + tail for _, q in self.questions]

    def key(self, record: dict) -> str:
        head, suffixes = self.prompt_parts(record)
        return sha256_text(canonical_json({"model": self.acfg["model"], "revision": self.acfg["revision"],
                                           "method": "yes_no", "head": head, "suffixes": suffixes}))

    # ------------------------------------------------------------------ scoring
    def _score(self, record: dict) -> dict:
        import torch

        self._load()
        tok, model = self.tokenizer(), self._model
        head, suffixes = self.prompt_parts(record)
        dev = model.device
        head_ids = tok(head, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        hl = head_ids.shape[1]
        p_yes, p_no = [], []
        with torch.inference_mode():
            cache = model(head_ids, use_cache=True).past_key_values
            for s in suffixes:  # one question at a time; the cache is trimmed back to the text
                ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
                logits = model(ids, past_key_values=cache, use_cache=True).logits[0, -1]
                cache.crop(hl)
                pr = torch.softmax(logits.float(), dim=-1).cpu().double()
                p_yes.append(float(pr[self._ids["yes"]].sum()))
                p_no.append(float(pr[self._ids["no"]].sum()))
        mass = [y + n for y, n in zip(p_yes, p_no, strict=True)]
        return {"p_yes": [y / max(m, 1e-12) for y, m in zip(p_yes, mass, strict=True)], "mass": mass}

    def scores(self, records: list[dict]) -> list[dict]:
        keys = [self.key(r) for r in records]
        todo = {}
        for k, r in zip(keys, records, strict=True):
            if k not in self.cache:
                todo.setdefault(k, r)
        print(f"[llm-auditor] {len(set(keys))} unique records, {len(set(keys)) - len(todo)} cached, "
              f"{len(todo)} to score", flush=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(self.cache_path, "a", encoding="utf-8") as f:
            for i, (k, r) in enumerate(sorted(todo.items())):
                row = {"key": k, **self._score(r)}
                self.cache[k] = row
                f.write(canonical_json(row) + "\n")
                f.flush()
                if (i + 1) % 20 == 0 or i + 1 == len(todo):
                    rate = (i + 1) / (time.time() - t0)
                    print(f"[llm-auditor] {i + 1}/{len(todo)} records, {rate:.2f}/s, "
                          f"ETA {(len(todo) - i - 1) / rate / 60:.1f} min", flush=True)
        return [self.cache[k] for k in keys]

    def to_probs(self, record: dict, row: dict) -> tuple[list[float], int]:
        prior = self.baseline.prior(record["age_band"], record["sex"])
        out, fallback = [], 0
        for j, (p, m) in enumerate(zip(row["p_yes"], row["mass"], strict=True)):
            if m < self.min_mass:
                out.append(prior[j])
                fallback += 1
            else:
                out.append(p)
        return out, fallback

    def predict(self, records: list[dict], return_details: bool = False):
        rows = self.scores(records)
        probs, details = [], []
        for r, row in zip(records, rows, strict=True):
            p, fb = self.to_probs(r, row)
            probs.append(p)
            self.stats["records"] += 1
            self.stats["labels_total"] += len(p)
            self.stats["labels_fallback"] += fb
            details.append({"key": row["key"], "status": "ok" if fb == 0 else "partial",
                            "raw": canonical_json({"p_yes": row["p_yes"], "mass": row["mass"]})})
        probs = np.array(probs)
        return (probs, details) if return_details else probs
