# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Optional commercial-API auditor. Off by default; never used for the demo or results/expected.

Enabled only when `auditors.api.enabled: true` in the config AND the API key is present in the
environment variable named by `auditors.api.env_var`. Uses the same prompt and parser as the
open-weight LLM auditor. Calls go out with the standard library only (no SDK dependency).
"""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np

from .baseline import BaselineAuditor
from .llm import SYSTEM, build_user_prompt, label_keys, parse_answer


def api_enabled(acfg: dict) -> bool:
    return bool(acfg.get("enabled")) and bool(os.environ.get(acfg.get("env_var", "HTA_API_KEY")))


class APIAuditor:
    name = "api"

    def __init__(self, acfg: dict, labels_cfg: dict, baseline: BaselineAuditor):
        if not api_enabled(acfg):
            raise RuntimeError("API auditor is disabled: set auditors.api.enabled and the API key")
        self.acfg, self.labels_cfg, self.baseline = acfg, labels_cfg, baseline
        self.key = os.environ[acfg.get("env_var", "HTA_API_KEY")]
        self.keys = label_keys(labels_cfg)

    def _call(self, user: str) -> str:
        a = self.acfg
        if a["provider"] == "anthropic":
            url = (a.get("base_url") or "https://api.anthropic.com") + "/v1/messages"
            body = {"model": a["model"], "max_tokens": 512, "temperature": 0, "system": SYSTEM,
                    "messages": [{"role": "user", "content": user}]}
            headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
        else:
            url = (a.get("base_url") or "https://api.openai.com/v1") + "/chat/completions"
            body = {"model": a["model"], "temperature": 0,
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": user}]}
            headers = {"authorization": f"Bearer {self.key}", "content-type": "application/json"}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.load(r)
        if a["provider"] == "anthropic":
            return "".join(b.get("text", "") for b in d["content"])
        return d["choices"][0]["message"]["content"]

    def predict(self, records: list[dict]) -> np.ndarray:
        short = [k for _, k, _ in self.keys]
        out = np.zeros((len(records), len(short)))
        for i, rec in enumerate(records):
            vals, _ = parse_answer(self._call(build_user_prompt(rec, self.labels_cfg)), short)
            prior = self.baseline.prior(rec["age_band"], rec["sex"])
            out[i] = [vals.get(k, prior[j]) for j, k in enumerate(short)]
        return out
