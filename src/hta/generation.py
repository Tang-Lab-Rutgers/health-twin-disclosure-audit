# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Greedy text generation with a disk cache, shared by the twin and the LLM auditor.

Determinism notes. Batched generation pads prompts, and padding changes floating-point sums, so
the batch a prompt lands in can change its output. The batches are therefore fixed up front from
the full prompt list (sorted by token length, then by key), and the cache stores whole batches.
A run that is interrupted and resumed uses exactly the same batches as an uninterrupted one.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import canonical_json, sha256_text


class CachedGenerator:
    def __init__(self, gcfg: dict, cache_path: Path, label: str):
        self.gcfg = gcfg
        self.cache_path = Path(cache_path)
        self.label = label
        self._backend = None
        self._tok = None
        self.cache: dict[str, dict] = {}
        self.bad_cache_lines = 0
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        self.cache[row["key"]] = row
                    except (json.JSONDecodeError, KeyError):
                        self.bad_cache_lines += 1  # a line cut off by a killed run
        if self.bad_cache_lines:
            print(f"[{label}] ignored {self.bad_cache_lines} truncated cache line(s)")

    # -------------------------------------------------------------- backend
    def tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.gcfg["model"], revision=self.gcfg["revision"])
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self._tok = tok
        return self._tok

    def _load(self):
        if self._backend is not None:
            return
        if self.gcfg.get("backend", "transformers") == "vllm":
            from vllm import LLM  # optional GPU backend

            self._backend = ("vllm", LLM(model=self.gcfg["model"], revision=self.gcfg["revision"],
                                         seed=0, dtype=self.gcfg.get("dtype", "auto")))
            return
        import torch
        from transformers import AutoModelForCausalLM

        threads = int(os.environ.get("HTA_THREADS", self.gcfg.get("torch_threads", 4)))
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        from .config import resolve_device

        device = resolve_device(self.gcfg.get("device", "auto"))
        self.device_used = device
        model = AutoModelForCausalLM.from_pretrained(
            self.gcfg["model"], revision=self.gcfg["revision"],
            dtype=getattr(torch, self.gcfg.get("dtype", "float32")), attn_implementation="sdpa",
        ).eval().to(device)
        self._backend = ("transformers", model)

    def key(self, prompt: str) -> str:
        g = self.gcfg
        return sha256_text(canonical_json({
            "model": g["model"], "revision": g["revision"], "backend": g.get("backend", "transformers"),
            "max_new_tokens": g["max_new_tokens"], "batch_size": g["batch_size"], "prompt": prompt,
        }))

    def chat_prompt(self, messages: list[dict], prefill: str = "") -> str:
        tok = self.tokenizer()
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill

    # -------------------------------------------------------------- generation
    def batches(self, prompts: list[str]) -> list[list[str]]:
        tok = self.tokenizer()
        uniq = sorted(set(prompts))
        lengths = {p: len(tok(p, add_special_tokens=False).input_ids) for p in uniq}
        ordered = sorted(uniq, key=lambda p: (lengths[p], self.key(p)))
        bs = int(self.gcfg["batch_size"])
        return [ordered[i:i + bs] for i in range(0, len(ordered), bs)]

    def generate(self, prompts: list[str]) -> dict[str, dict]:
        """Return {prompt: {"text", "n_new_tokens", "finished"}} for every prompt."""
        batches = self.batches(prompts)
        todo = [b for b in batches if not all(self.key(p) in self.cache for p in b)]
        n_cached = sum(len(b) for b in batches) - sum(len(b) for b in todo)
        print(f"[{self.label}] {len(set(prompts))} unique prompts, {n_cached} cached, "
              f"{len(todo)} batches to run", flush=True)
        if todo:
            self._load()
        t_start = time.time()
        done = 0
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            for bi, batch in enumerate(todo):
                t0 = time.time()
                outs = self._run_batch(batch)
                for p, o in zip(batch, outs, strict=True):
                    row = {"key": self.key(p), **o}
                    self.cache[row["key"]] = row
                    f.write(canonical_json(row) + "\n")
                f.flush()
                done += len(batch)
                el = time.time() - t_start
                rate = done / el
                left = sum(len(b) for b in todo[bi + 1:])
                print(f"[{self.label}] batch {bi + 1}/{len(todo)}: {len(batch)} in "
                      f"{time.time() - t0:.1f}s, {rate:.2f} gen/s, ETA {left / rate / 60:.1f} min",
                      flush=True)
        return {p: self.cache[self.key(p)] for p in set(prompts)}

    def _run_batch(self, batch: list[str]) -> list[dict]:
        kind, backend = self._backend
        n_max = int(self.gcfg["max_new_tokens"])
        if kind == "vllm":
            from vllm import SamplingParams

            sp = SamplingParams(temperature=0.0, max_tokens=n_max, seed=0)
            res = backend.generate(batch, sp, use_tqdm=False)
            return [{"text": r.outputs[0].text.strip(), "n_new_tokens": len(r.outputs[0].token_ids),
                     "finished": r.outputs[0].finish_reason == "stop"} for r in res]
        import torch

        tok = self.tokenizer()
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(backend.device) for k, v in enc.items()}
        # A static KV cache is about twice as fast on CPU. On CUDA it makes transformers compile
        # the model, and recompile for every new prompt length, so the default cache is used there.
        cache = "static" if backend.device.type == "cpu" else None
        with torch.inference_mode():
            out = backend.generate(
                **enc, max_new_tokens=n_max, do_sample=False, temperature=None, top_p=None,
                top_k=None, pad_token_id=tok.pad_token_id, cache_implementation=cache,
            )
        new = out[:, enc["input_ids"].shape[1]:].tolist()
        eos = set(backend.generation_config.eos_token_id
                  if isinstance(backend.generation_config.eos_token_id, list)
                  else [backend.generation_config.eos_token_id])
        eos.add(tok.eos_token_id)
        results = []
        for ids in new:
            cut = next((i for i, t in enumerate(ids) if t in eos), None)
            body = ids if cut is None else ids[:cut]
            results.append({"text": tok.decode(body, skip_special_tokens=True).strip(),
                            "n_new_tokens": len(body), "finished": cut is not None})
        return results
