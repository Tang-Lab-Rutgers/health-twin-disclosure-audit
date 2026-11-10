# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Scripted question sets: a seeded, per-person order over the frozen templates."""

from __future__ import annotations

import hashlib
import random


def person_rng(seed: int, person_id: str, purpose: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{purpose}:{person_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def questions_for(person_id: str, templates: list[dict], k: int, seed: int) -> list[dict]:
    if k > len(templates):
        raise ValueError(f"asked for {k} questions but only {len(templates)} templates exist")
    order = list(range(len(templates)))
    person_rng(seed, person_id, "questions").shuffle(order)
    return [{"turn": t + 1, "question_id": templates[i]["id"], "question": templates[i]["text"]}
            for t, i in enumerate(order[:k])]
