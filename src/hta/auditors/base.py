# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Shared auditor interface and transcript rendering."""

from __future__ import annotations

from typing import Protocol

import numpy as np


def render_transcript(rows: list[dict], upto: int) -> str:
    """The text an observer has seen after `upto` turns: questions and released answers."""
    parts = []
    for r in rows[:upto]:
        parts.append(f"App: {r['question']}\nTwin: {r['answer']}")
    return "\n\n".join(parts)


def audit_record(person: dict, text: str) -> dict:
    """What an auditor gets: the person's demographics (known to the app) and some text."""
    d = person["demographics"]
    return {"person_id": person["person_id"], "age_band": d["age_band"], "sex": d["sex"], "text": text}


class Auditor(Protocol):
    name: str

    def predict(self, records: list[dict]) -> np.ndarray:
        """Return an (n_records, n_labels) array of probabilities, label order as in labels.yaml."""
        ...
