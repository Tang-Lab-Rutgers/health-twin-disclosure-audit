# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
from hta.config import load_yaml, resolve
from hta.questions import questions_for

TEMPLATES = load_yaml(resolve("data/questions/templates.yaml"))


def test_order_fixed_by_seed_and_person():
    a = questions_for("p1", TEMPLATES, 10, 5)
    assert a == questions_for("p1", TEMPLATES, 10, 5)
    assert [q["question_id"] for q in a] != [q["question_id"] for q in questions_for("p2", TEMPLATES, 10, 5)]
    assert [q["turn"] for q in a] == list(range(1, 11))


def test_enough_templates_for_full_mode_and_no_direct_questions():
    assert len(TEMPLATES) >= 20
    banned = ["diagnos", "medication", "condition", "prescri", "what do i have"]
    for t in TEMPLATES:
        assert not any(b in t["text"].lower() for b in banned), t
