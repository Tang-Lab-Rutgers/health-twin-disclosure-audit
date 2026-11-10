# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Filtered transcripts on disk contain no identifier strings and no protected terms.

This reads the written files of every finished run under results/ (and of the smoke run made by
test_smoke_e2e), from scratch, so it tests what was released rather than an in-memory object.
"""

import json
from pathlib import Path

import pytest

from hta.config import REPO_ROOT, read_json, read_jsonl
from hta.filters import leaks, protected_terms

# A run counts once its manifest exists: the manifest is written last, so a run still in
# progress (or killed) is never half-checked.
RUNS = [p.parent for p in sorted((REPO_ROOT / "results").glob("*/run_manifest.json"))]


def check_run(rd: Path) -> tuple[int, int]:
    cfg = read_json(rd / "config.resolved.json")
    n_answers = n_people = 0
    for pop, prof_file, tdir in [("target", "profiles/target.jsonl", "transcripts/filtered"),
                                 ("reference", "profiles/reference.jsonl", "reference/transcripts/filtered")]:
        for p in read_jsonl(rd / prof_file):
            terms = protected_terms(p, cfg["synonyms"])
            rows = read_jsonl(rd / tdir / f"{p['person_id']}.jsonl")
            n_people += 1
            for r in rows:
                n_answers += 1
                found = leaks(r["answer"], p, terms)
                assert found == {"identifiers": [], "terms": []}, (pop, p["person_id"], r["turn"], found)
                # Independent of the filter's own matcher: no raw identifier value, lower-cased.
                for key in ("mrn", "ssn", "phone", "postal_code", "birth_date", "address_line"):
                    v = p["identifiers"][key]
                    assert not v or v.lower() not in r["answer"].lower(), (p["person_id"], key)
                assert set(r) == {"turn", "question_id", "question", "answer"}, set(r)
    return n_people, n_answers


@pytest.mark.parametrize("rd", RUNS, ids=[r.name for r in RUNS])
def test_filtered_transcripts_have_no_leaks(rd):
    n_people, n_answers = check_run(rd)
    print(f"{rd.name}: checked {n_answers} filtered answers of {n_people} people")
    assert n_answers > 0


def test_some_run_was_checked():
    if not RUNS:
        pytest.skip("no finished run under results/ yet (run `make smoke` or `make demo`)")
    assert json.dumps([str(r) for r in RUNS])
