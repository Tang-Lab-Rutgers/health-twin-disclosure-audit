# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke run: 5 people, 3 turns, 135M twin, lexical auditor only."""

import shutil

import pytest

from hta.cli import main

pytestmark = pytest.mark.slow


@pytest.mark.skipif(shutil.which("java") is None, reason="Java not installed")
def test_smoke_run_and_verify(tmp_path, monkeypatch):
    monkeypatch.setenv("HTA_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("HTA_WORK_DIR", str(tmp_path / "work"))
    assert main(["run", "--config", "configs/smoke.yaml"]) == 0
    rd = tmp_path / "results" / "smoke"
    for f in ["curves.png", "curves.pdf", "summary.csv", "run_manifest.json", "report.md"]:
        assert (rd / f).exists(), f
    assert len(list((rd / "per_person").glob("*.json"))) == 5
    assert main(["verify-all", "--run", str(rd)]) == 0

    from test_transcript_leaks import check_run

    n_people, n_answers = check_run(rd)
    assert (n_people, n_answers) == (25, 75)
