# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Same seeds give byte-identical FHIR output. Needs Java 17+ and the pinned Synthea jar."""

import shutil

import pytest

from hta import synthea
from hta.config import load_config

pytestmark = pytest.mark.slow


@pytest.mark.skipif(shutil.which("java") is None, reason="Java not installed")
def test_two_runs_same_seed_identical(tmp_path):
    scfg = load_config("configs/smoke.yaml")["synthea"]
    scfg = {**scfg, "target": {**scfg["target"], "n": 3}}
    a = synthea.generate(scfg, "target", tmp_path / "a")
    b = synthea.generate(scfg, "target", tmp_path / "b")
    assert len(a) == len(b) == 3
    ha, hb = synthea.bundle_hashes(tmp_path / "a"), synthea.bundle_hashes(tmp_path / "b")
    assert ha == hb, "Synthea output differs between two runs with the same seed"
    other = {**scfg, "target": {**scfg["target"], "n": 3, "seed": scfg["target"]["seed"] + 1}}
    synthea.generate(other, "target", tmp_path / "c")
    assert synthea.bundle_hashes(tmp_path / "c") != ha
