# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
from hta.profiles import age_band, freeze_labels, render_profile


def test_demographics_and_identifiers(profile):
    d, i = profile["demographics"], profile["identifiers"]
    assert d == {"age": 65, "age_band": "60-69", "sex": "female", "state": "MA"}
    assert i["given"] == ["Mia123", "Rose45"] and i["family"] == "Hegmann834"
    assert i["mrn"] == "bb0f1798-a377-3418-41f5-a58c8c6ed14d"
    assert i["phone"] == "555-786-4684" and i["postal_code"] == "01907"


def test_only_active_unabated_conditions(profile):
    names = [c["display"] for c in profile["all_active_conditions"]]
    assert "Viral sinusitis (disorder)" not in names            # resolved
    assert "Acute viral pharyngitis (disorder)" not in names    # abated
    clinical = [c["display"] for c in profile["conditions"]]
    assert "Full-time employment (finding)" not in clinical     # social finding
    assert "Body mass index 30+ - obesity (finding)" in clinical


def test_active_medications_and_observations(profile):
    assert [m["display"] for m in profile["medications"]] == [
        "24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet",
        "lisinopril 10 MG Oral Tablet"]
    obs = {o["code"]: o["value"] for o in profile["observations"]}
    assert obs["39156-5"] == "31.2 kg/m2" and obs["8480-6"] == "141.0 mm[Hg]"


def test_labels(profile):
    assert profile["labels"] == {
        "cond_essential_hypertension": 1, "cond_diabetes_mellitus_type_2": 1, "cond_anemia": 0,
        "med_antihypertensive": 1, "med_antidiabetic": 1, "med_nsaid": 0}  # acetaminophen stopped


def test_age_band():
    assert [age_band(a) for a in (18, 29, 30, 45, 79, 80, 95)] == [
        "18-29", "18-29", "30-39", "40-49", "70-79", "80+", "80+"]


def test_render_profile_contains_record(profile):
    text = render_profile(profile)
    assert "Mia123 Rose45 Hegmann834" in text and "Essential hypertension" in text
    assert "(disorder)" not in text


def test_freeze_labels_by_prevalence(profile):
    import copy

    a, b = copy.deepcopy(profile), copy.deepcopy(profile)
    b["all_active_conditions"] = [c for c in b["all_active_conditions"] if "hypertension" in c["display"]]
    out = freeze_labels([a, b], [{"id": "med_x", "name": "x", "keywords": ["lisinopril"]}], {})
    names = [c["name"] for c in out["conditions"]]
    assert names[0] == "Essential hypertension"                  # prevalence 2/2 first
    assert "Body mass index 30+ - obesity" not in names           # findings are not candidates
    assert out["medication_classes"][0]["reference_prevalence"] == 1.0
