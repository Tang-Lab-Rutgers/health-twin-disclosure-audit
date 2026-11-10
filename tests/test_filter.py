# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
from conftest import FILTER_CFG, SYNONYMS

from hta.filters import (
    apply_filter,
    condition_terms,
    leaks,
    medication_terms,
    protected_terms,
    redact,
)


def test_medication_terms():
    t = medication_terms("24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet")
    assert {"metformin hydrochloride", "metformin"} <= t
    t = medication_terms("Acetaminophen 325 MG / Oxycodone Hydrochloride 10 MG Oral Tablet [Percocet]")
    assert {"acetaminophen", "oxycodone", "percocet"} <= t
    assert "sodium" not in medication_terms("Naproxen sodium 220 MG Oral Tablet")


def test_condition_terms():
    assert {"essential hypertension", "hypertension"} <= condition_terms("Essential hypertension (disorder)")
    assert "obesity" in condition_terms("Body mass index 30+ - obesity (finding)")


def test_redaction_of_identifiers(profile):
    text = ("Hi Mia, born 1960-03-15 (March 15, 1960). Call 555-786-4684, MRN "
            "bb0f1798-a377-3418-41f5-a58c8c6ed14d, 371 Kovacek Grove Apt 31, Salem 01907. "
            "Mia123 Hegmann834, you will feel better.")
    out, n = redact(text, profile, "[REDACTED]")
    assert n >= 9
    for s in ["Mia", "1960", "555-786", "bb0f1798", "Kovacek", "Salem", "01907", "Hegmann"]:
        assert s not in out
    assert "you will feel better" in out   # lower-case words are not names


def test_refusal_on_profile_term_and_synonym(profile):
    terms = protected_terms(profile, SYNONYMS)
    f = apply_filter("Keep taking your metformin with meals.", profile, terms, FILTER_CFG)
    assert f["refused"] and f["answer"] == FILTER_CFG["refusal_text"]
    f = apply_filter("Watch your high blood pressure.", profile, terms, FILTER_CFG)
    assert f["refused"]
    f = apply_filter("Low iron is common.", profile, terms, FILTER_CFG)
    assert not f["refused"]   # synonym of a label this person does not have
    f = apply_filter("Try a 20 minute walk after dinner.", profile, terms, FILTER_CFG)
    assert not f["refused"] and f["answer"] == "Try a 20 minute walk after dinner."


def test_filtered_output_has_no_leaks(profile):
    terms = protected_terms(profile, SYNONYMS)
    answers = ["Mia123, your hypertension needs care; call 555-786-4684.",
               "Since you take lisinopril, avoid potassium salt substitutes.",
               "Hello Hegmann834! Drink water.", "Get some sleep."]
    released = [apply_filter(a, profile, terms, FILTER_CFG)["answer"] for a in answers]
    for r in released:
        assert leaks(r, profile, terms) == {"identifiers": [], "terms": []}
