# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
import json

import pytest


def _cc(system, code, display):
    return {"coding": [{"system": system, "code": code, "display": display}], "text": display}


SNOMED, RXNORM, LOINC = "http://snomed.info/sct", "http://www.nlm.nih.gov/research/umls/rxnorm", "http://loinc.org"


def make_bundle():
    def cond(code, display, status="active", abated=False):
        r = {"resourceType": "Condition", "code": _cc(SNOMED, code, display),
             "clinicalStatus": {"coding": [{"code": status}]}, "onsetDateTime": "2015-01-01T00:00:00Z"}
        if abated:
            r["abatementDateTime"] = "2016-01-01T00:00:00Z"
        return r

    res = [
        {"resourceType": "Patient", "id": "p-1", "gender": "female", "birthDate": "1960-03-15",
         "name": [{"use": "official", "family": "Hegmann834", "given": ["Mia123", "Rose45"], "prefix": ["Ms."]}],
         "address": [{"line": ["371 Kovacek Grove Apt 31"], "city": "Salem", "state": "MA", "postalCode": "01907"}],
         "telecom": [{"system": "phone", "value": "555-786-4684"}],
         "identifier": [
             {"system": "https://github.com/synthetichealth/synthea", "value": "p-1"},
             {"type": _cc("v2", "MR", "Medical Record Number"), "value": "bb0f1798-a377-3418-41f5-a58c8c6ed14d"},
             {"type": _cc("v2", "SS", "SSN"), "value": "999-60-5596"}]},
        cond("59621000", "Essential hypertension (disorder)"),
        cond("44054006", "Diabetes mellitus type 2 (disorder)"),
        cond("444814009", "Viral sinusitis (disorder)", status="resolved"),
        cond("195662009", "Acute viral pharyngitis (disorder)", abated=True),
        cond("160903007", "Full-time employment (finding)"),
        cond("162864005", "Body mass index 30+ - obesity (finding)"),
        {"resourceType": "MedicationRequest", "status": "active",
         "medicationCodeableConcept": _cc(RXNORM, "314076", "lisinopril 10 MG Oral Tablet")},
        {"resourceType": "MedicationRequest", "status": "active",
         "medicationCodeableConcept": _cc(
             RXNORM, "860975", "24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet")},
        {"resourceType": "MedicationRequest", "status": "stopped",
         "medicationCodeableConcept": _cc(RXNORM, "313782", "Acetaminophen 325 MG Oral Tablet")},
        {"resourceType": "Encounter", "period": {"start": "2025-06-01T09:00:00Z"},
         "type": [_cc(SNOMED, "162673000", "General examination of patient (procedure)")]},
        {"resourceType": "Observation", "code": _cc(LOINC, "39156-5", "BMI"),
         "effectiveDateTime": "2025-06-01T09:00:00Z", "valueQuantity": {"value": 31.24, "unit": "kg/m2"}},
        {"resourceType": "Observation", "code": _cc(LOINC, "85354-9", "Blood pressure panel"),
         "effectiveDateTime": "2025-06-01T09:00:00Z",
         "component": [{"code": _cc(LOINC, "8480-6", "Systolic"), "valueQuantity": {"value": 141, "unit": "mm[Hg]"}},
                       {"code": _cc(LOINC, "8462-4", "Diastolic"), "valueQuantity": {"value": 92, "unit": "mm[Hg]"}}]},
    ]
    return {"resourceType": "Bundle", "type": "collection", "entry": [{"resource": r} for r in res]}


LABELS = {
    "conditions": [
        {"id": "cond_essential_hypertension", "name": "Essential hypertension", "snomed_codes": ["59621000"]},
        {"id": "cond_diabetes_mellitus_type_2", "name": "Diabetes mellitus type 2", "snomed_codes": ["44054006"]},
        {"id": "cond_anemia", "name": "Anemia", "snomed_codes": ["271737000"]},
    ],
    "medication_classes": [
        {"id": "med_antihypertensive", "name": "blood pressure medication", "keywords": ["lisinopril"]},
        {"id": "med_antidiabetic", "name": "diabetes medication", "keywords": ["metformin"]},
        {"id": "med_nsaid", "name": "NSAID", "keywords": ["ibuprofen", "acetaminophen"]},
    ],
}

SYNONYMS = {"cond_essential_hypertension": ["high blood pressure"], "cond_anemia": ["low iron"],
            "med_antidiabetic": ["diabetes medication"]}

FILTER_CFG = {"redaction_token": "[REDACTED]", "refusal_text": "I'm sorry, I can't help with that request."}


@pytest.fixture
def bundle_path(tmp_path):
    p = tmp_path / "Mia123_Hegmann834_p-1.json"
    p.write_text(json.dumps(make_bundle()))
    return p


@pytest.fixture
def profile(bundle_path):
    from hta.profiles import extract_labels, parse_bundle

    prof = parse_bundle(bundle_path, "20260101")
    prof["labels"] = extract_labels(prof, LABELS)
    return prof
