# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""FHIR R4 bundle -> per-person profile, attribute labels and profile text."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

# Social and administrative "(finding)" codes Synthea puts on the problem list. They are not
# clinical content, so they are kept out of the twin's profile and out of the label candidates.
SOCIAL_FINDING = re.compile(
    r"employment|labor force|unemployed|education|educated|school|housing|homeless|transport|"
    r"criminal record|military|social contact|social isolation|risk activity|refugee|"
    r"violence|abuse|partner|income|insurance|stress|alcohol|tobacco|drug|misuses|"
    r"received|limited",
    re.I,
)

SEMANTIC_TAG = re.compile(r"\s*\(([^()]*)\)\s*$")

# Latest value of these LOINC codes goes into the profile.
OBSERVATION_CODES = {
    "39156-5": "BMI",
    "29463-7": "body weight",
    "8302-2": "body height",
    "8480-6": "systolic blood pressure",
    "8462-4": "diastolic blood pressure",
    "4548-4": "HbA1c",
    "2339-0": "glucose",
    "2093-3": "total cholesterol",
    "72166-2": "tobacco smoking status",
}

N_RECENT_ENCOUNTERS = 3


def semantic_tag(display: str) -> str:
    m = SEMANTIC_TAG.search(display)
    return m.group(1).strip().lower() if m else ""


def strip_tag(display: str) -> str:
    return SEMANTIC_TAG.sub("", display).strip()


def is_clinical_condition(display: str) -> bool:
    tag = semantic_tag(display)
    if tag == "disorder":
        return True
    return tag == "finding" and not SOCIAL_FINDING.search(display)


def age_on(birth: str, ref: date) -> int:
    b = date.fromisoformat(birth)
    return ref.year - b.year - ((ref.month, ref.day) < (b.month, b.day))


def age_band(age: int) -> str:
    if age < 30:
        return "18-29"
    if age >= 80:
        return "80+"
    lo = (age // 10) * 10
    return f"{lo}-{lo + 9}"


def _coding(cc: dict) -> dict:
    c = (cc.get("coding") or [{}])[0]
    return {"code": c.get("code", ""), "display": c.get("display") or cc.get("text", "")}


def _obs_value(r: dict) -> str | None:
    if "valueQuantity" in r:
        q = r["valueQuantity"]
        return f"{round(float(q['value']), 1)} {q.get('unit', '')}".strip()
    if "valueCodeableConcept" in r:
        return _coding(r["valueCodeableConcept"])["display"]
    if "valueString" in r:
        return r["valueString"]
    return None


def parse_bundle(path: str | Path, reference_date: str) -> dict:
    """Parse one Synthea FHIR R4 collection bundle into a profile dictionary."""
    ref = date(int(reference_date[:4]), int(reference_date[4:6]), int(reference_date[6:8]))
    with open(path, encoding="utf-8") as f:
        bundle = json.load(f)
    res = [e["resource"] for e in bundle["entry"]]
    patients = [r for r in res if r["resourceType"] == "Patient"]
    if len(patients) != 1:
        raise ValueError(f"{path}: expected one Patient, found {len(patients)}")
    p = patients[0]

    name = next((n for n in p.get("name", []) if n.get("use") == "official"), p["name"][0])
    given = list(name.get("given", []))
    family = name.get("family", "")
    addr = (p.get("address") or [{}])[0]
    phone = next((t["value"] for t in p.get("telecom", []) if t.get("system") == "phone"), "")
    ids = {}
    for idf in p.get("identifier", []):
        code = _coding(idf.get("type", {}))["code"] if "type" in idf else "synthea"
        ids[code or "other"] = idf["value"]

    age = age_on(p["birthDate"], ref)

    conditions: dict[str, dict] = {}
    for r in res:
        if r["resourceType"] != "Condition":
            continue
        status = _coding(r.get("clinicalStatus", {}))["code"]
        if status != "active" or "abatementDateTime" in r:
            continue
        c = _coding(r["code"])
        if c["code"] not in conditions:
            conditions[c["code"]] = {
                "code": c["code"],
                "display": c["display"],
                "onset": (r.get("onsetDateTime") or "")[:10],
            }

    meds: dict[str, dict] = {}
    for r in res:
        if r["resourceType"] != "MedicationRequest" or r.get("status") != "active":
            continue
        if "medicationCodeableConcept" not in r:
            continue
        c = _coding(r["medicationCodeableConcept"])
        meds.setdefault(c["code"], {"code": c["code"], "display": c["display"]})

    encounters = []
    for r in res:
        if r["resourceType"] == "Encounter":
            t = _coding((r.get("type") or [{}])[0])["display"]
            encounters.append({"date": r["period"]["start"][:10], "type": strip_tag(t)})
    encounters.sort(key=lambda e: (e["date"], e["type"]))

    latest: dict[str, tuple[str, str]] = {}
    for r in res:
        if r["resourceType"] != "Observation":
            continue
        codes = [_coding(r["code"])]
        # Blood pressure is a panel with components.
        for comp in r.get("component", []):
            cc = _coding(comp["code"])
            if cc["code"] in OBSERVATION_CODES and "valueQuantity" in comp:
                v = _obs_value(comp)
                when = (r.get("effectiveDateTime") or "")[:10]
                if v and (cc["code"] not in latest or when >= latest[cc["code"]][0]):
                    latest[cc["code"]] = (when, v)
        for c in codes:
            if c["code"] in OBSERVATION_CODES:
                v = _obs_value(r)
                when = (r.get("effectiveDateTime") or "")[:10]
                if v and (c["code"] not in latest or when >= latest[c["code"]][0]):
                    latest[c["code"]] = (when, v)

    return {
        "person_id": p["id"],
        "source_file": Path(path).name,
        "demographics": {"age": age, "age_band": age_band(age), "sex": p["gender"],
                         "state": addr.get("state", "")},
        "identifiers": {
            "prefix": " ".join(name.get("prefix", [])),
            "given": given,
            "family": family,
            "birth_date": p["birthDate"],
            "address_line": " ".join(addr.get("line", [])),
            "city": addr.get("city", ""),
            "postal_code": addr.get("postalCode", ""),
            "phone": phone,
            "mrn": ids.get("MR", ""),
            "ssn": ids.get("SS", ""),
            "drivers_license": ids.get("DL", ""),
            "passport": ids.get("PPN", ""),
        },
        "all_active_conditions": sorted(conditions.values(), key=lambda c: c["display"]),
        "conditions": sorted(
            (c for c in conditions.values() if is_clinical_condition(c["display"])),
            key=lambda c: c["display"],
        ),
        "medications": sorted(meds.values(), key=lambda m: m["display"]),
        "recent_encounters": encounters[-N_RECENT_ENCOUNTERS:][::-1],
        "observations": [
            {"code": k, "name": OBSERVATION_CODES[k], "date": latest[k][0], "value": latest[k][1]}
            for k in OBSERVATION_CODES
            if k in latest
        ],
    }


def full_name(profile: dict) -> str:
    i = profile["identifiers"]
    return " ".join([*i["given"], i["family"]]).strip()


def render_profile(profile: dict) -> str:
    """The text the twin holds as private context. Also the 'full record' control input."""
    i, d = profile["identifiers"], profile["demographics"]
    lines = [
        f"Name: {full_name(profile)}",
        f"Date of birth: {i['birth_date']} (age {d['age']}), sex: {d['sex']}",
        f"Address: {i['address_line']}, {i['city']}, {d['state']} {i['postal_code']}",
        f"Phone: {i['phone']}",
        f"Medical record number: {i['mrn']}",
        "Active conditions: "
        + ("; ".join(strip_tag(c["display"]) for c in profile["conditions"]) or "none recorded"),
        "Active medications: "
        + ("; ".join(m["display"] for m in profile["medications"]) or "none"),
        "Recent visits: "
        + ("; ".join(f"{e['date']} {e['type']}" for e in profile["recent_encounters"]) or "none"),
        "Latest measurements: "
        + ("; ".join(f"{o['name']} {o['value']} ({o['date']})" for o in profile["observations"])
           or "none"),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- labels


def _word_re(words: list[str]) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(re.escape(w.lower()) for w in words) + r")\b")


def med_in_class(display: str, keywords: list[str]) -> bool:
    return bool(_word_re(keywords).search(display.lower()))


def label_ids(labels_cfg: dict) -> list[str]:
    return [c["id"] for c in labels_cfg["conditions"]] + [m["id"] for m in labels_cfg["medication_classes"]]


def extract_labels(profile: dict, labels_cfg: dict) -> dict[str, int]:
    """Binary labels in the frozen order: conditions first (by SNOMED code), then med classes."""
    have = {c["code"] for c in profile["all_active_conditions"]}
    out = {}
    for c in labels_cfg["conditions"]:
        out[c["id"]] = int(any(code in have for code in c["snomed_codes"]))
    for m in labels_cfg["medication_classes"]:
        out[m["id"]] = int(any(med_in_class(x["display"], m["keywords"]) for x in profile["medications"]))
    return out


def load_population(bundles: list[Path], reference_date: str, labels_cfg: dict | None) -> list[dict]:
    people = []
    for b in bundles:
        prof = parse_bundle(b, reference_date)
        if labels_cfg is not None:
            prof["labels"] = extract_labels(prof, labels_cfg)
        people.append(prof)
    people.sort(key=lambda p: p["person_id"])
    return people


# ---------------------------------------------------------------- freezing the label lists

N_CONDITION_LABELS = 10
N_MEDICATION_LABELS = 6


def condition_candidate(display: str) -> bool:
    """Frozen rule: an active SNOMED '(disorder)' that is not an acute episode."""
    return semantic_tag(display) == "disorder" and not strip_tag(display).lower().startswith("acute ")


def freeze_labels(reference: list[dict], med_candidates: list[dict], source: dict) -> dict:
    """Pick the label lists from the reference population by prevalence (ties: by name).

    Conditions are grouped by display name, so two SNOMED codes with the same name form one label.
    """
    n = len(reference)
    by_name: dict[str, dict] = {}
    for p in reference:
        seen = set()
        for c in p["all_active_conditions"]:
            if not condition_candidate(c["display"]):
                continue
            name = strip_tag(c["display"])
            entry = by_name.setdefault(name, {"codes": set(), "count": 0})
            entry["codes"].add(c["code"])
            if name not in seen:
                entry["count"] += 1
                seen.add(name)
    ranked = sorted(by_name.items(), key=lambda kv: (-kv[1]["count"], kv[0]))
    conditions = []
    for name, e in ranked[:N_CONDITION_LABELS]:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        conditions.append({
            "id": f"cond_{slug}",
            "name": name,
            "snomed_codes": sorted(e["codes"]),
            "reference_prevalence": round(e["count"] / n, 4),
        })
    med_counts = []
    for m in med_candidates:
        k = sum(any(med_in_class(x["display"], m["keywords"]) for x in p["medications"]) for p in reference)
        med_counts.append((k, m))
    med_counts.sort(key=lambda t: (-t[0], t[1]["id"]))
    meds = [
        {**m, "reference_prevalence": round(k / n, 4)} for k, m in med_counts[:N_MEDICATION_LABELS]
    ]
    return {
        "frozen_from": {**source, "n_reference": n},
        "rule": {
            "conditions": f"top {N_CONDITION_LABELS} active SNOMED '(disorder)' names by prevalence, "
                          "excluding names starting with 'Acute'",
            "medication_classes": f"top {N_MEDICATION_LABELS} candidate classes from "
                                  "data/labels/medication_class_candidates.yaml by prevalence",
        },
        "candidate_table": {
            "conditions": [{"name": k, "count": v["count"]} for k, v in ranked[:30]],
            "medication_classes": [{"id": m["id"], "count": k} for k, m in med_counts],
        },
        "conditions": conditions,
        "medication_classes": meds,
    }
