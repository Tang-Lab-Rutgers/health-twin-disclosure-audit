# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Per-answer privacy filter: identifier redaction, then a sensitive-term refusal."""

from __future__ import annotations

import re

from .profiles import strip_tag

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]

# Generic identifier patterns, applied to every answer.
GENERIC_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                                       # ISO date
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),                                 # 06/04/1987
    re.compile(r"\b(?:" + "|".join(MONTHS) + r")\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b"),
    re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:" + "|".join(MONTHS) + r"),?\s+\d{4}\b"),
    re.compile(r"(?<!\w)(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b"),     # phone
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                                       # SSN
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),  # MRN
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),                                 # e-mail
]

LEADING_QUALIFIERS = {"chronic", "acute", "essential", "primary", "secondary", "recurrent",
                      "intractable", "severe", "mild", "moderate", "benign", "localized"}
SPLITTERS = [" due to ", " without ", " with ", " - "]

DOSE_FORM_WORDS = {
    "hr", "mg", "ml", "mcg", "unt", "actuat", "oral", "tablet", "tablets", "capsule", "capsules",
    "extended", "release", "delayed", "chewable", "disintegrating", "injectable", "injection",
    "solution", "suspension", "transdermal", "system", "patch", "inhaler", "inhalation",
    "inhalant", "powder", "spray", "mucosal", "nasal", "sublingual", "topical", "cream",
    "ointment", "prefilled", "syringe", "auto-injector", "pen", "metered", "dose", "dry",
    "vaginal", "ring", "intrauterine", "device", "cartridge", "film", "kit", "pack", "day",
    "abuse-deterrent", "enteric", "coated", "hour", "lozenge", "gum", "drug", "implant",
    "product", "pre-filled", "per", "jet", "nebulizer", "for", "and", "in", "the", "of",
}
SALT_WORDS = {
    "hydrochloride", "sodium", "potassium", "succinate", "tartrate", "citrate", "sulfate",
    "bitartrate", "besylate", "maleate", "fumarate", "calcium", "hcl", "acetate", "mesylate",
    "phosphate", "bromide", "hydrobromide", "monohydrate", "dihydrate", "magnesium",
    "disodium", "propionate", "xinafoate", "furoate", "dipropionate", "valerate",
}
MIN_TERM_LEN = 4


def condition_terms(display: str) -> set[str]:
    name = strip_tag(display).lower()
    pieces = {name}
    for sep in SPLITTERS:
        for p in list(pieces):
            if sep in p:
                pieces |= {s.strip() for s in p.split(sep)}
    out = set()
    for p in pieces:
        out.add(p)
        words = p.split()
        while words and words[0] in LEADING_QUALIFIERS:
            words = words[1:]
            if words:
                out.add(" ".join(words))
    return {t for t in out if len(t) >= MIN_TERM_LEN}


def medication_terms(display: str) -> set[str]:
    out = set()
    for brand in re.findall(r"\[([^\]]+)\]", display):
        out.add(brand.lower())
    text = re.sub(r"\[[^\]]*\]", " ", display)
    for comp in text.split(" / "):
        words = [w for w in comp.lower().replace(",", " ").split()
                 if not re.search(r"\d", w) and w not in DOSE_FORM_WORDS]
        if not words:
            continue
        out.add(" ".join(words))
        non_salt = [w for w in words if w not in SALT_WORDS]
        if non_salt:
            out.add(non_salt[0])
            out.add(" ".join(non_salt))
    return {t for t in out if len(t) >= MIN_TERM_LEN}


def protected_terms(profile: dict, synonyms: dict) -> list[str]:
    """Every term whose appearance in an answer triggers a refusal for this person."""
    terms: set[str] = set()
    for c in profile["conditions"]:
        terms |= condition_terms(c["display"])
    for m in profile["medications"]:
        terms |= medication_terms(m["display"])
    for label_id, value in profile.get("labels", {}).items():
        if value:
            terms |= {s.lower() for s in synonyms.get(label_id, [])}
    return sorted(terms, key=lambda t: (-len(t), t))


def identifier_strings(profile: dict) -> list[tuple[str, bool]]:
    """(string, case_sensitive). Digit-stripped name parts are matched case-sensitively, so that
    a first name such as 'Will' does not redact the verb 'will'."""
    i = profile["identifiers"]
    out: list[tuple[str, bool]] = []
    names = [*i["given"], i["family"]]
    for n in names:
        if n:
            out.append((n, False))
            bare = re.sub(r"\d+", "", n)
            if len(bare) >= 2 and bare != n:
                out.append((bare, True))
    full_bare = " ".join(re.sub(r"\d+", "", n) for n in names if n)
    out.append((full_bare, False))
    y, mo, d = i["birth_date"].split("-")
    month = MONTHS[int(mo) - 1]
    for s in [i["birth_date"], f"{mo}/{d}/{y}", f"{int(mo)}/{int(d)}/{y}",
              f"{month} {int(d)}, {y}", f"{int(d)} {month} {y}"]:
        out.append((s, False))
    for key in ["address_line", "city", "postal_code", "phone", "mrn", "ssn",
                "drivers_license", "passport"]:
        if i.get(key):
            out.append((i[key], key == "city"))
    seen, uniq = set(), []
    for s, cs in sorted(out, key=lambda t: (-len(t[0]), t[0])):
        if s and (s, cs) not in seen:
            seen.add((s, cs))
            uniq.append((s, cs))
    return uniq


def _bounded(s: str) -> str:
    return r"(?<![A-Za-z0-9])" + re.escape(s) + r"(?![A-Za-z0-9])"


def redact(text: str, profile: dict, token: str) -> tuple[str, int]:
    n = 0
    for s, case_sensitive in identifier_strings(profile):
        text, k = re.subn(_bounded(s), token, text, flags=0 if case_sensitive else re.I)
        n += k
    for pat in GENERIC_PATTERNS:
        text, k = pat.subn(token, text)
        n += k
    return text, n


def term_pattern(term: str) -> re.Pattern:
    # Optional plural suffix; whole words only.
    return re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?:s|es)?(?![a-z0-9])", re.I)


def find_terms(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms if term_pattern(t).search(text)]


def apply_filter(answer: str, profile: dict, terms: list[str], fcfg: dict) -> dict:
    redacted, n_red = redact(answer, profile, fcfg["redaction_token"])
    hits = find_terms(redacted, terms)
    if hits:
        return {"answer": fcfg["refusal_text"], "refused": True, "redactions": n_red,
                "matched_terms": hits}
    return {"answer": redacted, "refused": False, "redactions": n_red, "matched_terms": []}


def leaks(text: str, profile: dict, terms: list[str]) -> dict:
    """Independent check used by tests: which identifier strings and protected terms remain."""
    ids = [s for s, cs in identifier_strings(profile)
           if re.search(_bounded(s), text, flags=0 if cs else re.I)]
    return {"identifiers": ids, "terms": find_terms(text, terms)}
