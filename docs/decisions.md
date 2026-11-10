<!--
SPDX-FileCopyrightText: 2026 Ruixiang Tang
SPDX-License-Identifier: CC0-1.0
-->

# Design decisions

The build spec leaves some choices open. Each one is recorded here with when it was made relative
to the results, so that none of them is a hidden degree of freedom. "Before results" means before
any twin or auditor output existed for the target population. The git history shows the order:
the run config, label lists, synonym table and question templates were committed before any
output was generated (commit `Freeze run config, label lists, ...`).

## Fixed before any output existed

| Topic | Decision | Why |
|---|---|---|
| Models | Twin: `HuggingFaceTB/SmolLM2-360M-Instruct` (demo), `SmolLM2-1.7B-Instruct` (full). The spec's example models were Qwen2.5; this release uses Apache-2.0 models from Hugging Face (SmolLM2) and AI2 (OLMo 2). Every model is pinned by commit hash and its licence was checked on the exact revision. | The spec asks for Apache-2.0 or MIT models that run on CPU and are configurable. |
| Population | Adults aged 18 to 90, Massachusetts, Synthea `-r/-e 20260101`, living patients only. | The twin serves an adult consumer wellness app. A fixed reference date makes the output independent of the day it is generated. |
| Synthea export | FHIR R4 collection bundles with only Patient, Condition, MedicationRequest, Encounter and Observation; hospital, practitioner and metadata exports switched off. | Those three exports carry wall-clock timestamps and would break byte-identical output. |
| Profile | Active, unabated conditions with SNOMED tag `(disorder)`, plus `(finding)` codes that are not social or administrative (employment, education, housing and similar). Active medication requests, the 3 most recent encounters, latest values of 9 vital signs and labs. | The spec asks for clinical content. Social-history codes are not clinical diagnoses. |
| Labels | 10 conditions: the most prevalent active `(disorder)` names in the reference population, excluding names starting with "Acute". 6 medication classes: the most prevalent of 10 candidate classes written in `data/labels/medication_class_candidates.yaml` beforehand. Frozen in `configs/labels.yaml` by `hta freeze-labels`, shared by demo and full runs. | Mechanical rule, applied once. |
| Twin | One question at a time, from the profile and the question only (no chat history). Greedy decoding, 120 new tokens. System prompt in `src/hta/twin.py`. | Each answer is then a function of the profile and the question alone, which keeps the per-turn measurement clean. |
| Filter | Redaction of the person's own identifier strings (name parts with and without Synthea's digits, birth date in five formats, address line, city, ZIP, phone, MRN, SSN, licence, passport) and generic date, phone, SSN, UUID and e-mail patterns. Then refusal if any protected term appears: every name of the person's conditions and medications (ingredients, brand names, qualifier-stripped forms) and the frozen synonyms of the labels they have. | Section 4.5 of the spec. |
| Thresholding | A label counts as predicted present when its probability is strictly above 0.5. | The spec says "thresholded at 0.5" without saying which side 0.5 falls on. |
| Wrong-person control | Each person gets another person's transcript from the same age band and sex (rotation within the stratum, sorted by id); the auditor is told the person's own demographics. Scored at the final turn. | Keeps the donor transcript plausible for the demographics the auditor sees. |
| Control criteria | Empty transcript: \|mean gain at turn 0\| ≤ 5 points. Wrong person: mean gain ≤ 5 points. Full record: mean gain ≥ 50% of the mean headroom (headroom = 100 − baseline accuracy). | Written into `configs/demo.yaml` before any result. The wrong-person criterion is one-sided because an auditor that uses person-specific evidence is expected to do *worse* than the baseline on someone else's transcript; the control exists to catch spurious *positive* gain. |
| LLM auditor prompt | Labels listed under short keys (C1..C10, M1..M6), answer pre-filled with `{`, robust JSON parsing, per-label fallback to the baseline value, counts reported. | Keeps the answer short on CPU; the spec asks for fallback and counting. |

## Changed after the smoke test, before any demo result

**Lexical auditor demographics.** The first version gave the lexical auditor one-hot age-band × sex
features next to TF-IDF. On the 5-person smoke run its gain on an *empty* transcript was +8.75
points. With no text, the auditor was just modelling demographics differently from the baseline:
regularized one-hot features against prevalence tables. A comparison whose two arms use different
demographic models measures that difference, not disclosure.
The auditor now gets the demographics only through the baseline's own prior:
`logit P = w·tfidf(text) + a·logit(prior) + b`. This is a fix to a defectively constructed
comparison, recorded here with its timing. It is not a choice made among valid options by looking
at scores. The smoke config is a test fixture, not a result.

## Selecting the LLM auditor (reference population only)

See `scripts/auditor_precheck.py` and `docs/auditor_precheck.json`. The rule was written into the
script before it was run. The rule: take the smallest candidate whose full-record gain on the first
20 reference people reaches 50% of the mean headroom, with fewer than 20% unparseable answers.
