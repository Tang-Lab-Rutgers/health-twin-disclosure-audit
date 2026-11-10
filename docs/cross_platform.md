<!--
SPDX-FileCopyrightText: 2026 Ruixiang Tang
SPDX-License-Identifier: CC0-1.0
-->

# Cross-platform check (2026-11-09)

The demo's twin stage (same commit, same `uv.lock`, same config) was run on two machines:

| | Machine A | Machine B (committed `results/expected`) |
|---|---|---|
| Hardware | Apple M3 Pro, CPU only, 4 threads | Linux x86-64, NVIDIA H200 |
| torch | 2.8.0 (macOS arm64 wheel) | 2.8.0+cu128 |
| Java (Synthea) | Temurin 21.0.13, macOS aarch64 | Temurin 21.0.13, Linux x64 |

| Artefact | Identical |
|---|---|
| Parsed profiles and labels (target and reference) | yes, byte for byte |
| Synthea FHIR files, target | 25 of 30 files byte-identical |
| Synthea FHIR files, reference | 177 of 200 files byte-identical |
| Twin answers, target (unfiltered and filtered) | 300 of 300 |
| Twin answers, reference (unfiltered) | 1999 of 2000 |
| Filter statistics (refusals, redactions, truncations) | identical |

The Synthea files that differ do so in the last digit of a floating-point extension value, for
example `0.006835217361926344` against `0.006835217361926346`. That value is not one the pipeline
reads, which is why the parsed profiles are identical. On the same machine, two Synthea runs with
the same seeds are byte-identical (`tests/test_synthea_determinism.py`).

`hta verify-all --run results/expected` on Machine A reproduces every stored score of the
Machine B run exactly (60 of 60 person-setting pairs), because it recomputes the scores from the
stored transcripts, model files and LLM-auditor outputs.

The LLM auditor stage was run only on Machine B. A 7B model on Machine A's CPU was ruled out
(see `docs/decisions.md`).
