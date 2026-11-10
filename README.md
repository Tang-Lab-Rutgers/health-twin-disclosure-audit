<!--
SPDX-FileCopyrightText: 2026 Ruixiang Tang
SPDX-License-Identifier: Apache-2.0
-->

# health-twin-disclosure-audit

A conversational health twin answers everyday questions about one synthetic patient, on behalf of
a third-party wellness app. Every answer passes a per-answer privacy filter that redacts
identifiers and refuses any answer naming one of the person's conditions or medications. Fixed,
open auditors read the released transcript after every turn and try to infer the person's
sensitive attributes. They have to infer 10 chronic conditions and 6 medication classes. This
repository measures **how much more the auditors know than a baseline that only knows the
person's age band and sex**, turn by turn and person by person, with and without the filter. Any
single person's score can be re-checked from its transcript with one command.

It is the open, preliminary artefact for an application to the Open Science Foundation call
*Verifiable privacy for digital health twins*. All data are synthetic.

**Results of the committed demo run:** see [`results/expected/report.md`](results/expected/report.md)
and the figure below. The numbers are reported as they came out. No question, filter, prompt or
threshold was tuned after seeing them; [`docs/decisions.md`](docs/decisions.md) records every
design choice and when it was made.

![Mean disclosure gain per turn](results/expected/curves.png)

## Quickstart

Requirements: Linux or macOS, Java 17 or newer (for Synthea), [uv](https://docs.astral.sh/uv/),
about 4 GB of free RAM and 6 GB of disk. No GPU, no API key, no account.

```bash
git clone https://github.com/Ryan-Tang-RU/health-twin-disclosure-audit
cd health-twin-disclosure-audit
make demo          # or ./reproduce.sh if you do not have make
```

Or with Docker, which needs nothing else installed:

```bash
docker build -t hta .
docker run --rm -v "$PWD/results:/app/results" hta     # runs `make demo`
```

`make demo` does the following:
1. Installs the locked environment.
2. Downloads the pinned Synthea jar and checks its SHA256.
3. Generates the two synthetic populations.
4. Downloads the pinned models from Hugging Face.
5. Runs the twin, the filter, the auditors and the three controls.
6. Re-verifies every person's score.
7. Compares everything with `results/expected/`.

Results appear in `results/demo/`.

**Runtime.** See `timings` in [`results/expected/run_manifest.json`](results/expected/run_manifest.json)
for the measured wall time of the committed run and the machine it ran on. The spec's target was
20 minutes on a laptop CPU, and this run does not meet it: about 2,300 twin answers and roughly
700 LLM-auditor prompts dominate the time. For a quick end-to-end check, use `make smoke` (5 people,
3 turns, a 135M-parameter twin, a few minutes); it is a test, not a result.

| Command | What it does |
|---|---|
| `make demo` | demo run (`configs/demo.yaml`), verify, compare with `results/expected/` |
| `make reproduce` | full run (`configs/full.yaml`: 300 + 2000 people, 20 turns); GPU recommended |
| `make smoke` | tiny end-to-end run, minutes |
| `make test` | unit tests; `make test-all` adds the Synthea determinism and smoke tests |
| `make lint` | `ruff` and `reuse lint` |

## What the figure shows

Each panel plots, for every auditor, the **mean disclosure gain** across people against the turn.
The left panel shows unfiltered answers, the right panel filtered ones.

- *Disclosure gain* for one person at turn *t* = the auditor's accuracy over the 16 binary labels
  after reading turns 1..*t*, minus the baseline's accuracy for that person, in percentage points.
  A label counts as predicted present when its probability is above 0.5.
- The shaded band is a 95% bootstrap confidence interval across people (1000 resamples, fixed seed).
- The dashed line is 20 points. The **crossing rate** in `summary.csv` is the share of people
  whose gain reaches 20 points at any turn within the budget.
- **Headroom** is 100 minus the baseline's accuracy for that person, and caps how far that
  person's gain can go. Someone with no conditions and an all-negative baseline has zero headroom.
  `summary.csv` reports the share of people with at least 20 points of headroom; only they can
  cross the line at all.

The same file holds the three controls, the filter's refusal and redaction rates, the LLM
auditor's parse-failure counts and per-label AUROC at the last turn.

### Controls

Each control's pass criterion was fixed in `configs/demo.yaml` before any result existed:

| Control | What it does | Expected | Criterion |
|---|---|---|---|
| Empty transcript | gain at turn 0, before any answer | about 0 | \|mean\| ≤ 5 points |
| Wrong person | each person scored on another person's transcript (same age band and sex) | not above 0 | mean ≤ 5 points |
| Full record | the auditor reads the raw profile text | high: the ceiling | mean ≥ 50% of mean headroom |

If a control fails, `report.md` and `summary.csv` say so in plain words, and the curves of that
auditor should not be read as a disclosure measurement.

### A per-person report

`results/<run>/per_person/<person_id>.json` holds, for that person:
- the 16 labels and the baseline's probabilities, accuracy and headroom;
- for each setting, the transcript file and its **SHA256**;
- for each auditor, the per-turn probabilities, accuracy and gain, whether and when the gain
  crossed 20 points, and the auditor's model file and SHA256 (or model name and revision);
- this person's wrong-person and full-record control values;
- the config hash.

## Check a score yourself

```bash
uv run python -m hta verify \
  --person results/expected/per_person/<id>.json \
  --transcript results/expected/transcripts/filtered/<id>.jsonl
```

`verify` checks, in order:
1. The transcript's SHA256 matches one the report was computed from. This also tells it which
   setting the transcript is.
2. The run config and the auditor model files match their recorded hashes.
3. Recomputing the baseline and lexical-auditor accuracy and gain at every turn from that
   transcript gives exactly the stored values.
4. Re-parsing the LLM auditor's stored raw answers gives exactly the stored gain.

It prints `RESULT: MATCH` or `MISMATCH`. `--rerun-llm` also runs the LLM auditor again and
reports how many of its raw answers come out identical. `python -m hta verify-all --run <dir>`
does this for every person, and CI runs it on `results/expected/` on every push.

Edit one answer in a transcript and run `verify` again: the SHA256 check fails, which is the point.

## Reproducibility

- **Pinned:** Python packages (`uv.lock`, exact versions and hashes; separate CPU and GPU
  variants), the Synthea jar (version and SHA256 in the config), and every Hugging Face model
  (commit hash). `run_manifest.json` records versions, revisions, seeds, hardware and wall time.
- **Exact everywhere:**
  - the Synthea FHIR files (same seed, byte-identical; tested in `tests/test_synthea_determinism.py`);
  - the profiles, labels and question order;
  - the filter applied to a given answer;
  - the baseline and lexical scores computed from a given transcript.
- **LLM text:** greedy decoding on CPU, with batches fixed in advance so that resuming a run
  gives the same batches. Expect identical text on the same CPU architecture and image. Across
  CPU types, floating-point differences can flip a token, after which that answer diverges.
  `hta compare` then reports how many answers match and checks the headline numbers within a
  stated tolerance (±3 gain points and ±0.10 crossing rate; see `src/hta/compare.py`). It
  does not claim an exact match it cannot guarantee.

## Data statement

Every person in this repository is synthetic, generated by [Synthea](https://github.com/synthetichealth/synthea)
v4.0.0 with fixed seeds. There is no real patient information anywhere, and nothing requires a
data use agreement. Synthea is Apache-2.0. The generated outputs are released under
**CC0-1.0**: profiles, transcripts, scores, figures and everything under `results/`. The FHIR
bundles themselves are regenerated on demand and not committed; their SHA256 hashes are.

## Configuration

Everything that can change a result lives in `configs/*.yaml`, `configs/labels.yaml`,
`data/filter/synonyms.yaml` and `data/questions/templates.yaml`. They are all covered by the
config hash written into every report.

- **Models:** set `twin.model` / `twin.revision` and `auditors.llm.model` / `.revision`. Use a
  full commit hash, and check that model's licence.
- **GPU:** `uv sync --extra gpu` (Linux, CUDA 12.8) and `device: cuda`. Setting `backend: vllm`
  uses vLLM instead of `transformers`. `configs/full.yaml` is set up for a GPU.
- **Optional commercial API auditor:** off by default and never used for `results/expected/`.
  Set `auditors.api.enabled: true`, `provider` (`anthropic` or `openai`), `model`, and put
  the key in the environment variable `HTA_API_KEY`. It uses the same prompt and parser as the
  open LLM auditor.
- `HTA_THREADS`, `HTA_CACHE`, `HTA_RESULTS_DIR` and `HTA_WORK_DIR` override the CPU threads,
  download cache, output directory and Synthea work directory.

## Limitations

- **Small scale.** 30 target people and 10 turns in the demo, so the confidence intervals are
  wide. The full configuration (300 people, 20 turns) is provided but was not run for this release.
- **Small models.** The twin is a 360M-parameter model. A stronger twin gives more specific,
  and more revealing, advice.
- **The auditors give a lower bound.** What these auditors infer is a lower bound on what an
  observer could infer: a stronger auditor can only learn more. A per-answer filter that looks
  safe against them is not shown to be safe.
- **Synthetic people.** Synthea's disease modules are simpler and more correlated with age than
  real records. The labels are the most prevalent conditions, chosen by a mechanical rule.
- **The filter leaks through its own refusals.** It refuses whole answers, so a refusal is itself
  a signal. For example, an answer that mentions "diuretic" is refused only for people who take
  a blood-pressure medication.

## Licence, citation and archiving

Code is licensed under [Apache-2.0](LICENSE). Data and generated outputs are
[CC0-1.0](LICENSES/CC0-1.0.txt). Every file carries an SPDX header or is covered by `REUSE.toml`,
and `reuse lint` passes in CI.

Please cite using [`CITATION.cff`](CITATION.cff). Each GitHub release is archived on Zenodo with a
DOI (metadata in `.zenodo.json`). The public repository is also archived by Software Heritage.
