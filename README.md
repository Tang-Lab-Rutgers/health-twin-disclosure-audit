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

## Results of the committed demo run

30 synthetic adults, 10 questions each, 16 attribute labels. The twin is SmolLM2-360M, the LLM
auditor is OLMo-2-7B. The run took 11.7 minutes on one H200 GPU. Every number below comes
unchanged from [`results/expected/`](results/expected/); the full tables are in
[`report.md`](results/expected/report.md). Nothing was tuned after the results existed.

![Mean disclosure gain per turn](results/expected/curves.png)

**Two of the three controls failed, so the curves above cannot yet be read as disclosure
measurements.** That is the main finding of this preliminary run. The failures are reported
here as they came out:

- **Lexical auditor, full-record control: failed.** Given the raw profile text, it gains only
  +3.5 points. The criterion is at least +7.5, half the mean headroom. Trained on 200
  reference records, it never flags the rarer labels (hyperlipidemia, type 2 diabetes,
  opioids, ...) even when the record names them. Its small gains on transcripts are therefore a
  weak lower bound, not evidence of little leakage.
- **LLM auditor, empty-transcript control: failed narrowly.** Its mean gain is −5.4 points,
  against a criterion of |gain| ≤ 5. With no text it answers "no" for every label, while the
  baseline says "yes" for common conditions in older people. Its curve stays at that level
  through all 10 turns, so the transcripts do not move its yes/no calls. It passes the
  full-record control (+13.1 points), so it can read a record when the information is in it.
- Both auditors pass the wrong-person control.

What the numbers say, within those limits:

| Auditor | Setting | Gain at turn 10 (95% CI) | People crossing 20 points |
|---|---|---|---|
| Lexical | unfiltered | +0.8 (−2.3 to +4.6) | 2 of 30 |
| Lexical | filtered | +1.0 (−1.7 to +4.6) | 2 of 30 |
| LLM | unfiltered | −5.0 (−9.4 to −1.0) | 0 of 30 |
| LLM | filtered | −5.4 (−10.0 to −1.2) | 0 of 30 |

- The age-and-sex baseline is already 85% accurate. The mean headroom is 15 points, and only
  7 of the 30 people have 20 points of headroom at all, so the 20-point line is out of reach for
  most people by construction.
- The filter refused 3.3% of answers. 42% of answers named the person or another identifier and
  had it redacted.
- Per-label AUROC: most of the auditors' high values come from the age-and-sex prior they
  share with the baseline. With the baseline's own AUROC alongside, the text adds little
  ([`docs/demo_auroc.md`](docs/demo_auroc.md)).
- **Reproducibility.** The twin's answers were identical between an Apple M3 CPU and an H200 GPU
  (300 of 300 target answers), and `hta verify-all` reproduces all 60 stored scores exactly on
  another machine ([`docs/cross_platform.md`](docs/cross_platform.md)).

The next step is to fix the two auditor weaknesses. That would be a pre-registered change,
checked on the reference population only and labelled as made after these results. It is not
part of this release.

## Quickstart

Requirements:
- Linux with an NVIDIA GPU with at least 40 GB of memory and a CUDA 12.8 driver. The LLM auditor
  is a 7B model in float32.
- Java 17 or newer, for Synthea.
- [uv](https://docs.astral.sh/uv/).
- About 45 GB of disk for the models.

No API key or account is needed.

```bash
git clone https://github.com/Tang-Lab-Rutgers/health-twin-disclosure-audit
cd health-twin-disclosure-audit
make demo          # or ./reproduce.sh if you do not have make
```

Or with Docker (needs the NVIDIA Container Toolkit):

```bash
docker build -t hta .
docker run --rm --gpus all -v "$PWD/results:/app/results" hta     # runs `make demo`
```

`make demo` does the following:
1. Installs the locked environment.
2. Downloads the pinned Synthea jar and checks its SHA256.
3. Generates the two synthetic populations.
4. Downloads the pinned models from Hugging Face.
5. Runs the twin, the filter, the auditors and the three controls.
6. Re-verifies every person's score.
7. Compares everything with `results/expected/`.

Results appear in `results/demo/`. The measured wall time and the GPU of the committed run are in
`timings` and `environment` in [`results/expected/run_manifest.json`](results/expected/run_manifest.json).

Without a GPU, `make test` and `make smoke` still run on a CPU in minutes (`EXTRA=cpu`), and
`hta verify` re-checks any committed score on any machine in seconds. The full demo also runs
on a CPU (`EXTRA=cpu make demo`), but the 7B auditor then needs about 32 GB of RAM and hours.

| Command | What it does |
|---|---|
| `make demo` | demo run (`configs/demo.yaml`, GPU), verify, compare with `results/expected/` |
| `make reproduce` | full run (`configs/full.yaml`: 300 + 2000 people, 20 turns, GPU) |
| `make smoke` | tiny end-to-end run on a CPU, a few minutes; a test, not a result |
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
  - the synthetic population: same seed, same parsed profiles and labels on every machine.
    The raw Synthea files are byte-identical on the same CPU type
    (`tests/test_synthea_determinism.py`); across CPU types a few files differ in the last digit
    of a value the pipeline does not read;
  - the question order;
  - the filter applied to a given answer;
  - the baseline and lexical scores computed from a given transcript.
- **LLM output:** greedy decoding; deterministic CUDA kernels; TF32 off; batches fixed in
  advance, so that resuming a run gives the same batches. Expect identical text when you rerun on
  the same GPU model and image. On a different GPU or CPU, floating-point differences can flip a
  token, after which that answer diverges. `hta compare` then reports how many answers match and
  checks the headline numbers within a stated tolerance (±3 gain points and ±0.10 crossing rate;
  see `src/hta/compare.py`). It does not claim an exact match it cannot guarantee.
  `docs/cross_platform.md` reports the measured drift between two machines.

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
  and more revealing, advice. The LLM auditor is a 7B model, picked as the smallest of four
  candidates that can read a raw record (`docs/decisions.md`). With yes/no scoring, the smaller
  ones could not.
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
