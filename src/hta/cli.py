# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Command line: python -m hta <command>."""

from __future__ import annotations

import argparse
import sys

import yaml

from . import synthea
from .config import load_config, load_yaml, resolve, work_dir
from .profiles import freeze_labels, load_population


def cmd_synthea(args) -> int:
    cfg = load_config(args.config)
    for pop in ("target", "reference"):
        out = work_dir(cfg) / "synthea" / pop
        bundles = synthea.generate(cfg["synthea"], pop, out)
        print(f"[synthea] {pop}: {len(bundles)} bundles, digest {synthea.population_digest(out)}")
    return 0


def cmd_freeze_labels(args) -> int:
    cfg = load_config(args.config)
    out = work_dir(cfg) / "synthea" / "reference"
    bundles = synthea.generate(cfg["synthea"], "reference", out)
    ref = load_population(bundles, cfg["synthea"]["reference_date"], None)
    cands = load_yaml(resolve("data/labels/medication_class_candidates.yaml"))
    source = {
        "config": args.config,
        "synthea_version": cfg["synthea"]["version"],
        "population": "reference",
        "seed": cfg["synthea"]["reference"]["seed"],
        "clinician_seed": cfg["synthea"]["reference"]["clinician_seed"],
        "population_digest": synthea.population_digest(out),
    }
    labels = freeze_labels(ref, cands, source)
    # REUSE-IgnoreStart
    header = (
        "# SPDX-FileCopyrightText: 2026 Ruixiang Tang\n"
        "# SPDX-License-Identifier: CC0-1.0\n"
        "#\n# Frozen attribute labels. Generated once by `python -m hta freeze-labels`; do not edit.\n"
    )
    # REUSE-IgnoreEnd
    dest = resolve(cfg["labels_file"])
    if dest.exists() and not args.force:
        print(f"{dest} exists; refusing to overwrite a frozen label list (use --force)")
        return 1
    dest.write_text(header + yaml.safe_dump(labels, sort_keys=False, width=100))
    for c in labels["conditions"]:
        print(f"  {c['reference_prevalence']:.3f}  {c['id']}")
    for m in labels["medication_classes"]:
        print(f"  {m['reference_prevalence']:.3f}  {m['id']}")
    print(f"wrote {dest}")
    return 0


def cmd_run(args) -> int:
    from .pipeline import run

    return run(args.config, stop_after=args.stop_after, skip_llm=args.skip_llm)


def cmd_verify(args) -> int:
    from .verify import verify

    return verify(args.person, args.transcript, rerun_llm=args.rerun_llm)


def cmd_verify_all(args) -> int:
    """Run `verify` on every person and both settings of a finished run."""
    from pathlib import Path

    from .verify import verify

    rd = Path(args.run)
    people = sorted((rd / "per_person").glob("*.json"))
    bad = 0
    for pf in people:
        for s in ("unfiltered", "filtered"):
            bad += verify(str(pf), str(rd / "transcripts" / s / f"{pf.stem}.jsonl")) != 0
    print(f"verify-all: {2 * len(people) - bad}/{2 * len(people)} (person, setting) pairs match")
    return 1 if bad or not people else 0


def cmd_compare(args) -> int:
    from .compare import compare

    return compare(args.expected, args.actual)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hta", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synthea", help="download the pinned jar and generate both populations")
    p.add_argument("--config", default="configs/demo.yaml")
    p.set_defaults(func=cmd_synthea)

    p = sub.add_parser("freeze-labels", help="pick the label lists from the reference population")
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_freeze_labels)

    p = sub.add_parser("run", help="run the whole pipeline for a config")
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument("--stop-after", choices=["twin"], help="stop after generating the transcripts")
    p.add_argument("--skip-llm", action="store_true",
                   help="no LLM auditor; writes results/<run>-lite (for machines without a GPU)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("verify", help="recompute one person's score from a transcript")
    p.add_argument("--person", required=True, help="results/<run>/per_person/<id>.json")
    p.add_argument("--transcript", required=True, help="results/<run>/transcripts/<setting>/<id>.jsonl")
    p.add_argument("--rerun-llm", action="store_true", help="also run the LLM auditor again")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("verify-all", help="verify every person in a finished run")
    p.add_argument("--run", default="results/demo")
    p.set_defaults(func=cmd_verify_all)

    p = sub.add_parser("compare", help="compare a run with the committed expected outputs")
    p.add_argument("--expected", default="results/expected")
    p.add_argument("--actual", default="results/demo")
    p.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
