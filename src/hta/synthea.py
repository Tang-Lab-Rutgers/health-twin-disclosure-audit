# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Download the pinned Synthea jar, generate seeded populations, list FHIR bundles."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

from .config import cache_dir, canonical_json, sha256_file, write_json

# Only the resource types the pipeline reads. Patient and Encounter are always exported.
INCLUDED_RESOURCES = "Patient,Condition,MedicationRequest,Encounter,Observation"


def jar_path(scfg: dict) -> Path:
    return cache_dir() / "synthea" / scfg["version"] / "synthea-with-dependencies.jar"


def fetch_jar(scfg: dict) -> Path:
    """Return the pinned jar, downloading it if needed. Refuses a jar with the wrong SHA256."""
    dest = jar_path(scfg)
    if dest.exists():
        got = sha256_file(dest)
        if got == scfg["sha256"]:
            return dest
        raise RuntimeError(f"cached Synthea jar {dest} has SHA256 {got}, expected {scfg['sha256']}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".partial")
    print(f"[synthea] downloading {scfg['url']}", flush=True)
    urllib.request.urlretrieve(scfg["url"], tmp)
    got = sha256_file(tmp)
    if got != scfg["sha256"]:
        tmp.unlink()
        raise RuntimeError(f"downloaded Synthea jar has SHA256 {got}, expected {scfg['sha256']}")
    os.replace(tmp, dest)
    return dest


def java_major_version() -> int:
    java = shutil.which("java")
    if java is None:
        raise RuntimeError("Java 17 or newer is required to run Synthea, and `java` is not on PATH")
    out = subprocess.run([java, "-version"], capture_output=True, text=True)
    m = re.search(r'version "(\d+)(?:\.(\d+))?', out.stderr + out.stdout)
    if not m:
        raise RuntimeError(f"could not parse `java -version` output: {out.stderr!r}")
    major = int(m.group(1))
    return int(m.group(2)) if major == 1 else major


def synthea_args(scfg: dict, pop: dict, out_dir: Path) -> list[str]:
    age = scfg.get("age_range")
    args = [
        "-s", str(pop["seed"]),
        "-cs", str(pop["clinician_seed"]),
        "-p", str(pop["n"]),
        "-r", str(scfg["reference_date"]),
        "-e", str(scfg["reference_date"]),
    ]
    if age:
        args += ["-a", str(age)]
    args += [
        f"--exporter.baseDirectory={out_dir}/",
        "--exporter.fhir.export=true",
        "--exporter.fhir.transaction_bundle=false",
        f"--exporter.fhir.included_resources={INCLUDED_RESOURCES}",
        "--exporter.hospital.fhir.export=false",  # these file names carry a timestamp
        "--exporter.practitioner.fhir.export=false",
        "--exporter.metadata.export=false",  # carries the wall-clock run time
        "--generate.only_alive_patients=true",
        f"--generate.thread_pool_size={scfg.get('threads', 4)}",
        scfg["state"],
    ]
    return args


def generate(scfg: dict, population: str, out_dir: Path) -> list[Path]:
    """Generate one population (``target`` or ``reference``) into ``out_dir``.

    Re-uses an existing output only if it was produced by exactly the same arguments and jar.
    """
    pop = scfg[population]
    jar = fetch_jar(scfg)
    args = synthea_args(scfg, pop, out_dir)
    stamp = {"jar_sha256": scfg["sha256"], "args": args}
    stamp_file = out_dir / "GENERATED.json"
    if stamp_file.exists():
        import json

        if json.loads(stamp_file.read_text()) == stamp:
            return list_bundles(out_dir)
    major = java_major_version()
    if major < 17:
        raise RuntimeError(f"Synthea {scfg['version']} needs Java 17 or newer, found {major}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    print(f"[synthea] generating {population}: n={pop['n']} seed={pop['seed']}", flush=True)
    log = out_dir / "synthea.log"
    with open(log, "w") as f:
        proc = subprocess.run(
            ["java", "-jar", str(jar), *args], stdout=f, stderr=subprocess.STDOUT, cwd=out_dir
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Synthea failed with exit code {proc.returncode}; see {log}")
    bundles = list_bundles(out_dir)
    if len(bundles) != pop["n"]:
        raise RuntimeError(f"Synthea wrote {len(bundles)} bundles for {population}, expected {pop['n']}")
    write_json(stamp_file, stamp)
    return bundles


def list_bundles(out_dir: Path) -> list[Path]:
    return sorted((out_dir / "fhir").glob("*.json"))


def bundle_hashes(out_dir: Path) -> dict[str, str]:
    """SHA256 of every FHIR file, keyed by file name. Used by the determinism test and compare."""
    return {p.name: sha256_file(p) for p in list_bundles(out_dir)}


def population_digest(out_dir: Path) -> str:
    from .config import sha256_text

    return sha256_text(canonical_json(bundle_hashes(out_dir)))
