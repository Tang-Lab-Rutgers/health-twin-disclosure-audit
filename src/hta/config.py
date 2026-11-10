# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
"""Config loading, hashing and shared small utilities."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str | Path) -> Path:
    """Resolve a config-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | Path) -> dict:
    """Load a run config, inline the frozen label file and question templates, add a hash.

    The hash covers everything that can change a result: the run config itself, the frozen
    label definitions, the synonym table and the question templates.
    """
    cfg = load_yaml(path)
    cfg["_config_path"] = str(path)
    labels_path = resolve(cfg["labels_file"])
    cfg["labels"] = load_yaml(labels_path) if labels_path.exists() else None
    cfg["synonyms"] = load_yaml(resolve(cfg["filter"]["synonyms_file"]))
    cfg["question_templates"] = load_yaml(resolve(cfg["questions"]["templates_file"]))
    hashed = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg["_config_hash"] = sha256_text(canonical_json(hashed))
    return cfg


def run_dir(cfg: dict) -> Path:
    out = os.environ.get("HTA_RESULTS_DIR")
    base = Path(out) if out else REPO_ROOT / "results"
    return base / cfg["run_name"]


def work_dir(cfg: dict) -> Path:
    """Large intermediate files (Synthea FHIR bundles). Not committed."""
    out = os.environ.get("HTA_WORK_DIR")
    base = Path(out) if out else REPO_ROOT / "work"
    return base / cfg["run_name"]


def cache_dir() -> Path:
    return Path(os.environ.get("HTA_CACHE", Path.home() / ".cache" / "hta"))


def write_json(path: str | Path, obj: Any, indent: int | None = 2) -> None:
    """Write JSON atomically (temp file then rename), so a killed run never leaves a
    half-written file that a later 'skip if exists' check would trust."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(canonical_json(r) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
