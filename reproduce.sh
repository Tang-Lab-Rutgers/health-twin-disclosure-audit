#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
#
# Same steps as `make demo` / `make reproduce`, for machines without make.
#   ./reproduce.sh          # demo (CPU)
#   ./reproduce.sh full     # full run (GPU recommended)
set -euo pipefail
cd "$(dirname "$0")"
mode="${1:-demo}"
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
command -v java >/dev/null || { echo "Java 17 or newer is required for Synthea" >&2; exit 1; }
if [ "$mode" = "full" ]; then
  uv sync --locked --extra gpu
  uv run python -m hta run --config configs/full.yaml
  uv run python -m hta verify-all --run results/full
else
  uv sync --locked --extra cpu
  uv run python -m hta run --config configs/demo.yaml
  uv run python -m hta verify-all --run results/demo
  uv run python -m hta compare --expected results/expected --actual results/demo
fi
