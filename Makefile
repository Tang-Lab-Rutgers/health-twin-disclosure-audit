# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
.PHONY: check demo reproduce smoke test test-all lint verify compare clean

UV ?= uv

# The LLM steps run on an NVIDIA GPU (the gpu extra, Linux + CUDA 12.8). Override with EXTRA=cpu.
EXTRA ?= $(shell command -v nvidia-smi >/dev/null 2>&1 && echo gpu || echo cpu)

check:           ## CPU only, a few minutes, no Java or model download: unit tests + re-verify every committed score
	$(UV) sync --locked --extra cpu
	$(UV) run --extra cpu pytest -q -m "not slow"
	$(UV) run --extra cpu python -m hta verify-all --run results/expected

demo:            ## demo.yaml on a GPU, verify, compare with results/expected
	$(UV) sync --locked --extra $(EXTRA)
	$(UV) run python -m hta run --config configs/demo.yaml
	$(UV) run python -m hta verify-all --run results/demo
	$(UV) run python -m hta compare --expected results/expected --actual results/demo

reproduce:       ## full.yaml on a GPU (Linux + CUDA 12.8)
	$(UV) sync --locked --extra gpu
	$(UV) run python -m hta run --config configs/full.yaml
	$(UV) run python -m hta verify-all --run results/full

smoke:           ## 5 people, 3 turns, tiny model: a few minutes
	$(UV) sync --locked --extra cpu
	$(UV) run python -m hta run --config configs/smoke.yaml
	$(UV) run python -m hta verify-all --run results/smoke

test:            ## unit tests (no Java, no model download)
	$(UV) run --extra $(EXTRA) pytest -q -m "not slow"

test-all:        ## unit tests + Synthea determinism + end-to-end smoke
	$(UV) run --extra $(EXTRA) pytest -q

lint:            ## ruff + REUSE licence check
	$(UV) run --extra $(EXTRA) ruff check src tests
	$(UV) run --extra $(EXTRA) reuse lint

clean:
	rm -rf work results/demo results/smoke results/full
