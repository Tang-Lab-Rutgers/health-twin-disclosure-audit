# SPDX-FileCopyrightText: 2026 Ruixiang Tang
# SPDX-License-Identifier: Apache-2.0
#
# docker build -t hta .
# docker run --rm -v "$PWD/results:/app/results" hta          # runs `make demo`
FROM ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 AS uv

FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless make git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/venv \
    HTA_CACHE=/opt/hta-cache HF_HOME=/opt/hf-cache PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --extra cpu --no-install-project
COPY . .
RUN uv sync --locked --extra cpu \
 && uv run python -c "from hta import synthea; from hta.config import load_config; synthea.fetch_jar(load_config('configs/demo.yaml')['synthea'])"
CMD ["make", "demo"]
