#!/usr/bin/env bash
# One-time server setup: two venvs (vLLM pins torch; training needs transformers v5) (models download lazily on first serve).
# Run on the server from the project dir:  bash scripts/setup_server.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs runs

uv venv .venv-serve --python 3.12
VIRTUAL_ENV=.venv-serve uv pip install "vllm>=0.26" -e .
uv venv .venv-train --python 3.12
VIRTUAL_ENV=.venv-train uv pip install -e ".[train,dev]"
PYTHON=.venv-train/bin/python bash scripts/fetch_data.sh

# Models are not prefetched: `vllm serve` downloads each one on first use (fallbacks only if needed).
.venv-train/bin/python -m pytest -q
