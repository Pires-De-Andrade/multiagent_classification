#!/usr/bin/env bash
# One-time server setup: two venvs (vLLM pins torch; training needs transformers v5) + model downloads.
# Run on the server from the project dir:  bash scripts/setup_server.sh
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs runs

uv venv .venv-serve --python 3.12
VIRTUAL_ENV=.venv-serve uv pip install "vllm>=0.26" -e .
uv venv .venv-train --python 3.12
VIRTUAL_ENV=.venv-train uv pip install -e ".[train,dev]"
PYTHON=.venv-train/bin/python bash scripts/fetch_data.sh

for m in hyperhuzaifa/Qwen3.6-35B-A3B-W4A16-AWQ stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ \
         Qwen/Qwen3.5-2B Qwen/Qwen3-1.7B; do
  .venv-serve/bin/hf download "$m" >/dev/null && echo "downloaded $m"
done
.venv-train/bin/python -m pytest -q
