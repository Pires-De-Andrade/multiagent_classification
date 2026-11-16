# Keep every download/cache inside the project, so `rm -rf .venv-* .cache` uninstalls everything.
# Sourced by the other scripts; for manual commands:  source scripts/env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export XDG_CACHE_HOME="$ROOT/.cache"                  # default for uv, huggingface, vllm, flashinfer, pip
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/uv-python"
export HF_HOME="$ROOT/.cache/huggingface"
export VLLM_CACHE_ROOT="$ROOT/.cache/vllm"
export TRITON_CACHE_DIR="$ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/torchinductor"
export CUDA_CACHE_PATH="$ROOT/.cache/nv"
