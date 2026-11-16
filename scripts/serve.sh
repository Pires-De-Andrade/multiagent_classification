#!/usr/bin/env bash
# Serve a model with tool calling on :8000.  Usage: scripts/serve.sh <model or path> [extra vllm args]
# Qwen3.5/3.6 (hybrid, multimodal): qwen3_coder parser, text-only. Qwen3: hermes parser.
set -euo pipefail
MODEL=$1; shift
case "$MODEL" in
  *Qwen3.5*|*Qwen3.6*|*qwen35*) ARGS="--tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only" ;;
  *)                            ARGS="--tool-call-parser hermes" ;;
esac
exec .venv-serve/bin/vllm serve "$MODEL" --served-model-name "$MODEL" --enable-auto-tool-choice $ARGS \
  --max-model-len 32768 --max-num-seqs 16 --kv-cache-dtype fp8 --gpu-memory-utilization 0.92 "$@"
