#!/usr/bin/env bash
# Serve a model with tool calling on :8000.  Usage: scripts/serve.sh <model or path> [extra vllm args]
# Qwen3.5/3.6 (hybrid, multimodal): qwen3_coder parser, text-only. Qwen3: hermes parser.
set -euo pipefail
source "$(dirname "$0")/env.sh"
export VLLM_USE_FLASHINFER_SAMPLER=0   # FlashInfer JIT needs nvcc; no CUDA toolkit here (attention uses TRITON_ATTN below)
MODEL=$1; shift
# local dir: read model_type from config.json (a fine-tuned copy can have any name); hub id: match the name
KIND=$MODEL; [ -f "$MODEL/config.json" ] && KIND=$(grep -o '"model_type": *"[^"]*"' "$MODEL/config.json" | head -1)
case "$KIND" in
  *Qwen3.5*|*Qwen3.6*|*qwen3_5*|*qwen3_6*) ARGS="--tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only" ;;
  *)                                       ARGS="--tool-call-parser hermes" ;;
esac
exec .venv-serve/bin/vllm serve "$MODEL" --served-model-name "$MODEL" --enable-auto-tool-choice $ARGS \
  --attention-backend TRITON_ATTN --max-model-len 32768 --max-num-seqs 16 --kv-cache-dtype fp8 --gpu-memory-utilization 0.92 "$@"
