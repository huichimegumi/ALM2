#!/usr/bin/env bash
set -euo pipefail

PROJECT="$HOME/projects/aeollm2-e1"

cd "$PROJECT"
source "$PROJECT/.venv/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/models/hf-cache}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-e1-local-key}"

# 调用脚本前通过 CUDA_VISIBLE_DEVICES 指定实际空闲 GPU。
# 例如：CUDA_VISIBLE_DEVICES=3 bash scripts/start_vllm.sh
: "${CUDA_VISIBLE_DEVICES:?Please set CUDA_VISIBLE_DEVICES first}"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "VLLM_PORT=$VLLM_PORT"
echo "HF_HOME=$HF_HOME"

exec vllm serve Qwen/Qwen3-8B \
  --served-model-name qwen3-8b-e1 \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --api-key "$VLLM_API_KEY" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.88 \
  --max-num-seqs 4 \
  --enable-prefix-caching
