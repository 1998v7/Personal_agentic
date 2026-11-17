#!/usr/bin/env bash
# 冒烟用:单独起一个 vLLM OpenAI 服务(Gateway 经 /v1/completions + return_token_ids 取 token)。训练时不需要它,verl 自己起 rollout。
# 用法:serve_vllm.sh [端口=8100] [模型目录=$WORK/models/Qwen3.5-2B]
source "$(dirname "$0")/env.sh"
PORT=${1:-8100}
MODEL=${2:-$WORK/models/Qwen3.5-2B}
mkdir -p "$WORK/logs"
exec "$WORK/uni-agent/.venv/bin/vllm" serve "$MODEL" --port "$PORT" --host 127.0.0.1 \
  --served-model-name "$(basename "$MODEL")" --max-model-len 32768 --gpu-memory-utilization 0.80 \
  --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-seqs 32
