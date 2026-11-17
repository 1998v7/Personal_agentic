#!/usr/bin/env bash
# 兼容旧入口名称;所有模型/API/参数来自 models.yaml,不再生成或改写模型配置。
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON:-.venv/bin/python}"
ARGS=(--config "${IMACT_CONFIG:-config.vitabench.yaml}" --num-tasks 1 --max-subtasks 1
      --max-concurrency 1 --num-trials 1 --max-steps 20 --max-task-retries 0)
CHECK_ONLY=false
MEMORY_ARGS=()  # 默认遵循运行配置,不再隐式关闭首轮 Memory 预取。
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      echo 'Usage: scripts/smoke_matrixllm.sh [qwen|gpt|claude|gemini|模型别名] [--check-only] [--memory|--no-memory]'
      echo '默认使用 models.yaml 的三个 roles。位置参数只覆盖 Action;用户/裁判仍由 roles 指定。'
      echo '--check-only 不发请求。Memory 默认遵循配置;--memory 使用 MEMORY_GATEWAY_URL 或原测试网关;--no-memory 显式关闭。'
      exit 0 ;;
    --check-only) CHECK_ONLY=true ;;
    --no-memory) MEMORY_ARGS=(--no-memory) ;;
    --memory)
      MEMORY_ARGS=(--memory-url "${MEMORY_GATEWAY_URL:-https://paiplusinferencepre.alipay.com/inference/2220c69b1f3f829b_identity_memory/V0.1}") ;;
    qwen) ARGS+=(--action-model qwen3.8-27b) ;;
    gpt) ARGS+=(--action-model gpt-5.6-sol) ;;
    claude) ARGS+=(--action-model claude-opus-5) ;;
    gemini) ARGS+=(--action-model gemini-3.5-flash) ;;
    --*) echo "Unknown option: $arg" >&2; exit 2 ;;
    *) ARGS+=(--action-model "$arg") ;;
  esac
done
# 旧脚本的环境变量曾隐式改写请求,现在明确报错并要求迁移到唯一配置入口。
for name in VITA_SIM_MODEL VITA_USER_LLM VITA_EVALUATOR_LLM SMOKE_MAX_TOKENS MATRIXLLM_CLAUDE_MODEL MATRIXLLM_GPT_MODEL MATRIXLLM_GEMINI_MODEL; do
  if [[ -n "${!name:-}" ]]; then
    echo "$name 不再覆盖模型配置;请改 models.yaml 的 roles/models,并 unset $name。" >&2
    exit 2
  fi
done
if (( ${#MEMORY_ARGS[@]} > 0 )); then
  ARGS+=("${MEMORY_ARGS[@]}")
fi
if "$CHECK_ONLY"; then
  exec "$PYTHON_BIN" -m IMAct.benchmarks.vitabench.run "${ARGS[@]}" --check
fi
exec "$PYTHON_BIN" -m IMAct.benchmarks.vitabench.run "${ARGS[@]}" --out "results/model-smoke-$(date +%Y%m%d-%H%M%S)"
