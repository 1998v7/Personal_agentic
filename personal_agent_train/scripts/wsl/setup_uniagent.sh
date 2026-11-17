#!/usr/bin/env bash
# 在 WSL 里准备 uni-agent 训练环境:uv、uni-agent(与 Windows 侧同一 commit)、verl 子模块、vLLM + FSDP 依赖。
# 可重复执行:已完成的步骤会跳过。日志:$WORK/logs/setup_uniagent.log
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$WORK/logs"
exec > >(tee -a "$WORK/logs/setup_uniagent.log") 2>&1
echo "=== $(date) setup start"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

SRC="$PAT_ROOT/uni-agent"
COMMIT=$(git -C "$SRC" rev-parse HEAD)
if [ ! -d "$WORK/uni-agent/.git" ]; then
  git clone "$SRC" "$WORK/uni-agent"
fi
cd "$WORK/uni-agent"
git fetch -q "$SRC" "$COMMIT" || true
git checkout -q "$COMMIT"
git remote set-url origin https://github.com/verl-project/uni-agent.git
git submodule update --init --recursive
echo "uni-agent at $(git rev-parse --short HEAD), verl at $(git -C verl rev-parse --short HEAD)"

uv python install 3.12
time uv sync --locked --all-packages --extra vllm --extra fsdp
uv run --locked --all-packages --extra vllm --extra fsdp python - <<'PY'
import torch, uni_agent, verl, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
print("vllm", vllm.__version__, "uni_agent", uni_agent.__file__, "verl", verl.__file__)
x = torch.randn(1024, 1024, device="cuda"); print("matmul ok", float((x @ x).sum()) != 0)
PY
echo "=== $(date) setup done"
