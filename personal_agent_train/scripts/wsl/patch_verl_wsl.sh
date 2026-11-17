#!/usr/bin/env bash
# 给本机 verl 克隆打 WSL 补丁(幂等):WSL2 不实现 CUDA IPC(cudaIpcGetMemHandle),而 verl 的 CUDA 平台把
# is_ipc_supported() 写死为 True,于是 colocate 的权重同步在 WSL 上必然报 "CUDA error: invalid resource handle"。
# verl 自己为 NPU 准备了共享内存(shm)回退路径,这里只是让它在 WSL 下也走那条路。
#
#   bash scripts/wsl/patch_verl_wsl.sh          # 打补丁
#   bash scripts/wsl/patch_verl_wsl.sh --check  # 只检查是否已打
set -euo pipefail
source "$(dirname "$0")/env.sh"
TARGET=$WORK/uni-agent/verl/verl/plugin/platform/platform_cuda.py

"$WORK/uni-agent/.venv/bin/python" - "$TARGET" "${1:-}" <<'PY'
import sys
from pathlib import Path

path, mode = Path(sys.argv[1]), sys.argv[2]
text = path.read_text(encoding="utf-8")
MARK = "# patched by personal_agent_train: WSL2 has no CUDA IPC"
if MARK in text:
    print(f"already patched: {path}")
    raise SystemExit(0)
if mode == "--check":
    print(f"NOT patched: {path}")
    raise SystemExit(1)

old = """    def is_ipc_supported(self) -> bool:
        return True
"""
new = f"""    def is_ipc_supported(self) -> bool:
        {MARK}
        # (cudaIpcGetMemHandle), so colocated weight sync must take verl's
        # shared-memory fallback instead of the CUDA IPC path.
        import platform as _platform

        if "microsoft" in _platform.uname().release.lower():
            return False
        return True
"""
if text.count(old) != 1:
    raise SystemExit(f"unexpected is_ipc_supported() in {path}; patch manually")
path.write_text(text.replace(old, new), encoding="utf-8")
print(f"patched: {path}")
PY
