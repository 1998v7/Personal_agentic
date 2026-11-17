#!/usr/bin/env bash
# 下载策略模型到 $WORK/models/<name>。用法:download_model.sh [Qwen/Qwen3.5-2B]
set -euo pipefail
source "$(dirname "$0")/env.sh"
REPO=${1:-Qwen/Qwen3.5-2B}
DEST=$WORK/models/${REPO##*/}
mkdir -p "$DEST"
"$WORK/uni-agent/.venv/bin/python" - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2]))
PY
ls -la "$DEST"; du -sh "$DEST"
