#!/usr/bin/env bash
# 把 IMAct 的依赖装进 uni-agent 的 venv,不改动 uni-agent 已锁定的版本(以现有包为 constraint)。
# 注意:之后若在 uni-agent 里再跑 `uv sync`,要加 --inexact,否则这些额外的包会被删掉。
set -euo pipefail
source "$(dirname "$0")/env.sh"
PY=$WORK/uni-agent/.venv/bin/python
mkdir -p "$WORK/logs"
uv pip freeze --python "$PY" | grep -v '^-e\|file://' > "$WORK/locked-constraints.txt"
# IMAct 的版本下限只是"写代码时的最新版";已被 uni-agent 锁定的包以锁定版本为准,所以只取包名。
sed -e 's/#.*//' -e 's/[<>=!~;].*//' -e 's/[[:space:]]*$//' "$PAT_IMACT_ROOT/requirements-server.txt" | grep -v '^$' > "$WORK/imact-reqs.txt"
uv pip install --python "$PY" -c "$WORK/locked-constraints.txt" -r "$WORK/imact-reqs.txt"
"$PY" - <<'EOF'
import mcp, httpx2, fastapi, yfinance, json_repair, thefuzz
print("imact deps ok", mcp.__file__)
EOF
