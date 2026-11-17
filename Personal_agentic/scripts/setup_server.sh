#!/usr/bin/env bash
# 在服务器(Linux/macOS)上重建环境并做一次不发请求的自检。用法:  bash scripts/setup_server.sh [python可执行文件]
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${1:-python3}"
"$PY" - <<'PYEOF' || { echo "需要 Python >= 3.11(执行层的要求)。可指定: bash scripts/setup_server.sh python3.12"; exit 1; }
import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)
PYEOF

# VitaBench 的环境与工具代码在 IMAct/benchmarks/vitabench。这里只检查题目数据。
"$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -r requirements-server.txt

[ -f data/vitabench/tasks.json ] || \
  echo "!! 缺数据集: 将 personalization tasks.json 放到 data/vitabench/tasks.json"
.venv/bin/python -m unittest discover -s tests -t . 2>&1 | tail -3
.venv/bin/python -m IMAct.models list
.venv/bin/python -m IMAct.benchmarks.vitabench.run --config config.vitabench.yaml \
  --num-tasks 1 --max-subtasks 1 --check
