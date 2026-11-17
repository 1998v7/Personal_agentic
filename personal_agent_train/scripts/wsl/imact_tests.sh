#!/usr/bin/env bash
# 在 uni-agent 的 venv 里跑 IMAct 的测试(确认依赖版本兼容)。用法:imact_tests.sh [unittest 参数...];完整输出在 $WORK/logs/imact_tests.log
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$PAT_IMACT_ROOT"
mkdir -p "$WORK/logs"
PYTHONUTF8=1 "$WORK/uni-agent/.venv/bin/python" -m unittest "${@:-discover}" > "$WORK/logs/imact_tests.log" 2>&1
code=$?
grep -E "^(Ran|OK|FAILED|ERROR:|FAIL:)" "$WORK/logs/imact_tests.log"
exit $code
