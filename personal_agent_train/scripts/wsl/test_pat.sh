#!/usr/bin/env bash
# pat_train 的离线测试。用法:test_pat.sh [unittest 参数,缺省 discover]
source "$(dirname "$0")/env.sh"
export PYTHONPATH=$PAT_IMACT_ROOT:$PAT_ROOT/personal_agent_train:$WORK/uni-agent PYTHONUTF8=1
cd "$PAT_ROOT/personal_agent_train"
exec "$WORK/uni-agent/.venv/bin/python" -m unittest "${@:-discover}"
