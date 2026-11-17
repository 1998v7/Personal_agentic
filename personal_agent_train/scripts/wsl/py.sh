#!/usr/bin/env bash
# 用 uni-agent 的 venv 运行 Python,PYTHONPATH 含 IMAct(Personal_agentic)与 pat_train。工作目录切到 $PY_CWD(缺省 Personal_agentic)。
# 用法:py.sh <python 参数...>   例:py.sh -m unittest discover -s tests
source "$(dirname "$0")/env.sh"
export PYTHONPATH=$PAT_IMACT_ROOT:$PAT_ROOT/personal_agent_train:$WORK/uni-agent${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUTF8=1
cd "${PY_CWD:-$PAT_IMACT_ROOT}"
exec "$WORK/uni-agent/.venv/bin/python" "$@"
