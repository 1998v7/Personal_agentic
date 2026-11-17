"""uni-agent 的 agent runner 入口(``custom.agent_framework.agent_runners.task.runner_fqn``)。

uni-agent 的 ``TASK_MODULES`` 是写死的;这里先把 pat_train 的 Task 登记进注册表,再原样交给官方的
``uni_agent.framework.task_runner.run_task``(Task Config 解析、Gateway 地址注入、rl_insight 记录都沿用官方实现)。
"""

from __future__ import annotations

from typing import Any

from uni_agent.framework.task_runner import run_task as _official_run_task
from uni_agent.framework.task_runner import score_from_runner_result  # noqa: F401  训练脚本的 reward 函数也指向本模块
from uni_agent.tasks.registry import TASK_MODULES

from . import agent  # noqa: F401  登记 imact agent

TASKS = {
    "imact_tau2": "pat_train.tasks.tau2",
    "imact_mcp_universe": "pat_train.tasks.mcp_universe",
    "imact_vitabench": "pat_train.tasks.vitabench",
}
for _name, _module in TASKS.items():
    TASK_MODULES.setdefault(_name, _module)


async def run_task(**kwargs: Any):
    return await _official_run_task(**kwargs)
