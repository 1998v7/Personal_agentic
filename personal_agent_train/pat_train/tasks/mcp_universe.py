"""deep research 类训练数据(MCP-Universe 的 web_search / financial_analysis / location_navigation):
策略模型按 ReAct 文本协议,经 MCP 网关调用真实工具(搜索、抓取、行情、地图……)作答,再按任务自带的评估器打分。

只用只读的任务:mcp_servers 里有 notion / github 的(会在真实服务里创建资源)一律拒绝。

ReAct 每一步都把完整历史重新渲染成一个 prompt,所以一个 episode 在 Gateway 上是多条单轮样本(每步一条,
工具结果的摘要调用也是策略模型生成的,同样是样本),它们共享这个 episode 的 reward。

reward 缺省 = 通过的评估器占比(MCP-Universe 的 AE,0~1,比只看全对的 SR 更密);``reward: sr`` 改成全对才 1。
评估阶段的基础设施错误(网关、远端评估超时等)抛异常,样本被丢弃。
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import Field

from IMAct.benchmarks.mcp_universe import tasks as T
from IMAct.benchmarks.mcp_universe.runner import McpUniverseRunner, needs_judge
from IMAct.harness.mcp import GatewayConfig
from uni_agent.tasks.base import Task, TaskConfig, TaskResult
from uni_agent.tasks.registry import register_task

from .. import shared
from ..agent import ImactAgentConfig, sample_fields
from ..policy import PolicyLLM, policy_model_config

logger = logging.getLogger(__name__)

NAME = "imact_mcp_universe"
DOMAINS = ("web_search", "financial_analysis", "location_navigation")
WRITE_SERVERS = frozenset({"notion", "github"})
DEFAULT_ENV_FILE = shared.imact_root() / ".env"


def read_only(task: T.McpTask) -> bool:
    return not task.has_side_effects and not ({s.get("name") for s in task.mcp_servers} & WRITE_SERVERS)


def train_tasks(env: dict[str, str] | None = None) -> list[T.McpTask]:
    return [t for d in DOMAINS for t in T.load_tasks(d, env) if read_only(t)]


@lru_cache(maxsize=4)
def load_env(env_file: str) -> dict[str, str]:
    """.env 与进程环境合并(进程环境优先),与 ``IMAct.benchmarks.mcp_universe.run`` 的口径一致。"""
    values = {k: v for k, v in dotenv_values(env_file).items() if v is not None} if Path(env_file).is_file() else {}
    return {**values, **os.environ}


@lru_cache(maxsize=None)
def task_by_id(domain: str, task_id: str, env_file: str) -> T.McpTask:
    env = load_env(env_file)
    found = [t for t in T.load_tasks(domain, env) if t.id == task_id]
    if not found:
        raise ValueError(f"mcp_universe {domain} 里没有 task {task_id!r}")
    return found[0]


class McpUniverseTaskConfig(TaskConfig):
    name: str = NAME
    agent: ImactAgentConfig = Field(default_factory=ImactAgentConfig)
    reward: Literal["ae", "sr"] = "ae"
    env_file: str = str(DEFAULT_ENV_FILE)
    models_file: str | None = None


@register_task(NAME)
class McpUniverseTask(Task):
    config_model = McpUniverseTaskConfig

    async def run(self) -> TaskResult:
        return await asyncio.to_thread(self.run_sync)

    def run_sync(self) -> TaskResult:
        cfg: McpUniverseTaskConfig = self.config  # type: ignore[assignment]
        domain, task_id = sample_fields(cfg, "domain", "task_id")
        if domain not in DOMAINS:
            raise ValueError(f"{NAME} 只支持 {DOMAINS},收到 {domain!r}")
        task = task_by_id(domain, task_id, cfg.env_file)
        if not read_only(task):
            raise ValueError(f"{task.id} 会写真实服务(notion / github),不用于训练")
        env = load_env(cfg.env_file)
        bridge = shared.bridge()
        policy = PolicyLLM(policy_model_config(cfg.agent.model, max_tokens=cfg.agent.max_tokens_per_turn, timeout_s=cfg.agent.timeout_s), bridge)
        try:
            runner = McpUniverseRunner(
                T.load_domain(domain), policy, bridge, GatewayConfig.from_env(env),
                judge=shared.role("judge", cfg.models_file) if needs_judge([task]) else None, env=env, side_effects=False,
            )
            run = runner.run_task(task, run_id=NAME)
        finally:
            policy.close()
        info = run.reward_info.info
        if run.infra_errors:
            raise RuntimeError(f"mcp_universe {task.id}: 评估出现基础设施错误,丢弃这条样本: {run.infra_errors[:3]}")
        success = float(info["success"])
        reward = float(info["evaluator_score"]) if cfg.reward == "ae" else success
        return TaskResult(
            # 步数用尽没作答也是模型自己的行为,按 0 分训练;finished=False 会被 mask_unfinished_episode 整条屏蔽
            reward=reward, accuracy=success, finished=True,
            extra_info={
                "domain": domain, "task_id": task.id, "evaluator_score": info["evaluator_score"],
                "iterations": run.iterations, "llm_calls": info["llm_calls"], "num_tool_calls": info["num_tool_calls"], "answered": run.answered,
            },
        )
