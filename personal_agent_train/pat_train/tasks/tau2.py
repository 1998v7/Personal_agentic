"""τ²-bench 类训练数据:用户模拟器 ⇄ 策略模型(原生 function calling)⇄ 工具环境,多轮对话。

一条训练样本 = 一个 τ² task 的一个 episode。对话循环、用户模拟、评分全是 IMAct 的(与评测完全同一套代码),
只改两处训练口径:策略请求不重试(``pat_train.policy``),助手的无效消息不重新生成(``max_agent_retries=1``)。

reward = τ² 官方 reward(0/1,按 task 的 reward_basis 相乘)。评分本身出错(裁判不可用等)时抛异常,
这条样本由框架丢弃,不当成 0 分:那不是模型的错。
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from pydantic import Field

from IMAct.benchmarks.tau2.multilingual import Cell, frozen_canonical_ids, load_cell, load_pack
from IMAct.benchmarks.tau2.runner import DEFAULT_MAX_ERRORS, Tau2Runner
from IMAct.benchmarks.tau2.tasks import DOMAINS, get_tasks as registry_tasks
from uni_agent.tasks.base import Task, TaskConfig, TaskResult
from uni_agent.tasks.registry import register_task

from .. import shared
from ..agent import ImactAgentConfig, sample_fields
from ..policy import PolicyLLM, policy_model_config

logger = logging.getLogger(__name__)

NAME = "imact_tau2"
# 训练用的划分:各 domain 的任务池去掉 text-v1 的冻结评测子集(airline 的 base 50 个全在冻结子集里,所以不用于训练)。
TRAIN_SPLITS = {"retail": "base", "telecom": "full"}


@lru_cache(maxsize=None)
def train_cell(domain: str, split: str) -> Cell:
    """英文、原始数据库,任务 = ``split`` 划分减去冻结评测子集。"""
    frozen = set(frozen_canonical_ids(domain))
    tasks = [t for t in registry_tasks(domain, split) if t.id not in frozen]
    ids = [t.id for t in tasks]
    return Cell("en", domain, tasks, ids, {i: i for i in ids}, {i: {"canonical_task_id": i} for i in ids}, load_pack("en"), label=f"train-{split}")


@lru_cache(maxsize=None)
def eval_cell(domain: str) -> Cell:
    """text-v1 的英文 cell(冻结 50 题),做验证集。"""
    return load_cell("en", domain)


def resolve_cell(domain: str, split: str) -> Cell:
    if domain not in DOMAINS:
        raise ValueError(f"unknown tau2 domain {domain!r}")
    return eval_cell(domain) if split == "text-v1" else train_cell(domain, split)


class Tau2TaskConfig(TaskConfig):
    name: str = NAME
    agent: ImactAgentConfig = Field(default_factory=ImactAgentConfig)
    max_steps: int = 60
    max_errors: int = DEFAULT_MAX_ERRORS
    communicate_scorer: str = "semantic"
    models_file: str | None = None


@register_task(NAME)
class Tau2Task(Task):
    config_model = Tau2TaskConfig

    async def run(self) -> TaskResult:
        return await asyncio.to_thread(self.run_sync)

    def run_sync(self) -> TaskResult:
        cfg: Tau2TaskConfig = self.config  # type: ignore[assignment]
        domain, split, task_id = sample_fields(cfg, "domain", "split", "task_id")
        cell = resolve_cell(domain, split)
        task = cell.task_by_canonical(task_id)
        policy = PolicyLLM(policy_model_config(cfg.agent.model, max_tokens=cfg.agent.max_tokens_per_turn, timeout_s=cfg.agent.timeout_s), shared.bridge())
        try:
            runner = Tau2Runner(
                cell, agent_llm=policy, user_llm=shared.role("user", cfg.models_file), judge=shared.role("judge", cfg.models_file),
                max_steps=cfg.max_steps, max_errors=cfg.max_errors, communicate_scorer=cfg.communicate_scorer,
                conversation_options={"max_agent_retries": 1},
            )
            run = runner.run_task(task, run_id=NAME)
        finally:
            policy.close()
        info = run.reward_info.info or {}
        if info.get("evaluation_error"):
            raise RuntimeError(f"tau2 {domain}/{task_id}: 评分出错,丢弃这条样本: {info.get('note')}")
        reward = float(run.reward_info.reward)
        termination = getattr(run.termination_reason, "value", run.termination_reason)
        return TaskResult(
            reward=reward, accuracy=reward, finished=True,
            extra_info={
                "domain": domain, "split": split, "task_id": task_id, "termination_reason": str(termination),
                "num_messages": len(run.messages), "policy_calls": policy.calls,
            },
        )
