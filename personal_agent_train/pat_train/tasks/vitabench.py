"""VitaBench 类训练数据:策略模型经 IMAct 执行层(JSON 决策 + 校验)与用户模拟器、工具环境对话,带外部 Memory。

一条样本 = 一个 personalization task 里的一个 subtask(各 subtask 的环境、评分标准独立)。每个 episode 单独建一个执行层
运行时(自己的 SQLite),策略模型 = 本次会话的 Gateway 地址;决策不重试(``retry.model_call.max_retries=0``,执行层
每轮只给一次机会),所以 Gateway 上的每条样本都是模型的原始决策。

reward:
* 执行层拒绝过模型的任何一次决策(attempts 里有非基础设施类的错误码,如 invalid_decision / ungrounded_id /
  unsupported_tool …)→ 整个 episode 记 0,不调用裁判;
* 否则 = 这个 subtask 的 rubric 得分(IMAct 的 VitaBench 评分,与评测同一套代码)。
* 基础设施错误(模型请求传输 / 鉴权 / HTTP 错误、决策超时或中断、Memory 预取失败、裁判出错)→ 抛异常,样本被丢弃。

必须带 Memory:外层配置的 ``memory.backend`` 为 none 时直接报错。Memory 网关只在内网可达。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import tempfile
from functools import lru_cache
from pathlib import Path

from pydantic import Field

from IMAct.benchmarks.vitabench.personalization import PersonalizationRunner
from IMAct.benchmarks.vitabench.tasks import get_tasks
from IMAct.exec_layer import build_runtime, load_outer_config
from uni_agent.tasks.base import Task, TaskConfig, TaskResult
from uni_agent.tasks.registry import register_task

from .. import shared
from ..agent import ImactAgentConfig, sample_fields
from ..policy import policy_model_config

logger = logging.getLogger(__name__)

NAME = "imact_vitabench"
DEFAULT_OUTER_CONFIG = shared.imact_root() / "config.vitabench.yaml"
# 执行层 attempts.error 里属于基础设施(不是模型的错)的错误码:样本丢弃,不计 0 分
INFRA_CODES = frozenset({
    "model_transport_error", "model_auth_error", "model_http_error", "model_not_configured",
    "decision_timeout", "decision_interrupted",
})


def attempt_errors(db_path: str | Path) -> list[str]:
    """这次 episode 里执行层记录的每次决策失败的错误码(按发生顺序)。"""
    with sqlite3.connect(str(db_path)) as db:
        rows = db.execute("SELECT error FROM attempts WHERE error IS NOT NULL ORDER BY id").fetchall()
    codes = []
    for (raw,) in rows:
        try:
            codes.append(str(json.loads(raw).get("code") or "unknown"))
        except (ValueError, AttributeError):
            codes.append("unknown")
    return codes


@lru_cache(maxsize=4)
def _tasks(language: str) -> dict:
    return {t.id: t for t in get_tasks(language=language)}


class VitaBenchTaskConfig(TaskConfig):
    name: str = NAME
    agent: ImactAgentConfig = Field(default_factory=ImactAgentConfig)
    outer_config: str = Field(default=str(DEFAULT_OUTER_CONFIG), description="执行层与 Memory 的运行配置(config.vitabench.yaml)")
    max_steps: int = 50
    language: str = "chinese"
    models_file: str | None = None


@register_task(NAME)
class VitaBenchTask(Task):
    config_model = VitaBenchTaskConfig

    async def run(self) -> TaskResult:
        return await asyncio.to_thread(self.run_sync)

    def run_sync(self) -> TaskResult:
        cfg: VitaBenchTaskConfig = self.config  # type: ignore[assignment]
        task_id, index = sample_fields(cfg, "task_id", "subtask")
        task = _tasks(cfg.language)[task_id]
        subtask = task.subtasks[int(index)]
        outer = load_outer_config(cfg.outer_config)
        if outer.memory.backend == "none":
            raise ValueError(f"{NAME} 必须带 Memory:{cfg.outer_config} 的 memory.backend 不能是 none")
        if outer.model_call.params:
            raise ValueError("训练时策略模型的请求参数由 Task Config 的 agent.max_tokens_per_turn 决定,请清空 model_call.params")
        outer.model = policy_model_config(cfg.agent.model, max_tokens=cfg.agent.max_tokens_per_turn, timeout_s=cfg.agent.timeout_s)
        outer.retry.model_call.max_retries = 0
        outer.memory.require_prefetch = True
        with tempfile.TemporaryDirectory(prefix="pat-vita-") as tmp:
            outer.exec.db_path = str(Path(tmp) / "exec.sqlite3")
            rt = build_runtime(outer)
            try:
                runner = PersonalizationRunner(task, rt=rt, user_llm=shared.role("user", cfg.models_file), judge=shared.role("judge", cfg.models_file),
                                               max_steps=cfg.max_steps, language=cfg.language)
                result = runner.run_subtask(subtask, int(index))
                errors = attempt_errors(outer.exec.db_path)
                infra = [c for c in errors if c in INFRA_CODES]
                if infra:
                    raise RuntimeError(f"vitabench {subtask.subtask_id}: 模型请求出现基础设施错误,丢弃这条样本: {infra}")
                rejections = [c for c in errors if c not in INFRA_CODES]
                if rejections:
                    reward, note = 0.0, f"执行层拒绝了 {len(rejections)} 次决策"
                else:
                    info = runner.evaluate_subtask(subtask, result)
                    if info is None:
                        raise RuntimeError(f"vitabench {subtask.subtask_id}: 没有评分标准,不能用于训练")
                    if (info.info or {}).get("evaluation_error"):
                        raise RuntimeError(f"vitabench {subtask.subtask_id}: 评分出错,丢弃这条样本: {info.info.get('note')}")
                    reward, note = float(info.reward), "rubric"
            finally:
                rt.close()
        conversation = result.conversation
        return TaskResult(
            reward=reward, accuracy=reward, finished=True,
            extra_info={
                "task_id": task_id, "subtask": int(index), "subtask_id": subtask.subtask_id, "domain": subtask.domain,
                "termination_reason": conversation.termination_reason.value, "num_messages": len(conversation.messages),
                "rejections": rejections, "reward_source": note,
            },
        )
