"""τ² 单个 task 的运行流程:环境初始化 → 用户模拟器 + agent 对话 → 评分。

和 VitaBench 一样,这里只做 τ² 自己的事(把 task 变成 harness 能跑的东西);对话循环、用户模拟、评分流程都是通用的。
一个 ``Tau2Runner`` 对应一个 cell(语言 × domain),只读,可以被并发的多个 task 共用。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ...evaluation.base import RewardInfo
from ...evaluation.tau2 import Tau2EvalTask, Tau2Evaluator
from ...harness import EpisodeSpec, LLMCaller, SimulationRun, UserSimulator, run_episode
from ...harness.messages import now_str
from ...harness.native_agent import NativeAgent
from ...trace import TraceRecorder
from .data_model.tasks import Task
from .environment.toolkit import get_tool_types
from .harness_env import Tau2HarnessEnvironment
from .multilingual import Cell
from .prompts import agent_domain_policy, agent_system_prompt, user_system_prompt

logger = logging.getLogger(__name__)

BENCH = "tau2"
DEFAULT_MAX_STEPS = 200
DEFAULT_MAX_ERRORS = 10


@dataclass
class Prepared:
    """一个 task 跑起来所需的全部东西。"""

    environment: Tau2HarnessEnvironment
    spec: EpisodeSpec
    user: UserSimulator
    greeting: str
    system_prompt: str


class Tau2Runner:
    def __init__(
        self,
        cell: Cell,
        *,
        agent_llm: LLMCaller | None = None,
        make_agent: Optional[Callable[[Prepared, Optional[int]], Any]] = None,
        user_llm: LLMCaller,
        judge: LLMCaller | None = None,
        score: bool = True,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_errors: int = DEFAULT_MAX_ERRORS,
        communicate_scorer: str = "semantic",
        trace: TraceRecorder | None = None,
        conversation_options: dict | None = None,
    ):
        """``score=False`` 只跑对话不评分。评分本身不一定要裁判:环境终态、动作类评分是纯规则,``judge`` 只在
        语义类分量(NL 断言、语义模式的 communicate)用到;没给 judge 而 task 需要它时,该 task 的评分报 ``JudgeError``。

        被测 agent 二选一:``agent_llm``(原生 function-calling,默认)或 ``make_agent(prepared, seed)``(自定义,如对接执行层)。
        每个 task 都会新建一个 agent:对话历史、执行层会话都是 per-task 的。
        ``conversation_options``:额外交给对话循环的选项(如 RL 训练时的 ``max_agent_retries=1``),不覆盖官方口径的那几项。
        """
        if (agent_llm is None) == (make_agent is None):
            raise ValueError("agent_llm 与 make_agent 必须且只能给一个")
        self.cell = cell
        self.domain = cell.domain
        self.agent_llm = agent_llm
        self.make_agent = make_agent
        self.user_llm = user_llm
        self.evaluator = Tau2Evaluator(judge, communicate_scorer=communicate_scorer) if score else None
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.trace = trace
        self.conversation_options = dict(conversation_options or {})

    # ------------------------------------------------------------------ 准备
    def prepare(self, task: Task, *, seed: Optional[int] = None) -> Prepared:
        cell = self.cell
        environment = Tau2HarnessEnvironment(cell.environment(), task)
        persona = cell.persona(task)
        system_prompt = agent_system_prompt(agent_domain_policy(environment.env.get_policy(), cell, persona))
        user_tools = environment.user_tool_schemas()
        user = UserSimulator(
            self.user_llm,
            user_system_prompt(cell, task, use_tools=user_tools is not None),
            opening=None,  # 用户的每一句(包括第一句)都由模型按场景生成
            tools=user_tools,
            seed=seed,
            trace=self.trace,
        )
        spec = EpisodeSpec(bench=BENCH, user_id=task.id, episode_id=task.id, domain=cell.domain, instruction="",
                           timestamp=None, system_prompt=system_prompt, tools=environment.tool_schemas())
        return Prepared(environment, spec, user, cell.greeting(task), system_prompt)

    def eval_task(self, task: Task) -> Tau2EvalTask:
        state = task.initial_state
        env = self.cell.environment()
        tool_types = dict(get_tool_types(env.tools)) if env.tools is not None else {}
        if env.user_tools is not None:
            tool_types.update(get_tool_types(env.user_tools))
        return Tau2EvalTask(
            criteria=task.evaluation_criteria,
            environment_factory=self.cell.environment,
            initialization_data=state.initialization_data if state else None,
            initialization_actions=state.initialization_actions if state else None,
            message_history=list(state.message_history) if state and state.message_history else [],
            tool_types=tool_types,
            customer_language=self.cell.language if self.cell.cross_lingual else None,
        )

    # ------------------------------------------------------------------ 运行
    def run_task(self, task: Task, *, run_id: str, seed: Optional[int] = None) -> SimulationRun:
        started_at, started = now_str(), time.perf_counter()
        prepared = self.prepare(task, seed=seed)
        agent = self.make_agent(prepared, seed) if self.make_agent else NativeAgent(
            self.agent_llm, prepared.system_prompt, prepared.spec.tools, seed=seed, trace=self.trace)
        result = run_episode(
            agent, prepared.spec, prepared.user, prepared.environment,
            message_history=list(task.initial_state.message_history) if task.initial_state and task.initial_state.message_history else None,
            max_steps=self.max_steps, max_errors=self.max_errors, greeting=prepared.greeting,
            agent_sees_greeting=True, agent_can_stop=False, finish_tool_round_before_limits=True, language=self.cell.language,
            **self.conversation_options,
        )
        reward = self.evaluate(task, result) if self.evaluator is not None else None
        return SimulationRun(
            id=str(uuid.uuid4()), task_id=task.id, start_time=started_at, end_time=now_str(), duration=time.perf_counter() - started,
            termination_reason=result.termination_reason, reward_info=reward, agent_cost=result.agent_cost, user_cost=result.user_cost,
            messages=result.messages, states={}, seed=seed,
        )

    def evaluate(self, task: Task, result) -> RewardInfo:
        """评分失败(裁判输出不可用)按 0 分记录并在 info 里标出,不拖垮整个任务。"""
        try:
            return self.evaluator.evaluate(result, self.eval_task(task))
        except Exception as exc:  # noqa: BLE001
            logger.error("Error evaluating task %s: %s", task.id, exc)
            return RewardInfo(reward=0.0, info={"note": f"Evaluation error: {exc}", "evaluation_error": True})
