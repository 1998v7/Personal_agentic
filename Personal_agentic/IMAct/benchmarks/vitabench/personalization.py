"""VitaBench personalization 的任务编排:一个用户 = 若干个按顺序执行的 subtask。

这里只做 VitaBench 自己的事 —— 把 subtask 变成 harness 能跑的东西(agent 的 system prompt、环境与工具、用户模拟器的人设),
然后交给通用的 ``run_episode`` 去跑,最后用 ``evaluation.vitabench`` 打分并汇总。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from ...evaluation.base import RewardInfo
from ...evaluation.vitabench import RubricTask, VitaBenchEvaluator
from ...exec_layer import ExecRuntime
from ...harness import HarnessAgent, LLMCaller, SimulationRun, UserSimulator, run_episode
from ...harness.conversation import ConversationResult, TerminationReason, sum_costs
from ...harness.messages import Message, now_str
from ...harness.protocols import EpisodeSpec
from .data_model.personalization_task import PersonalizationTask, SubTask
from .data_model.tasks import EvaluationCriteria
from .domains import build_environment
from .environment.environment import Environment
from .prompts import get_prompts
from .tasks import current_time_map
from .utils import get_weekday

logger = logging.getLogger(__name__)

BENCH = "vitabench"

# 链路诊断用:用户同意推荐不等于任务已执行完,避免用户在助手还没下单时就结束对话。
COMPLETION_GUARD = """
补充结束条件说明（链路诊断）：同意推荐、确认日期/房型/价格或授权执行，只表示允许助手继续，
不等于原请求已执行完成。若原请求要求下单、付款、预订或预约，在助手仍询问是否执行时，
请回复确认并等待执行结果，不要同时输出 ###STOP###。仅查询/推荐任务可在得到满意答案后结束。
你仍可明确拒绝、取消或结束任务；不要为了继续对话而编造已完成的业务结果。
""".strip()


# ---------------------------------------------------------------------- prompt 与评分点
def format_user_profile(profile: dict[str, Any]) -> str:
    lines = []
    for key, value in (profile or {}).items():
        if isinstance(value, list):
            value = "、".join(str(v) for v in value)
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def agent_system_prompt(subtask: SubTask, user_profile: dict[str, Any], clock: str | None, language: str) -> str:
    """助手的 system prompt。不含 VitaBench 自带的记忆段:记忆只能通过 start.memory 与 query_memory 进入。"""
    prompts = get_prompts(language)
    template = prompts.personalization_agent_proactive_system_prompt if "proactive" in subtask.skill_tested else prompts.personalization_agent_system_prompt
    base = (template + "\n\n" + prompts.personalization_agent_addendum).format(time=clock if clock is not None else time.strftime("%Y-%m-%d %H:%M:%S"))
    profile = format_user_profile(user_profile)
    if not profile:
        return base
    return f"{base}\n\n## 当前用户基础信息\n以下是该用户的账户注册信息,可直接使用:\n{profile}"


def user_system_prompt(subtask: SubTask, persona: str, language: str, *, completion_guard: bool = False) -> str:
    """用户模拟器的 system prompt;只有带隐藏意图的 proactive subtask 才加 proactive 段。"""
    prompts = get_prompts(language)
    proactive = ""
    if "proactive" in subtask.skill_tested and subtask.user_intention:
        proactive = prompts.personalization_user_proactive_section.format(proactive_info=subtask.user_intention) + "\n"
    prompt = prompts.personalization_user_system_prompt.format(persona=persona, instruction=subtask.instruction, proactive_section=proactive)
    return prompt + "\n\n" + COMPLETION_GUARD if completion_guard else prompt


def rubrics_of(criteria: EvaluationCriteria | None) -> list[str]:
    rubrics: list[str] = []
    if criteria is None:
        return rubrics
    for state in criteria.expected_states or []:
        rubrics.extend(getattr(state, "state_rubrics", None) or [])
    rubrics.extend(criteria.overall_rubrics or [])
    return list(dict.fromkeys(rubrics))


# ---------------------------------------------------------------------- Memory 时间口径
def memory_timestamp(raw: Any, mode: str) -> Any:
    """VitaBench 的 environment.time 是 'YYYY-MM-DD HH:MM:SS';Memory 网关的 task.timestamp 示例是纯日期。"""
    if mode == "date" and isinstance(raw, str):
        return raw.strip().replace("T", " ").split(" ")[0]
    return raw


def episode_timestamp(subtask: SubTask, memory_cfg) -> Any:
    """Memory 要求 ``task.timestamp`` 等于原任务的 ``current_time``;读不到时,本地 Memory 报错,网关回退到环境时间。"""
    raw = current_time_map().get(subtask.subtask_id)
    if memory_cfg.backend == "local":
        if raw is None:
            raise ValueError(f"本地 Memory 需要原始 current_time: {subtask.subtask_id};检查 IMACT_VITABENCH_TASKS")
        return raw
    return memory_timestamp(raw or (subtask.environment or {}).get("time"), memory_cfg.timestamp_mode)


# ---------------------------------------------------------------------- 编排
@dataclass
class SubtaskResult:
    subtask_id: str
    subtask_idx: int
    conversation: ConversationResult

    def as_dict(self) -> dict[str, Any]:
        c = self.conversation
        return {"subtask_id": self.subtask_id, "subtask_idx": self.subtask_idx, "messages": c.messages, "states": c.states,
                "termination_reason": c.termination_reason, "duration": c.duration}


class PersonalizationRunner:
    """跑一个 PersonalizationTask。``judge`` 为空时只跑对话不打分(链路诊断)。"""

    def __init__(
        self,
        task: PersonalizationTask,
        *,
        rt: ExecRuntime,
        user_llm: LLMCaller,
        judge: LLMCaller | None = None,
        run_id: str | None = None,
        max_steps: int = 100,
        language: str = "chinese",
        seed: Optional[int] = None,
        completion_guard: bool = False,
    ):
        self.task = task
        self.rt = rt
        self.user_llm = user_llm
        self.evaluator = VitaBenchEvaluator(judge, language) if judge else None
        self.agent = HarnessAgent(runtime=rt, run_id=run_id)
        self.max_steps = max_steps
        self.language = language
        self.seed = seed
        self.completion_guard = completion_guard
        # 助手看到的"现在时间":每个有时间的 subtask 会更新;没有时间的沿用上一个
        first_time = task.subtasks[0].environment.get("time") if task.subtasks else None
        self._clock: str | None = f"{first_time} {get_weekday(first_time, language)}" if first_time else None

    def prepare(self, subtask: SubTask) -> tuple[EpisodeSpec, Environment, UserSimulator]:
        environment = build_environment(subtask.domain, subtask.environment, self.language)
        subtask_time = subtask.environment.get("time")
        if subtask_time:
            self._clock = f"{subtask_time} {get_weekday(subtask_time, self.language)}"
        spec = EpisodeSpec(
            bench=BENCH,
            user_id=self.task.user_id,
            episode_id=subtask.subtask_id,
            domain=subtask.domain,
            instruction=subtask.instruction,
            timestamp=episode_timestamp(subtask, self.rt.cfg.memory),
            system_prompt=agent_system_prompt(subtask, self.task.user_profile, self._clock, self.language),
            tools=environment.tool_schemas(),
        )
        user = UserSimulator(
            self.user_llm,
            user_system_prompt(subtask, str(self.task.user_profile), self.language, completion_guard=self.completion_guard),
            opening=subtask.instruction,
            seed=self.seed,
            trace=self.rt.trace,
        )
        return spec, environment, user

    def run_subtask(self, subtask: SubTask, index: int = 0, *, on_prepared=None) -> SubtaskResult:
        """``on_prepared(spec)`` 在环境与用户都建好、对话开始之前调用(诊断模式用来记录起始轨迹)。"""
        spec, environment, user = self.prepare(subtask)
        if on_prepared:
            on_prepared(spec)
        result = run_episode(self.agent, spec, user, environment, message_history=subtask.message_history,
                             max_steps=self.max_steps, language=self.language)
        return SubtaskResult(subtask.subtask_id, index, result)

    def evaluate_subtask(self, subtask: SubTask, result: SubtaskResult) -> Optional[RewardInfo]:
        """没有评分标准返回 None;裁判失败按 0 分记录并在 info 里标出,不拖垮整个任务。"""
        criteria = subtask.evaluation_criteria
        if criteria is None or self.evaluator is None:
            return None
        task = RubricTask(instruction=subtask.instruction, rubrics=rubrics_of(criteria), env_time=subtask.environment.get("time"))
        try:
            return self.evaluator.evaluate(result.conversation, task)
        except Exception as exc:
            logger.error("Error evaluating subtask %s: %s", subtask.subtask_id, exc)
            return RewardInfo(reward=0.0, info={"note": f"Evaluation error: {exc}", "evaluation_error": True})

    def run(self) -> SimulationRun:
        started_at, started = now_str(), time.perf_counter()
        logger.info("personalization start: task=%s user=%s subtasks=%d", self.task.id, self.task.user_id, len(self.task.subtasks))
        results: list[SubtaskResult] = []
        rewards: list[Optional[RewardInfo]] = []
        for index, subtask in enumerate(self.task.subtasks):
            logger.info("subtask %d/%d: id=%s domain=%s", index + 1, len(self.task.subtasks), subtask.subtask_id, subtask.domain)
            result = self.run_subtask(subtask, index)
            reward = self.evaluate_subtask(subtask, result)
            logger.info("subtask %s: %d messages, termination=%s, reward=%s", subtask.subtask_id, len(result.conversation.messages),
                        result.conversation.termination_reason.value, reward.reward if reward else None)
            results.append(result)
            rewards.append(reward)

        messages: list[Message] = [m for r in results for m in r.conversation.messages]
        states: dict[str, Any] = {"old_states": [], "new_states": []}
        for r in results:
            states["old_states"].extend(r.conversation.states.get("old_states", []))
            states["new_states"].extend(r.conversation.states.get("new_states", []))
        costs = sum_costs(messages)
        return SimulationRun(
            id=str(uuid.uuid4()),
            task_id=self.task.id,
            start_time=started_at,
            end_time=now_str(),
            duration=time.perf_counter() - started,
            termination_reason=overall_termination([r.conversation.termination_reason for r in results]),
            reward_info=self.aggregate(rewards),
            agent_cost=costs[0] if costs else None,
            user_cost=costs[1] if costs else None,
            messages=messages,
            states=states,
            seed=self.seed,
        )

    def aggregate(self, rewards: list[Optional[RewardInfo]]) -> RewardInfo:
        """各 subtask 得分取平均;没有评分标准的 subtask 不计入。"""
        valid = [r for r in rewards if r is not None]
        if not valid:
            return RewardInfo(reward=0.0, info={"note": "No subtasks were evaluated"})
        return RewardInfo(
            reward=sum(r.reward for r in valid) / len(valid),
            info={
                "num_subtasks": len(self.task.subtasks),
                "num_evaluated": len(valid),
                "subtask_rewards": {f"subtask_{i}_reward": r.reward for i, r in enumerate(rewards) if r is not None},
                "subtask_skill_tested": {f"subtask_{i}": s.skill_tested for i, s in enumerate(self.task.subtasks) if i < len(rewards)},
                "subtask_domains": {f"subtask_{i}": s.domain for i, s in enumerate(self.task.subtasks) if i < len(rewards)},  # 结果表按业务域拆分用
                "aggregation": "average",
            },
        )


def overall_termination(reasons: list[TerminationReason]) -> TerminationReason:
    """有 subtask 因错误过多终止就整体记为 too_many_errors,否则取最后一个 subtask 的结束原因。"""
    if not reasons:
        return TerminationReason.MAX_STEPS
    if TerminationReason.TOO_MANY_ERRORS in reasons:
        return TerminationReason.TOO_MANY_ERRORS
    return reasons[-1]
