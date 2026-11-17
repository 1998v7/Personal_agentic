"""τ²-bench 的评分:环境终态、动作、必须告知的信息、自然语言断言。

得分 = 各分量(由 task 的 ``reward_basis`` 决定哪些参与)的乘积。口径与上游一致:

* DB:把 agent 的整段对话里的有状态工具调用在新环境里重放,和"重放参考动作"得到的参考环境比较数据库哈希;
* ENV_ASSERTION:在重放后的环境上跑任务给定的断言函数;
* ACTION:参考动作都出现在 agent(或用户)的工具调用里;
* COMMUNICATE:必须告知用户的信息,由裁判模型按语义判断(也可用子串匹配);
* NL_ASSERTION:自然语言断言,由裁判模型逐条判断。

和上游不同的一点:裁判回复缺项、少项、类型不对时抛 ``JudgeError``,不会因为"没有任何一项不满足"而悄悄得 1 分。
本模块不依赖任何 benchmark 包:环境和任务以鸭子类型传入。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

from json_repair import repair_json

from ..harness.conversation import ConversationResult, TerminationReason
from ..harness.llm import LLMCaller
from ..harness.messages import AssistantMessage, Message, SystemMessage, UserMessage
from .base import (
    ActionCheck, CommunicateCheck, DBCheck, EnvAssertionCheck, JudgeError, NLAssertionCheck, RewardInfo, RewardType,
)

logger = logging.getLogger(__name__)

_PROMPTS = Path(__file__).parent / "tau2_prompts"
FINISHED = {TerminationReason.AGENT_STOP, TerminationReason.USER_STOP}
ENV_BASES = {RewardType.DB, RewardType.ENV_ASSERTION}


@lru_cache(maxsize=None)
def _prompt(name: str) -> str:
    return (_PROMPTS / name).read_text(encoding="utf-8")


@dataclass
class Tau2EvalTask:
    """评分一个 task 需要的全部输入。``criteria`` 是 task 的 evaluation_criteria(鸭子类型)。"""

    criteria: Any  # actions / env_assertions / communicate_info / nl_assertions / reward_basis
    environment_factory: Callable[[], Any]  # 新建一个干净的环境
    initialization_data: Any = None
    initialization_actions: Any = None
    message_history: list = field(default_factory=list)
    tool_types: Optional[dict[str, str]] = None  # 工具名 -> read / write / think / generic
    customer_language: Optional[str] = None  # 通话语言与场景语言不同时的语言代码,语义判断时告诉裁判


def _basis(criteria) -> set[RewardType]:
    return {RewardType(getattr(b, "value", b)) for b in criteria.reward_basis}


def extract_json(text: str) -> str:
    """裁判回复里的 JSON:优先 ```json 代码块,其次第一个 { 到最后一个 }。"""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _parse_json(text: str, what: str) -> Any:
    try:
        return json.loads(repair_json(extract_json(text or "")))
    except (ValueError, TypeError) as exc:
        raise JudgeError(f"{what}: 裁判响应无法解析为 JSON: {exc}") from exc


class Tau2Evaluator:
    def __init__(self, judge: LLMCaller | None = None, *, communicate_scorer: str = "semantic"):
        """``judge`` 只在任务需要语义判断(COMMUNICATE 的语义模式、NL_ASSERTION)时才会被调用。"""
        if communicate_scorer not in ("semantic", "substring"):
            raise ValueError("communicate_scorer must be 'semantic' or 'substring'")
        self.judge = judge
        self.communicate_scorer = communicate_scorer

    # ------------------------------------------------------------------ 入口
    def evaluate(self, result: ConversationResult, task: Tau2EvalTask) -> RewardInfo:
        if result.termination_reason not in FINISHED:
            return RewardInfo(reward=0.0, info={"note": f"Simulation terminated prematurely. Termination reason: {result.termination_reason.value}"})
        criteria = task.criteria
        if criteria is None:
            return RewardInfo(reward=1.0, info={"note": "No evaluation criteria"})
        trajectory = result.messages
        basis = _basis(criteria)

        env_info = self.evaluate_env(task, trajectory)
        action_info = self.evaluate_actions(task, trajectory)
        comm_info = self.evaluate_communicate(task, trajectory) if RewardType.COMMUNICATE in basis else None
        nl_info = self.evaluate_nl_assertions(task, trajectory) if RewardType.NL_ASSERTION in basis else None

        evaluated = ENV_BASES | {RewardType.ACTION}
        if comm_info is not None:
            evaluated.add(RewardType.COMMUNICATE)
        if nl_info is not None:
            evaluated.add(RewardType.NL_ASSERTION)
        if basis - evaluated:
            raise ValueError(f"Task reward_basis includes {sorted(b.value for b in basis - evaluated)} but these were not evaluated")

        reward, breakdown = 1.0, {}
        for part, bases in ((env_info, ENV_BASES), (action_info, {RewardType.ACTION}), (nl_info, {RewardType.NL_ASSERTION})):
            if part is not None and basis & bases:
                breakdown.update(part.reward_breakdown or {})
                reward *= part.reward
        if comm_info is not None:  # 只要评了就相乘,与上游一致
            breakdown.update(comm_info.reward_breakdown or {})
            reward *= comm_info.reward

        return RewardInfo(
            reward=reward,
            db_check=env_info.db_check,
            env_assertions=env_info.env_assertions,
            action_checks=action_info.action_checks,
            nl_assertions=nl_info.nl_assertions if nl_info else None,
            communicate_checks=comm_info.communicate_checks if comm_info else None,
            reward_basis=criteria.reward_basis,
            reward_breakdown=breakdown,
            info={"env": env_info.info, "nl": nl_info.info if nl_info else None,
                  "communicate": comm_info.info if comm_info else None, "action": action_info.info},
        )

    # ------------------------------------------------------------------ 环境终态
    def evaluate_env(self, task: Tau2EvalTask, trajectory: list[Message], *, strict_replay: bool = True) -> RewardInfo:
        criteria = task.criteria
        actions, assertions = criteria.actions, criteria.env_assertions
        if actions is None and assertions is None:
            return RewardInfo(reward=1.0, db_check=DBCheck(db_match=True, db_reward=1.0), info={"note": "No expected actions or env assertions"})

        predicted = task.environment_factory()
        predicted.set_state(initialization_data=task.initialization_data, initialization_actions=task.initialization_actions,
                            message_history=list(trajectory), strict=strict_replay)
        gold = task.environment_factory()
        gold.set_state(initialization_data=task.initialization_data, initialization_actions=task.initialization_actions,
                       message_history=list(task.message_history), strict=strict_replay)
        for action in actions or []:
            try:
                gold.make_tool_call(tool_name=action.name, requestor=action.requestor, **action.arguments)
            except Exception as exc:  # noqa: BLE001 - 参考动作报错不影响评分,与上游一致
                logger.warning("Error in golden actions %s(%s): %s", action.name, action.arguments, exc)

        match = gold.get_db_hash() == predicted.get_db_hash() and gold.get_user_db_hash() == predicted.get_user_db_hash()
        db_reward = 1.0 if match else 0.0
        checks, assertion_reward = [], 1.0
        for assertion in assertions or []:
            ok = predicted.run_env_assertion(assertion, raise_assertion_error=False)
            checks.append(EnvAssertionCheck(env_assertion=assertion.model_dump(mode="json"), met=ok, reward=1.0 if ok else 0.0))
            assertion_reward *= 1.0 if ok else 0.0

        basis, reward, breakdown = _basis(criteria), 1.0, {}
        if RewardType.DB in basis:
            breakdown[RewardType.DB] = db_reward
            reward *= db_reward
        if RewardType.ENV_ASSERTION in basis:
            breakdown[RewardType.ENV_ASSERTION] = assertion_reward
            reward *= assertion_reward
        return RewardInfo(reward=reward, db_check=DBCheck(db_match=match, db_reward=db_reward), env_assertions=checks,
                          reward_basis=criteria.reward_basis, reward_breakdown=breakdown)

    # ------------------------------------------------------------------ 动作
    def evaluate_actions(self, task: Tau2EvalTask, trajectory: list[Message]) -> RewardInfo:
        golden = task.criteria.actions
        if not golden:
            return RewardInfo(reward=1.0, info={"note": "No actions to evaluate"}, reward_breakdown={RewardType.ACTION: 1.0})
        calls = [c for m in trajectory if isinstance(m, (AssistantMessage, UserMessage)) and m.is_tool_call() for c in m.tool_calls]
        checks = []
        for gold in golden:
            found = any(gold.compare_with_tool_call(call) for call in calls)
            tool_type = (task.tool_types or {}).get(gold.name)
            checks.append(ActionCheck(action=gold.model_dump(mode="json"), action_match=found, action_reward=1.0 if found else 0.0,
                                      tool_type=getattr(tool_type, "value", tool_type)))
        reward = 1.0 if all(c.action_match for c in checks) else 0.0
        return RewardInfo(reward=reward, action_checks=checks, reward_breakdown={RewardType.ACTION: reward})

    # ------------------------------------------------------------------ 必须告知的信息
    def evaluate_communicate(self, task: Tau2EvalTask, trajectory: list[Message]) -> RewardInfo:
        infos = task.criteria.communicate_info or []
        if not infos and self.communicate_scorer == "substring":
            return RewardInfo(reward=1.0, info={"note": "No communicate_info to evaluate"}, reward_breakdown={RewardType.COMMUNICATE: 1.0})
        checks = (self._semantic_communicate(trajectory, infos, task.customer_language) if self.communicate_scorer == "semantic"
                  else self._substring_communicate(trajectory, infos))
        reward = float(all(c.met for c in checks))
        return RewardInfo(reward=reward, communicate_checks=checks, reward_breakdown={RewardType.COMMUNICATE: reward},
                          info={"communicate_scorer": self.communicate_scorer} if self.communicate_scorer == "semantic" else None)

    @staticmethod
    def _substring_communicate(trajectory: list[Message], infos: list[str]) -> list[CommunicateCheck]:
        out = []
        for info in infos:
            hit = next((m for m in trajectory if isinstance(m, AssistantMessage) and m.has_text_content()
                        and info.lower() in m.content.lower().replace(",", "")), None)
            out.append(CommunicateCheck(info=info, met=hit is not None,
                                        justification=f"Information '{info}' communicated in the message:\n '{hit.content}'" if hit else f"Information '{info}' not communicated."))
        return out

    def _semantic_communicate(self, trajectory: list[Message], infos: list[str], customer_language: str | None = None) -> list[CommunicateCheck]:
        if not infos:
            return []
        judge = self._require_judge("COMMUNICATE")
        transcript = "\n".join(f"{m.role}: {m.content}" for m in trajectory if isinstance(m, (AssistantMessage, UserMessage)) and m.has_text_content())
        system = _prompt("communicate_system.txt")
        if customer_language is not None:
            system += _prompt("communicate_language_addendum.txt").format(language=customer_language)
        checks = []
        for info in infos:
            reply = judge.chat([SystemMessage(role="system", content=system),
                                UserMessage(role="user", content=_prompt("communicate_user.txt").format(info_str=info, transcript=transcript))])
            data = _parse_json(reply.content, f"communicate_info {info[:30]!r}")
            if not isinstance(data, dict) or type(data.get("met")) is not bool or not isinstance(data.get("reasoning"), str) or not data["reasoning"]:
                raise JudgeError(f"communicate_info {info[:30]!r}: 裁判必须返回含布尔 met 与非空 reasoning 的对象")
            checks.append(CommunicateCheck(info=info, met=data["met"], justification=f"[llm_judge] {data['reasoning']}"))
        return checks

    # ------------------------------------------------------------------ 自然语言断言
    def evaluate_nl_assertions(self, task: Tau2EvalTask, trajectory: list[Message]) -> RewardInfo:
        assertions = task.criteria.nl_assertions
        if not assertions:
            return RewardInfo(reward=1.0, nl_assertions=[], info={"note": "No nl_assertions to evaluate"}, reward_breakdown={RewardType.NL_ASSERTION: 1.0})
        judge = self._require_judge("NL_ASSERTION")
        trajectory_str = "\n".join(f"{m.role}: {m.content}" for m in trajectory)
        reply = judge.chat([SystemMessage(role="system", content=_prompt("nl_assertions_system.txt")),
                            UserMessage(role="user", content=_prompt("nl_assertions_user.txt").format(trajectory_str=trajectory_str, nl_assertions=assertions))])
        data = _parse_json(reply.content, "nl_assertions")
        rows = data.get("results") if isinstance(data, dict) else None
        if not isinstance(rows, list) or len(rows) != len(assertions):
            raise JudgeError(f"nl_assertions: 裁判必须对 {len(assertions)} 条断言各返回一项结果,实际 {len(rows) if isinstance(rows, list) else '无 results 列表'}")
        checks = []
        for assertion, row in zip(assertions, rows):
            if not isinstance(row, dict) or type(row.get("metExpectation")) is not bool or not isinstance(row.get("reasoning"), str):
                raise JudgeError("nl_assertions: 每项结果必须含布尔 metExpectation 与文字 reasoning")
            checks.append(NLAssertionCheck(nl_assertion=assertion, met=row["metExpectation"], justification=row["reasoning"]))
        reward = 1.0 if all(c.met for c in checks) else 0.0
        return RewardInfo(reward=reward, nl_assertions=checks, reward_breakdown={RewardType.NL_ASSERTION: reward})

    def _require_judge(self, what: str) -> LLMCaller:
        if self.judge is None:
            raise JudgeError(f"{what} 需要裁判模型,但没有配置 judge")
        return self.judge
