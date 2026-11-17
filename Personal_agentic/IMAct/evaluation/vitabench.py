"""VitaBench personalization 的 rubric 评分。

一个 subtask 的完整对话被切成重叠的滑动窗口,裁判模型逐窗口更新每个得分点(rubric)是否满足,状态跨窗口保留;
全部得分点满足记 1 分,否则 0 分,同时报告逐项通过比例。裁判输出必须完整合法(每个窗口都返回全部 rubric,
布尔 ``meetExpectation`` 与文字 ``justification``),否则抛 ``JudgeError``,不会悄悄当成零分。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from json_repair import repair_json

from ..harness.conversation import ConversationResult, TerminationReason
from ..harness.llm import LLMCaller
from ..harness.messages import Message, SystemMessage, UserMessage
from .base import JudgeError, NLRubricCheck, RewardInfo, RewardType

WINDOW_SIZE = 10
OVERLAP = 2
PREMATURE = {TerminationReason.TOO_MANY_ERRORS, TerminationReason.MAX_STEPS, TerminationReason.INVALID_AGENT_MESSAGE}
_TEMPLATE = Path(__file__).with_name("vitabench_sliding_window.yaml")


@dataclass
class RubricTask:
    """评估一个 subtask 所需的任务信息。"""

    instruction: str
    rubrics: list[str]
    env_time: str | None = None  # 任务 environment.time,告诉裁判"现在几点"


@lru_cache(maxsize=None)
def _template(language: str) -> str:
    return yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))[language]


def _weekday(date: str, language: str) -> str:
    from datetime import datetime

    weekday = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").weekday()
    if language == "english":
        return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday]
    return f"星期{'一二三四五六日'[weekday]}"


class VitaBenchEvaluator:
    def __init__(self, judge: LLMCaller, language: str = "chinese"):
        self.judge = judge
        self.language = language

    def evaluate(self, result: ConversationResult, task: RubricTask) -> RewardInfo:
        if result.termination_reason in PREMATURE:
            return RewardInfo(reward=0.0, info={"note": f"Simulation terminated prematurely. Termination reason: {result.termination_reason.value}"})
        rubrics = list(dict.fromkeys(task.rubrics))
        if not rubrics:
            return RewardInfo(reward=1.0, nl_rubrics=[], info={"note": "No rubric to evaluate"}, reward_breakdown={RewardType.NL_ASSERTION: 1.0})

        env_info = {"system_time": "", "database": []}
        if task.env_time:
            env_info["system_time"] = f"{task.env_time} {_weekday(task.env_time, self.language) or ''}"

        states = {f"rubric_{i}": {"rubric": r, "justification": "Not evaluated yet", "meetExpectation": False} for i, r in enumerate(rubrics)}
        windows = self._windows(result.messages)
        step = WINDOW_SIZE - OVERLAP
        window_evaluations = []
        for i, window in enumerate(windows):
            try:
                states, info = self._evaluate_window(env_info, task, window, states, i + 1, len(windows), i * step)
            except JudgeError as exc:
                exc.windows = window_evaluations + exc.windows
                raise
            window_evaluations.append(info)

        checks = [NLRubricCheck(nl_rubric=s["rubric"], met=s["meetExpectation"], justification=s["justification"]) for s in states.values()]
        score = sum(c.met for c in checks) / len(checks)
        return RewardInfo(
            reward=1.0 if all(c.met for c in checks) else 0.0,
            nl_rubrics=checks,
            reward_breakdown={RewardType.NL_ASSERTION: score},
            info={"evaluation_method": "sliding_window", "num_windows": len(windows), "window_size": WINDOW_SIZE},
            window_evaluations=window_evaluations,
        )

    @staticmethod
    def _windows(messages: list[Message]) -> list[list[Message]]:
        if len(messages) <= WINDOW_SIZE:
            return [messages]
        windows, step, i = [], WINDOW_SIZE - OVERLAP, 0
        while i < len(messages):
            window = messages[i : i + WINDOW_SIZE]
            if window:
                windows.append(window)
            if i + WINDOW_SIZE >= len(messages):
                break
            i += step
        return windows

    def _evaluate_window(self, env_info, task, window, states, idx, total, start) -> tuple[dict, dict]:
        system_prompt = _template(self.language).format(env_info=env_info, user_instruction=task.instruction, window_idx=idx, total_windows=total)
        user_prompt = f"""
# Input
<window_content>
{format_window(window, start)}
</window_content>

<current_rubrics>
{format_rubrics(states)}
</current_rubrics>
"""
        reply = self.judge.chat([SystemMessage(role="system", content=system_prompt), UserMessage(role="user", content=user_prompt)])
        info = {"window_idx": idx, "system_prompt": system_prompt, "user_prompt": user_prompt,
                "assistant_message_content": reply.content, "assistant_message_usage": reply.usage}
        try:
            return apply_judgement(states, reply.content, window_idx=idx), info
        except JudgeError as exc:
            exc.windows = [info]
            raise


def format_window(window: list[Message], start: int = 0) -> str:
    """窗口内的对话,带全局序号;助手的工具调用写成 ``name(arg=...)`` 追加在正文后。"""
    lines = []
    for i, message in enumerate(window):
        role = getattr(message, "role", "unknown")
        text = getattr(message, "content", "")
        calls = getattr(message, "tool_calls", None) if role == "assistant" else None
        if calls:
            rendered = [f"{c.name}({', '.join(f'{k}={v!r}' for k, v in c.arguments.items())})" for c in calls]
            text = f"{text} {'.'.join(rendered)}" if text else ".".join(rendered)
        if text:
            lines.append(f"[{start + i + 1}] {role}: {text}")
    return "\n".join(lines)


def format_rubrics(states: dict) -> str:
    return json.dumps([{"rubric_idx": k, "rubric": s["rubric"], "justification": s["justification"], "meetExpectation": s["meetExpectation"]}
                       for k, s in states.items()], ensure_ascii=False, indent=2)


def parse_judgement(content: str) -> list[dict[str, Any]]:
    try:
        return json.loads(repair_json(content))
    except (ValueError, TypeError) as exc:
        raise JudgeError(f"裁判响应无法解析为 JSON: {exc}") from exc


def apply_judgement(states: dict, content: str, *, window_idx: int | None = None) -> dict:
    """校验并应用一个窗口的裁判输出,返回新的 rubric 状态。"""
    where = f"窗口 {window_idx}: " if window_idx else ""
    rows = parse_judgement(content)
    if not isinstance(rows, list) or not rows:
        raise JudgeError(f"{where}裁判响应必须是非空 rubric 数组")
    updated, seen = copy.deepcopy(states), set()
    for row in rows:
        if not isinstance(row, dict):
            raise JudgeError(f"{where}裁判 rubric 必须是对象")
        key = row.get("rubric_idx")
        if not isinstance(key, str) or key not in states or key in seen:
            raise JudgeError(f"{where}裁判响应含未知或重复 rubric_idx")
        if type(row.get("meetExpectation")) is not bool or not isinstance(row.get("justification"), str):
            raise JudgeError(f"{where}裁判评分必须含布尔 meetExpectation 与文字 justification")
        seen.add(key)
        updated[key]["justification"] = row["justification"]
        updated[key]["meetExpectation"] = row["meetExpectation"]
    if seen != set(states):
        raise JudgeError(f"{where}裁判窗口缺少 rubric: {sorted(set(states) - seen)}")
    return updated
