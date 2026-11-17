"""对话循环:用户 ⇄ 助手 ⇄ 环境,直到有人结束。与 benchmark 无关。

三方的消息路由与原 VitaBench orchestrator 一致:
* 助手/环境 → 用户:用户模拟器生成下一句;
* 用户/环境 → 助手:助手最多重试 3 次拿到有效消息(每次用原状态的深拷贝,执行层按相同请求重放,不会重复决策);
* 助手/用户 → 环境:逐个执行工具调用,多个回执合并成 ``MultiToolMessage``。
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from .agent import HarnessAgent, is_agent_stop, is_valid_agent_history_message
from .messages import AssistantMessage, Message, MultiToolMessage, ToolMessage, UserMessage
from .protocols import Environment
from .user import UserSimulator, UserState, is_valid_user_history_message

logger = logging.getLogger(__name__)

MAX_AGENT_RETRIES = 3


class TerminationReason(str, Enum):
    USER_STOP = "user_stop"
    AGENT_STOP = "agent_stop"
    MAX_STEPS = "max_steps"
    TOO_MANY_ERRORS = "too_many_errors"
    INVALID_AGENT_MESSAGE = "invalid_agent_message"


class Role(str, Enum):
    AGENT = "agent"
    USER = "user"
    ENV = "env"


@dataclass
class ConversationResult:
    messages: list[Message]
    termination_reason: TerminationReason
    duration: float
    states: dict[str, Any] = field(default_factory=lambda: {"old_states": [], "new_states": []})
    agent_cost: Optional[float] = None
    user_cost: Optional[float] = None


def default_greeting(language: str | None = None) -> AssistantMessage:
    content = "你好，请问需要什么服务？" if language in (None, "chinese") else "Hello, how can I help you?"
    return AssistantMessage(role="assistant", content=content, cost=0.0)


# 默认的收尾规则下,任何一步之后只要达到步数 / 错误上限就立刻结束。
# 工具往返优先:助手刚发出工具调用时,先让环境回执,再判断上限(τ²-bench 的口径)。


def sum_costs(messages: list[Message]) -> tuple[float, float] | None:
    """(助手成本, 用户成本);任一条消息缺成本就整体放弃。"""
    agent_cost = user_cost = 0.0
    for m in messages:
        if isinstance(m, (ToolMessage, MultiToolMessage)):
            continue
        if m.cost is None:
            return None
        if isinstance(m, AssistantMessage):
            agent_cost += m.cost
        elif isinstance(m, UserMessage):
            user_cost += m.cost
    return agent_cost, user_cost


class Conversation:
    def __init__(
        self,
        agent: HarnessAgent,
        user: UserSimulator,
        environment: Environment,
        *,
        message_history: list[Message] | None = None,
        max_steps: int = 100,
        language: str | None = None,
        max_errors: int | None = None,
        greeting: str | None = None,
        agent_sees_greeting: bool = False,
        agent_can_stop: bool = True,
        finish_tool_round_before_limits: bool = False,
        max_agent_retries: int = MAX_AGENT_RETRIES,
    ):
        """
        ``max_errors``:工具回执报错累计达到该值就以 too_many_errors 结束;None 表示不统计。
        ``greeting``:助手的开场白;缺省按 ``language`` 取默认问候。
        ``agent_sees_greeting``:True 时开场白也放进助手自己的历史(助手知道自己先说了话)。
        ``agent_can_stop``:False 时助手消息里的 ###STOP### 不结束对话,只有用户能结束。
        ``finish_tool_round_before_limits``:True 时,助手刚发出工具调用、环境还没回执的那一步不检查上限。
        ``max_agent_retries``:助手给出无效消息(既无文本也无工具调用)时最多重新生成几次。RL 训练时设为 1:
        每次重新生成在训练侧都会变成一条额外的样本。
        """
        if max_agent_retries < 1:
            raise ValueError("max_agent_retries 必须 >= 1")
        self.agent = agent
        self.user = user
        self.environment = environment
        self.message_history = message_history
        self.max_steps = max_steps
        self.language = language
        self.max_errors = max_errors
        self.greeting = greeting
        self.agent_sees_greeting = agent_sees_greeting
        self.agent_can_stop = agent_can_stop
        self.finish_tool_round_before_limits = finish_tool_round_before_limits
        self.max_agent_retries = max_agent_retries
        self.num_errors = 0
        self.agent_state: Any = None
        self.user_state: Optional[UserState] = None
        self.trajectory: list[Message] = []
        self.step_count = 0
        self.done = False
        self.termination_reason: Optional[TerminationReason] = None
        self.from_role: Optional[Role] = None
        self.to_role: Optional[Role] = None
        self.message: Optional[Message] = None

    # ------------------------------------------------------------------ 初始化
    def initialize(self) -> None:
        history = deepcopy(self.message_history) if self.message_history else []
        for msg in history:
            msg.turn_idx = None
        if not history:
            first = default_greeting(self.language)
            if self.greeting is not None:
                first.content = self.greeting
            self.agent_state = self.agent.get_init_state(message_history=[first] if self.agent_sees_greeting else None)
            self.user_state = self.user.get_init_state()
            self.trajectory = [first]
            self.message = first
            self.from_role, self.to_role = Role.AGENT, Role.USER
            return

        self.validate_message_history(history)
        last = history[-1]
        agent_hist = lambda msgs: [m for m in msgs if is_valid_agent_history_message(m)]  # noqa: E731
        user_hist = lambda msgs: [m for m in msgs if is_valid_user_history_message(m)]  # noqa: E731
        if isinstance(last, AssistantMessage):
            self.from_role, self.to_role = Role.AGENT, (Role.ENV if last.is_tool_call() else Role.USER)
            self.agent_state = self.agent.get_init_state(message_history=agent_hist(history))
            self.user_state = self.user.get_init_state(message_history=user_hist(history[:-1]))
            if self.agent_can_stop and is_agent_stop(last):
                self.done, self.termination_reason = True, TerminationReason.AGENT_STOP
        elif isinstance(last, UserMessage):
            self.from_role, self.to_role = Role.USER, (Role.ENV if last.is_tool_call() else Role.AGENT)
            self.user_state = self.user.get_init_state(message_history=user_hist(history))
            self.agent_state = self.agent.get_init_state(message_history=agent_hist(history[:-1]))
            self.done = self.user.is_stop(last)
            if self.done:
                self.termination_reason = TerminationReason.USER_STOP
        elif isinstance(last, ToolMessage):
            self.from_role = Role.ENV
            if last.requestor == "assistant":
                self.to_role = Role.AGENT
                self.agent_state = self.agent.get_init_state(message_history=agent_hist(history[:-1]))
                self.user_state = self.user.get_init_state(message_history=user_hist(history))
            else:
                self.to_role = Role.USER
                self.agent_state = self.agent.get_init_state(message_history=agent_hist(history))
                self.user_state = self.user.get_init_state(message_history=user_hist(history[:-1]))
        else:
            raise ValueError(f"Last message should be of type AssistantMessage, UserMessage, or ToolMessage, got {type(last)}")
        self.message = last
        self.trajectory = history

    # ------------------------------------------------------------------ 运行
    def run(self) -> ConversationResult:
        started = time.perf_counter()
        self.initialize()
        while not self.done:
            self.step()
            self._check_limits()
        messages = self.get_trajectory()
        costs = sum_costs(messages)
        return ConversationResult(
            messages=messages,
            termination_reason=self.termination_reason,
            duration=time.perf_counter() - started,
            states=self.environment.get_states(),
            agent_cost=costs[0] if costs else None,
            user_cost=costs[1] if costs else None,
        )

    def _check_limits(self) -> None:
        if self.finish_tool_round_before_limits and self.to_role == Role.ENV:
            return
        if self.step_count >= self.max_steps:
            self.done, self.termination_reason = True, TerminationReason.MAX_STEPS
        if self.max_errors is not None and self.num_errors >= self.max_errors:
            self.done, self.termination_reason = True, TerminationReason.TOO_MANY_ERRORS

    def step(self) -> None:
        if self.done:
            raise ValueError("Conversation is done")
        logger.debug("Step %d: %s -> %s", self.step_count, self.from_role, self.to_role)
        if self.from_role in (Role.AGENT, Role.ENV) and self.to_role == Role.USER:
            self._user_turn()
        elif self.from_role in (Role.USER, Role.ENV) and self.to_role == Role.AGENT:
            if not self._agent_turn():
                return
        elif self.from_role in (Role.AGENT, Role.USER) and self.to_role == Role.ENV:
            self._env_turn()
        else:
            raise ValueError(f"Invalid role combination. From role: {self.from_role}, To role: {self.to_role}")
        self.step_count += 1

    def _user_turn(self) -> None:
        user_msg, self.user_state = self.user.generate_next_message(self.message, self.user_state)
        user_msg.validate()
        if self.user.is_stop(user_msg):
            self.done, self.termination_reason = True, TerminationReason.USER_STOP
        self.trajectory.append(user_msg)
        self.message = user_msg
        self.from_role = Role.USER
        self.to_role = Role.ENV if user_msg.is_tool_call() else Role.AGENT

    def _agent_turn(self) -> bool:
        """返回 False 表示连续无效消息,对话已终止(这一步不计数)。"""
        original = deepcopy(self.agent_state)
        agent_msg = None
        for attempt in range(1, self.max_agent_retries + 1):
            candidate, updated = self.agent.generate_next_message(self.message, deepcopy(original))
            if candidate is not None and (candidate.has_text_content() or candidate.is_tool_call()):
                agent_msg, self.agent_state = candidate, updated
                break
            logger.warning("Agent generated invalid message (attempt %d/%d): %s", attempt, self.max_agent_retries, candidate)
        if agent_msg is None:
            self.done, self.termination_reason = True, TerminationReason.INVALID_AGENT_MESSAGE
            return False
        agent_msg.validate()
        if self.agent_can_stop and is_agent_stop(agent_msg):
            self.done, self.termination_reason = True, TerminationReason.AGENT_STOP
        self.trajectory.append(agent_msg)
        self.message = agent_msg
        self.from_role = Role.AGENT
        self.to_role = Role.ENV if agent_msg.is_tool_call() else Role.USER
        return True

    def _env_turn(self) -> None:
        if not self.message.is_tool_call():
            raise ValueError("Agent or User should send tool call to environment")
        tool_msgs = [self.environment.get_response(call) for call in self.message.tool_calls]
        self.num_errors += sum(1 for m in tool_msgs if m.error)
        self.trajectory.extend(tool_msgs)
        self.message = MultiToolMessage(role="tool", tool_messages=tool_msgs) if len(tool_msgs) > 1 else tool_msgs[0]
        self.to_role = self.from_role
        self.from_role = Role.ENV
        sync = getattr(self.environment, "sync_tools", None)
        if sync is not None:
            sync()

    def get_trajectory(self) -> list[Message]:
        out = []
        for i, msg in enumerate(self.trajectory):
            msg = deepcopy(msg)
            msg.turn_idx = i
            out.append(msg)
        return out

    @classmethod
    def validate_message_history(cls, message_history: list[Message]) -> None:
        """只含助手/用户/工具消息;每条带 tool_calls 的消息后紧跟数量相同、requestor 匹配的回执。"""
        expected = 0
        requestor = None
        for msg in message_history:
            if isinstance(msg, (AssistantMessage, UserMessage)):
                msg.validate()
                if msg.is_tool_call():
                    if expected > 0:
                        raise ValueError(f"{expected} tool messages are missing. Got {msg.role} message.")
                    expected = len(msg.tool_calls)
                    requestor = msg.role
                else:
                    requestor = None
            elif isinstance(msg, ToolMessage):
                if expected == 0 or requestor is None:
                    raise ValueError("No tool messages expected.")
                if requestor != msg.requestor:
                    raise ValueError(f"Got tool message from {msg.requestor}, expected {requestor}.")
                expected -= 1
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")
