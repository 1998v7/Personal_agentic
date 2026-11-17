"""一个 episode 的完整生命周期:登记会话 → 跑对话 → 通知执行层结束。"""

from __future__ import annotations

from ..trace import trace_scope
from .agent import HarnessAgent
from .conversation import Conversation, ConversationResult
from .messages import Message
from .protocols import Environment, EpisodeSpec
from .user import UserSimulator

# 对话结束原因 → 执行层 terminate.reason
TERMINATE_REASON = {
    "user_stop": "user_stop",
    "agent_stop": "user_stop",  # 对话正常结束
    "max_steps": "max_turns",
    "too_many_errors": "infrastructure_error",
    "invalid_agent_message": "infrastructure_error",
}


def run_episode(
    agent: HarnessAgent,
    spec: EpisodeSpec,
    user: UserSimulator,
    environment: Environment,
    *,
    message_history: list[Message] | None = None,
    max_steps: int = 100,
    language: str | None = None,
    **conversation_options,
) -> ConversationResult:
    """对话结束(包括抛异常)后一定给执行层发 terminate;预取失败时不发,因为会话还没建立。

    ``conversation_options`` 原样交给 ``Conversation``(max_errors、greeting 等)。
    """
    agent.begin_episode(spec)
    reason = "infrastructure_error"  # 抛异常时的缺省
    try:
        with trace_scope(query_id=spec.episode_id, domain=spec.domain):
            result = Conversation(agent, user, environment, message_history=message_history, max_steps=max_steps, language=language, **conversation_options).run()
            trace = getattr(getattr(agent, "rt", None), "trace", None) or getattr(agent, "trace", None)
            if trace:
                trace.emit("conversation_end", termination_reason=str(result.termination_reason.value),
                           messages=[m.model_dump(mode="json") for m in result.messages])
        reason = TERMINATE_REASON.get(result.termination_reason.value, "cancelled")
        return result
    finally:
        agent.end_episode(reason)
