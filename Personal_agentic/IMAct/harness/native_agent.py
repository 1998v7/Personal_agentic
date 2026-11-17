"""原生 function-calling agent:不经过执行层,模型直接看到 policy、历史和工具定义,一次请求产出下一条消息。

对应上游 τ² 的 ``LLMAgent``:system prompt 由 benchmark 给出,历史就是对话消息,工具用模型原生的 tool calling。
这样测到的是 agent 模型本身,分数可以和 benchmark 公布的榜单比较。与 ``HarnessAgent``(对接 IMAct 执行层)并列,
对话循环用同一套接口驱动二者。
"""

from __future__ import annotations

from typing import Any, Optional

from ..trace import trace_scope
from .agent import AgentState, ValidAgentInputMessage, is_valid_agent_history_message
from .llm import LLMCaller
from .messages import AssistantMessage, Message, MultiToolMessage, SystemMessage


class NativeAgent:
    def __init__(self, llm: LLMCaller, system_prompt: str, tools: list[dict[str, Any]] | None = None, *, seed: int | None = None, trace=None):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or None
        self.seed = seed
        self.trace = trace  # episode 结束时写 conversation_end;没有执行层运行时(rt)可借

    # ------------------------------------------------------------------ episode 生命周期(无会话可管,留空)
    def begin_episode(self, spec) -> None:
        return None

    def end_episode(self, reason: str = "user_stop") -> None:
        return None

    # ------------------------------------------------------------------ 对话循环使用的接口
    def get_init_state(self, message_history: Optional[list[Message]] = None) -> AgentState:
        history = list(message_history or [])
        if not all(is_valid_agent_history_message(m) for m in history):
            raise ValueError("Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent.")
        return AgentState(system_messages=[SystemMessage(role="system", content=self.system_prompt)], messages=history)

    def generate_next_message(self, message: ValidAgentInputMessage, state: AgentState) -> tuple[AssistantMessage, AgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        reply = self.llm.chat(state.system_messages + state.messages, tools=self.tools, seed=self.seed)
        state.messages.append(reply)
        return reply, state
