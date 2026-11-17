"""用户模拟器:LLM 扮演用户,与 benchmark 无关。

benchmark 只负责给出这一段对话的 system prompt 和开场白(例如 VitaBench 的 subtask 指令);
轮次状态、角色翻转、结束标记、调用模型都在这里。一个实例对应一段对话。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from ..trace import TraceRecorder
from .llm import LLMCaller
from .messages import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

STOP = "###STOP###"
TRANSFER = "###TRANSFER###"
OUT_OF_SCOPE = "###OUT-OF-SCOPE###"

ValidUserInputMessage = UserMessage | ToolMessage | MultiToolMessage


def is_valid_user_history_message(message: Message) -> bool:
    return (
        isinstance(message, UserMessage)
        or (isinstance(message, AssistantMessage) and not message.is_tool_call())
        or (isinstance(message, ToolMessage) and message.requestor == "user")
    )


class UserState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    opened: bool = False  # 开场白是否已经发出

    def flip_roles(self) -> list[APICompatibleMessage]:
        """把对话翻成用户模拟器视角:用户说的算 assistant,助手说的算 user。"""
        flipped: list[APICompatibleMessage] = []
        for message in self.messages:
            if isinstance(message, UserMessage):
                flipped.append(AssistantMessage(role="assistant", tool_calls=message.tool_calls, content=message.content))
            elif isinstance(message, AssistantMessage):
                if message.is_tool_call():
                    raise ValueError(f"Tool calls are not supported in the flipped messages: {message}")
                flipped.append(UserMessage(role="user", content=message.content))
            elif isinstance(message, ToolMessage):
                if message.requestor != "user":
                    raise ValueError(f"Tool messages should be sent to the user in this message history: {message}")
                flipped.append(ToolMessage(id=message.id, name=message.name, role=message.role, content=message.content))
            else:
                raise ValueError(f"Unknown message role: {message.role}")
        return flipped


class UserSimulator:
    def __init__(
        self,
        llm: LLMCaller,
        system_prompt: str,
        opening: str | None = None,
        *,
        tools: list[dict[str, Any]] | None = None,
        seed: int | None = None,
        trace: TraceRecorder | None = None,
    ):
        """``opening`` 是用户的第一句话,原样发出,不调用模型;为 None 时每一句(包括第一句)都由模型生成。

        ``tools`` 是用户自己能调用的工具(OpenAI function 格式),用于"助手让用户在自己的设备上操作"的场景。
        """
        self.llm = llm
        self.system_prompt = system_prompt
        self.opening = opening
        self.tools = tools or None
        self.seed = seed
        self.trace = trace

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> UserState:
        message_history = message_history or []
        if not all(is_valid_user_history_message(m) for m in message_history):
            raise ValueError("Invalid user message history.")
        return UserState(system_messages=[SystemMessage(role="system", content=self.system_prompt)], messages=message_history)

    @staticmethod
    def is_stop(message: UserMessage) -> bool:
        if message.is_tool_call():
            return False
        assert message.content is not None
        return STOP in message.content or TRANSFER in message.content or OUT_OF_SCOPE in message.content

    def generate_next_message(self, message: ValidUserInputMessage, state: UserState) -> tuple[UserMessage, UserState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        if not state.opened and self.opening is not None:
            state.opened = True
            reply = UserMessage(role="user", content=self.opening)
        else:
            state.opened = True
            response = self.llm.chat(state.system_messages + state.flip_roles(), tools=self.tools, seed=self.seed)
            reply = UserMessage(role="user", content=response.content, cost=response.cost, usage=response.usage, raw_data=response.raw_data)
            if response.tool_calls is not None:
                reply.tool_calls = [call.model_copy(update={"requestor": "user"}) for call in response.tool_calls]
        state.messages.append(reply)
        if self.trace:
            self.trace.emit("user_reply", message=reply.model_dump(mode="json"))
        return reply, state
