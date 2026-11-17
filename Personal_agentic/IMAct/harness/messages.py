"""对话消息模型。字段与 benchmark 无关;保持与历史 trajectory JSON 同形,旧的 simulation.json 仍可读取。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

ToolRequestor = Literal["user", "assistant"]


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class SystemMessage(BaseModel):
    role: Literal["system"]
    content: Optional[str] = None
    turn_idx: Optional[int] = None
    timestamp: Optional[str] = Field(default_factory=now_str)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SystemMessage) and self.content == other.content


class ToolCall(BaseModel):
    id: str = ""
    name: str
    arguments: dict
    requestor: ToolRequestor = "assistant"

    def __str__(self) -> str:
        return f"ToolCall(from {self.requestor}) {self.name}({json.dumps(self.arguments, ensure_ascii=False)})"


class ParticipantMessageBase(BaseModel):
    """用户或助手的一条消息:要么有文本,要么有 tool_calls。"""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    turn_idx: Optional[int] = None
    timestamp: Optional[str] = Field(default_factory=now_str)
    cost: Optional[float] = None
    usage: Optional[dict] = None
    raw_data: Optional[dict] = None

    def validate(self):  # noqa: D401 - 沿用历史接口(实例方法,非 pydantic 的 classmethod)
        if not (self.has_text_content() or self.is_tool_call()):
            raise ValueError(f"{self.role} message must have either content or tool calls. Got {self}")

    def has_text_content(self) -> bool:
        return isinstance(self.content, str) and self.content.strip() != ""

    def is_tool_call(self) -> bool:
        return self.tool_calls is not None

    def __str__(self) -> str:
        lines = [f"{self.role.capitalize()}Message"]
        if self.content is not None:
            lines.append(f"content: {self.content}")
        if self.tool_calls:
            lines.extend(str(tc) for tc in self.tool_calls)
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and (self.role, self.content, self.tool_calls) == (other.role, other.content, other.tool_calls)


class AssistantMessage(ParticipantMessageBase):
    role: Literal["assistant"]


class UserMessage(ParticipantMessageBase):
    role: Literal["user"]


class ToolMessage(BaseModel):
    id: str
    name: Optional[str] = None  # 工具名;有些 benchmark 的回执不带
    role: Literal["tool"]
    content: Optional[str] = None
    requestor: ToolRequestor = "assistant"
    error: bool = False
    execution_status: Literal["executed", "rejected"] = "executed"  # rejected:对话循环在执行前就拒绝了这次调用

    @model_validator(mode="after")
    def _rejected_is_an_error(self):
        if self.execution_status == "rejected" and not self.error:
            raise ValueError("A rejected tool call must have error=True")
        return self
    turn_idx: Optional[int] = None
    timestamp: Optional[str] = Field(default_factory=now_str)

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and (self.id, self.content, self.requestor, self.error) == (other.id, other.content, other.requestor, other.error)


class MultiToolMessage(BaseModel):
    role: Literal["tool"]
    tool_messages: list[ToolMessage]


APICompatibleMessage = SystemMessage | AssistantMessage | UserMessage | ToolMessage
Message = SystemMessage | AssistantMessage | UserMessage | ToolMessage | MultiToolMessage


def to_chat_messages(messages: list[APICompatibleMessage]) -> list[dict]:
    """转成 Chat Completions 形态的 dict,交给 ``models`` 层的 provider。"""
    out: list[dict] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AssistantMessage):
            entry: dict = {"role": "assistant", "content": m.content}
            if m.is_tool_call():
                entry["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                    for tc in m.tool_calls
                ]
            out.append(entry)
        elif isinstance(m, ToolMessage):
            entry = {"role": "tool", "content": m.content, "tool_call_id": m.id}
            if m.name:
                entry["name"] = m.name
            out.append(entry)
    return out
