"""外层 Harness 唯一需要依赖的事件协议。模型输出另由领域 JSON Schema 约束。"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class MemoryPayload(StrictModel):
    status: Literal["ok", "empty", "error", "not_requested"] = "not_requested"
    context: str = ""
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def service_payload(cls, value):
        if isinstance(value, dict) and "status" not in value:
            if value.get("error"):
                return {**value, "status": "error"}
            if isinstance(value.get("context"), str):
                nonempty = bool(value["context"].strip())
                return {
                    **value,
                    "status": "ok" if nonempty else "empty",
                    "context": value["context"] if nonempty else "",
                }
        return value

    @model_validator(mode="after")
    def consistent(self):
        if self.status == "ok" and not self.context.strip():
            raise ValueError("ok memory requires nonempty context")
        if self.status != "ok" and self.context:
            raise ValueError("non-ok memory requires empty context")
        if self.status == "error" and not self.error:
            raise ValueError("error memory requires error description")
        if self.status != "error" and self.error is not None:
            raise ValueError("non-error memory cannot include error description")
        return self


class ToolFunction(StrictModel):
    name: Text
    description: str = ""
    parameters: dict[str, Any]
    strict: bool | None = None


class ToolDefinition(StrictModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class StartEvent(StrictModel):
    type: Literal["start"]
    query: Text
    domain: str = "delivery"
    system_prompt: str | None = None
    system_ref: str | None = None
    tools: list[ToolDefinition] | None = None
    tools_ref: str | None = None
    history: str | None = None
    history_ref: str | None = None
    memory: MemoryPayload = Field(default_factory=MemoryPayload)
    memory_mode: Literal["external", "benchmark"] = "external"
    max_turns: Annotated[int, Field(ge=1, le=10000)] = 100

    @model_validator(mode="after")
    def exclusive_refs(self):
        for direct, ref in [
            ("system_prompt", "system_ref"),
            ("tools", "tools_ref"),
            ("history", "history_ref"),
        ]:
            if getattr(self, direct) is not None and getattr(self, ref) is not None:
                raise ValueError(f"provide either {direct} or {ref}")
        if self.tools is None and self.tools_ref is None:
            raise ValueError("tools or tools_ref required; use [] for no tools")
        return self


class UserMessageEvent(StrictModel):
    type: Literal["user_message"]
    content: Text
    reply_to: str | None = None
    # 可选：外层已确认的撤销字段。首版不靠关键词猜测用户是否明确撤销。
    allowed_clears: list[str] = Field(default_factory=list)


class ToolResult(StrictModel):
    call_id: Text
    name: Text
    delivery: Literal["received", "unknown"]
    content: Any = None

    @model_validator(mode="after")
    def received_has_content(self):
        if self.delivery == "received" and self.content is None:
            raise ValueError("received result requires content")
        return self


class ToolResultsEvent(StrictModel):
    type: Literal["tool_results"]
    results: Annotated[list[ToolResult], Field(min_length=1)]


class MemoryResultEvent(MemoryPayload):
    type: Literal["memory_result"]
    request_id: Text
    status: Literal["ok", "empty", "error"]


class TerminateEvent(StrictModel):
    type: Literal["terminate"]
    reason: Literal["cancelled", "user_stop", "max_turns", "infrastructure_error"] = "user_stop"


class ResumeEvent(StrictModel):
    type: Literal["resume"]
    additional_turns: Annotated[int, Field(ge=1, le=10000)] = 100


Event = Annotated[
    StartEvent
    | UserMessageEvent
    | ToolResultsEvent
    | MemoryResultEvent
    | TerminateEvent
    | ResumeEvent,
    Field(discriminator="type"),
]


class AdvanceInput(StrictModel):
    task_id: Text
    event_id: Text
    base_rev: Annotated[int, Field(ge=0)]
    event: Event
    # 外层路由扩展区；持久化但不会自动拼入模型上下文。
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutgoingCall(StrictModel):
    call_id: Text
    name: Text
    arguments: dict[str, Any]


class ToolAction(StrictModel):
    type: Literal["call_tools"]
    calls: list[OutgoingCall]


class MemoryAction(StrictModel):
    type: Literal["query_memory"]
    request_id: Text
    question: Text


class ReplyAction(StrictModel):
    type: Literal["reply_user"]
    mode: Literal["ask", "inform", "final"]
    content: Text
    outcome: Literal["completed", "partial", "blocked"] | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class AdvanceOutput(StrictModel):
    task_id: str
    event_id: str
    rev: int
    action: Annotated[ToolAction | MemoryAction | ReplyAction, Field(discriminator="type")]
