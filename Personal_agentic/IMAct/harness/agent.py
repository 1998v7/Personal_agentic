"""Agent:把对话里的消息翻译成执行层事件,再把执行层的动作翻译回消息。与 benchmark 无关。

一个 episode(VitaBench 里的一个 subtask)对应执行层的一个 task:
``begin_episode`` 预取记忆并登记会话,``generate_next_message`` 逐条推进,``end_episode`` 发 terminate。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel

from ..exec_layer import ExecRuntime, PublicAction
from ..trace import trace_scope
from .messages import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from .protocols import EpisodeSpec

logger = logging.getLogger(__name__)

ValidAgentInputMessage = UserMessage | ToolMessage | MultiToolMessage
STOP_TOKEN = "###STOP###"


def is_valid_agent_history_message(message: Message) -> bool:
    return (
        isinstance(message, AssistantMessage)
        or (isinstance(message, UserMessage) and not message.is_tool_call())
        or (isinstance(message, ToolMessage) and message.requestor == "assistant")
    )


def is_agent_stop(message: AssistantMessage) -> bool:
    return message.content is not None and STOP_TOKEN in message.content


class AgentState(BaseModel):
    """会话身份放在 state 而不是 agent 对象上。

    对话循环在 agent 返回无效消息时,会用**深拷贝的原状态**重试(最多 3 次)。
    把 task_id / rev 放进 state,重试就会发出完全相同的请求,命中执行层的"原样重放"缓存,不会重复决策或重复调工具。
    """

    system_messages: list[SystemMessage] = []
    messages: list[APICompatibleMessage] = []
    harness_task_id: Optional[str] = None
    harness_rev: int = 0
    harness_pending: dict[str, str] = {}  # call_id -> 工具名,用来给回执补 name
    harness_last_reply_event: Optional[str] = None  # 上一个动作是 reply_user 时,下一条用户消息的 reply_to


@dataclass
class EpisodeContext:
    task_id: str
    spec: EpisodeSpec
    memory: dict[str, Any]  # 预取结果,start 事件里原样携带(重放时必须一致)
    metadata: dict[str, Any] = field(default_factory=dict)
    rev: Optional[int] = None  # 已提交的最新 rev,end_episode 用


class HarnessAgent:
    def __init__(self, *, runtime: ExecRuntime, run_id: str | None = None):
        self.rt = runtime
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self._ctx: EpisodeContext | None = None

    # ------------------------------------------------------------------ episode 生命周期
    def begin_episode(self, spec: EpisodeSpec) -> EpisodeContext:
        task_id = f"{self.run_id}-{spec.user_id}-{spec.episode_id}-{uuid.uuid4().hex[:8]}"
        metadata = {
            "bench": spec.bench,
            "user_id": spec.user_id,
            "query_id": spec.episode_id,
            "instruction": spec.instruction,
            "timestamp": spec.timestamp,
            **spec.metadata,
        }
        with trace_scope(task_id=task_id, query_id=spec.episode_id, phase="prefetch"):
            memory = self._prefetch(metadata)
            if self.rt.trace:
                self.rt.trace.emit("memory_normalized", payload=memory, metadata=metadata)
            if self.rt.cfg.memory.backend != "none" and self.rt.cfg.memory.require_prefetch and memory.get("status") != "ok":
                raise RuntimeError("首轮 Memory 预取未成功,require_prefetch 阻止调用 Action")
            self._ctx = EpisodeContext(task_id, spec, memory=memory, metadata=metadata)
        return self._ctx

    def _prefetch(self, md: dict[str, Any]) -> dict[str, Any]:
        cfg = self.rt.cfg.memory
        if not cfg.prefetch or self.rt.memory is None:
            return {"status": "not_requested"}
        payload = self.rt.bridge.run(
            self.rt.memory.query(user_id=md["user_id"], query_id=md["query_id"], user_query=md["instruction"], timestamp=md["timestamp"])
        )
        if payload["status"] == "error":
            logger.warning("memory prefetch failed for %s: %s", md["query_id"], payload.get("error"))
        return payload

    def end_episode(self, reason: str = "user_stop") -> None:
        """对话结束后通知执行层。循环不会告诉 agent 对话何时结束,所以由 ``run_episode`` 调用。"""
        ctx, self._ctx = self._ctx, None
        if ctx is None or ctx.rev is None:
            return
        try:
            self.rt.bridge.run(self.rt.outer.advance(ctx.task_id, ctx.rev, {"type": "terminate", "reason": reason}, ctx.metadata))
        except Exception:  # 收尾失败不应让已经跑完的 episode 作废
            logger.exception("terminate failed for %s", ctx.task_id)
        finally:
            self.rt.outer.forget(ctx.task_id)

    # ------------------------------------------------------------------ 对话循环使用的接口
    def get_init_state(self, message_history: Optional[list[Message]] = None) -> AgentState:
        if self._ctx is None:
            raise RuntimeError("begin_episode() 必须在 get_init_state() 之前调用(请使用 run_episode)")
        return AgentState(
            system_messages=[SystemMessage(role="system", content=self._ctx.spec.system_prompt)],
            messages=list(message_history or []),
            harness_task_id=self._ctx.task_id,
        )

    def generate_next_message(self, message: ValidAgentInputMessage, state: AgentState) -> tuple[AssistantMessage, AgentState]:
        ctx = self._ctx
        if ctx is None or state.harness_task_id != ctx.task_id:
            raise RuntimeError("agent 状态不属于当前 episode")
        event = self._to_event(message, state, ctx)
        incoming = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        if self.rt.trace:
            with trace_scope(task_id=ctx.task_id, query_id=ctx.spec.episode_id):
                self.rt.trace.emit("benchmark_feedback", messages=[m.model_dump(mode="json") for m in incoming])
        state.messages.extend(incoming)

        action = self.rt.bridge.run(self.rt.outer.drive(ctx.task_id, state.harness_rev, event, ctx.metadata))
        reply = self._to_assistant(action, state)

        state.harness_rev = action.rev
        ctx.rev = max(ctx.rev or 0, action.rev)
        state.harness_last_reply_event = action.event_id if action.type == "reply_user" else None
        state.messages.append(reply)
        return reply, state

    # ------------------------------------------------------------------ 消息 → 执行层事件
    def _to_event(self, message: ValidAgentInputMessage, state: AgentState, ctx: EpisodeContext) -> dict[str, Any]:
        if isinstance(message, (ToolMessage, MultiToolMessage)):
            tool_messages = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
            results = []
            for tm in tool_messages:
                results.append(
                    {
                        "call_id": tm.id,
                        "name": state.harness_pending.get(tm.id) or tm.name,
                        "delivery": "received",  # 业务报错也是已收到的回执
                        "content": tm.content if tm.content is not None else "",
                    }
                )
                state.harness_pending.pop(tm.id, None)
            return {"type": "tool_results", "results": results}

        if not isinstance(message, UserMessage):
            raise TypeError(f"unexpected message type {type(message).__name__}")
        text = message.content if (message.content or "").strip() else "(空消息)"
        if state.harness_rev == 0:  # 还没有提交过任何事件:这是 episode 的第一条用户消息
            event: dict[str, Any] = {
                "type": "start",
                "domain": ctx.spec.domain,
                "query": text,
                "system_prompt": ctx.spec.system_prompt,
                "tools": ctx.spec.tools,
                "memory": ctx.memory,
                "memory_mode": "external",
                "max_turns": self.rt.cfg.max_turns,
            }
            history = _render_history(state.messages)  # 当前消息此时还没追加进 state,里面只有预置历史
            if history:
                event["history"] = history
            return event
        event = {"type": "user_message", "content": text}
        if state.harness_last_reply_event:
            event["reply_to"] = state.harness_last_reply_event
        return event

    # ------------------------------------------------------------------ 执行层动作 → 消息
    def _to_assistant(self, action: PublicAction, state: AgentState) -> AssistantMessage:
        usage = {"prompt_tokens": action.usage.get("prompt_tokens", 0), "completion_tokens": action.usage.get("completion_tokens", 0)}
        common = {
            "cost": 0.0,  # 必须是数值:成本统计遇到 None 会整体放弃
            "usage": usage,
            "raw_data": {
                "imact": {
                    "task_id": state.harness_task_id, "event_id": action.event_id, "rev": action.rev,
                    "memory_hops": action.memory_hops, "model_calls": action.usage.get("model_calls", 0),
                    "mode": action.mode, "outcome": action.outcome,
                }
            },
        }
        if action.type == "call_tools":
            if not action.calls:
                return AssistantMessage(role="assistant", content=None, tool_calls=None, **common)  # 交给对话循环的无效消息重试
            for c in action.calls:
                state.harness_pending[c["call_id"]] = c["name"]
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(id=c["call_id"], name=c["name"], arguments=c["arguments"], requestor="assistant") for c in action.calls],
                **common,
            )
        return AssistantMessage(role="assistant", content=action.content, **common)


def _render_history(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        if isinstance(m, UserMessage) and m.has_text_content():
            lines.append(f"用户: {m.content}")
        elif isinstance(m, AssistantMessage) and m.has_text_content():
            lines.append(f"助手: {m.content}")
    return "\n".join(lines)
