"""外层驱动:把执行层的 ``advance`` 包成"事件进,公开动作出"。与具体 benchmark 无关。

执行层的合同(见其设计文档)里,外层要做的事在这里:
* 给每个事件稳定的 ``event_id`` / ``base_rev``;网络或模型失败时用**完全相同的请求**重试;
* ``query_memory`` 不对外暴露:路由到 Memory 服务,再用 ``memory_result`` 回送,直到得到 ``call_tools`` / ``reply_user``;
* 回送 Memory 结果要幂等:benchmark 可能用同一状态重复调用我们(例如 VitaBench 的 orchestrator 在无效消息时会重试),
  所以同一个 ``request_id`` 的结果缓存下来,重放时不再请求网关,否则两次内容不同会触发 ``event_payload_conflict``。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from ..retry import RetryPolicy
from ..trace import TraceRecorder, trace_scope
from .loader import ensure_importable
from .model import NON_RETRYABLE_MODEL_CODES, usage_sink

ensure_importable()
from imact.errors import HarnessError  # noqa: E402
from imact.harness import ExecutionHarness  # noqa: E402
from imact.protocol import AdvanceOutput  # noqa: E402

logger = logging.getLogger(__name__)

# 重试也无法修复的错误:事件冲突 / 状态 / 配置类。其余(模型格式、证据、参数、校验、传输……)一律用相同请求重投:
# 执行层会把前次错误和原输出作为 repair_feedback 交给模型,并自己统计尝试次数,用尽后返回 blocked。
# 若是输入类错误,重投时执行层会返回 event_rejected,落在这个集合里,立即停止,不会空转。
NON_RETRYABLE_CODES = frozenset(
    {
        "event_in_progress", "event_not_owned", "event_payload_conflict", "event_rejected", "event_superseded",
        "evidence_not_found", "not_processing", "pending_event", "revision_conflict", "task_exists", "task_not_found",
        "invalid_config", "unsupported_domain", "unsupported_tool", "reference_unavailable", "external_schema_ref",
        "duplicate_tool", "context_overflow", "unauthorized",
    }
) | NON_RETRYABLE_MODEL_CODES

@dataclass
class MemoryRequest:
    """执行层发出 ``query_memory`` 时,交给路由函数的完整上下文。"""

    task_id: str
    request_id: str
    question: str
    metadata: Mapping[str, Any]


MemoryRouter = Callable[[MemoryRequest], Awaitable[dict]]


@dataclass
class PublicAction:
    """对 benchmark 可见的动作(已处理掉内部的 query_memory)。"""

    type: str  # call_tools | reply_user
    calls: list[dict] = field(default_factory=list)  # {call_id, name, arguments}
    content: str | None = None
    mode: str | None = None  # ask | inform | final
    outcome: str | None = None  # completed | partial | blocked
    rev: int = 0  # 最后一次提交后的 rev,即下一个事件的 base_rev
    event_id: str = ""  # 产生该动作的事件 id(reply_to 用)
    usage: dict[str, int] = field(default_factory=dict)
    memory_hops: int = 0


class ExecutionError(RuntimeError):
    pass


def event_id_for(base_rev: int, event_type: str = "") -> str:
    """event_id 完全由 (base_rev, 事件类型) 决定:同一状态的重放产生相同的请求。

    terminate 用独立前缀:前一个事件若失败(未提交),它占着 e{n},terminate 必须是**新**事件才能从暂存事实继续,
    复用 id 会触发 event_payload_conflict。"""
    return f"term{base_rev + 1}" if event_type == "terminate" else f"e{base_rev + 1}"


class ExecutionOuter:
    def __init__(
        self,
        harness: ExecutionHarness,
        *,
        retry: RetryPolicy | None = None,
        memory_router: MemoryRouter | None = None,
        max_memory_hops: int = 3,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        trace: TraceRecorder | None = None,
    ) -> None:
        self.harness = harness
        self.retry = retry or RetryPolicy(max_retries=3, backoff_s=1.0)
        self.memory_router = memory_router
        self.max_memory_hops = max_memory_hops
        self._sleep = sleep
        self._memory_cache: dict[tuple[str, str], dict] = {}
        self.trace = trace

    async def advance(self, task_id: str, base_rev: int, event: dict, metadata: Mapping[str, Any] | None = None) -> AdvanceOutput:
        """提交一个事件。可重试错误会用相同的请求重投,不重放任何业务工具。"""
        request = {
            "task_id": task_id,
            "event_id": event_id_for(base_rev, event.get("type", "")),
            "base_rev": base_rev,
            "event": event,
            "metadata": dict(metadata or {}),
        }
        attempt = 0
        while True:
            try:
                with trace_scope(task_id=task_id, event_id=request["event_id"], query_id=(metadata or {}).get("query_id")):
                    if self.trace:
                        self.trace.emit("harness_event", request=request, retry_index=attempt)
                    result = await self.harness.advance(request)
                    if self.trace:
                        self.trace.emit("harness_result", output=result.model_dump(mode="json"))
                        snapshot = self.harness.store.inspect(task_id)
                        workspace = snapshot["workspace"]
                        self.trace.emit(
                            "workspace_committed", rev=snapshot["rev"],
                            workspace={key: workspace.get(key) for key in (
                                "task", "task_sources", "notes", "progress", "orders",
                                "pending_calls", "pending_memory", "user_memory",
                                "failed_attempts", "last_reply", "lifecycle", "turns", "stop_reason",
                            )},
                            candidate_count=len(workspace.get("facts", {})),
                        )
                    return result
            except HarnessError as exc:
                if self.trace:
                    with trace_scope(task_id=task_id, event_id=request["event_id"]):
                        self.trace.emit("harness_error", code=exc.code, error=str(exc), retry_index=attempt)
                if exc.code in NON_RETRYABLE_CODES:
                    raise
                if attempt >= self.retry.max_retries:
                    raise
                delay = min(self.retry.max_backoff_s, self.retry.backoff_s * (self.retry.backoff_factor**attempt))
                logger.warning("advance %s/%s failed (%s); retrying in %.1fs", task_id, request["event_id"], exc.code, delay)
                attempt += 1
                await self._sleep(delay)

    async def drive(
        self, task_id: str, base_rev: int, event: dict, metadata: Mapping[str, Any] | None = None
    ) -> PublicAction:
        """提交事件,内部消化所有 query_memory,返回对外动作。"""
        usage: list[dict] = []
        token = usage_sink.set(usage)
        hops = 0
        try:
            rev, ev = base_rev, event
            while True:
                out = await self.advance(task_id, rev, ev, metadata)
                action = out.action.model_dump(exclude_none=True)
                if action["type"] != "query_memory":
                    return PublicAction(
                        type=action["type"],
                        calls=action.get("calls", []),
                        content=action.get("content"),
                        mode=action.get("mode"),
                        outcome=action.get("outcome"),
                        rev=out.rev,
                        event_id=out.event_id,
                        usage=_sum_usage(usage),
                        memory_hops=hops,
                    )
                hops += 1
                if hops > self.max_memory_hops:
                    raise ExecutionError(f"{task_id}: query_memory 超过 {self.max_memory_hops} 次仍未得到可执行动作")
                payload = await self._route_memory(task_id, action, metadata or {})
                rev, ev = out.rev, {"type": "memory_result", "request_id": action["request_id"], **payload}
        finally:
            usage_sink.reset(token)

    async def _route_memory(self, task_id: str, action: dict, metadata: Mapping[str, Any]) -> dict:
        key = (task_id, action["request_id"])
        if key in self._memory_cache:
            return self._memory_cache[key]
        if self.memory_router is None:
            payload = {"status": "error", "context": "", "error": "未配置 memory 服务"}
        else:
            try:
                payload = await self.memory_router(MemoryRequest(task_id, action["request_id"], action["question"], metadata))
            except Exception as exc:  # 路由函数自己的缺陷不能拖垮任务,按 memory 故障处理
                logger.exception("memory router failed")
                payload = {"status": "error", "context": "", "error": f"memory 路由异常: {type(exc).__name__}"}
        payload = {k: v for k, v in payload.items() if k in ("status", "context", "error")}
        if payload.get("status") == "error":
            payload.setdefault("error", "memory 服务故障")
            payload["context"] = ""
        self._memory_cache[key] = payload
        return payload

    def forget(self, task_id: str) -> None:
        for key in [k for k in self._memory_cache if k[0] == task_id]:
            del self._memory_cache[key]


def _sum_usage(items: list[dict]) -> dict[str, int]:
    return {
        "prompt_tokens": sum(int(u.get("prompt_tokens", 0) or 0) for u in items),
        "completion_tokens": sum(int(u.get("completion_tokens", 0) or 0) for u in items),
        "model_calls": len(items),
    }
