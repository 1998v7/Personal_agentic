"""按配置装配:执行层 + 我们的模型 + 本地/网关 Memory + 同步桥。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..models.base import ModelProvider
from ..models.config import create_provider
from ..trace import TraceRecorder, trace_scope
from .bridge import AsyncBridge
from .config import OuterConfig
from .loader import ensure_importable
from .memory_mps import MpsMemoryClient
from .memory_local import AsyncMemoryClient, LocalMemoryClient
from .model import ProviderActionModel
from .outer import ExecutionOuter, MemoryRequest

logger = logging.getLogger(__name__)


@dataclass
class ExecRuntime:
    """一个进程里共享的运行时:所有任务共用一个事件循环线程和一个执行层实例。"""

    cfg: OuterConfig
    bridge: AsyncBridge
    outer: ExecutionOuter
    memory: AsyncMemoryClient | None
    provider: ModelProvider
    trace: TraceRecorder | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        async def _close() -> None:
            try:
                await self.provider.aclose()
            finally:
                if self.memory is not None:
                    await self.memory.aclose()

        try:
            self.bridge.run(_close())  # 等本地同步查询排空,不能 10 秒后直接销毁事件循环
        except Exception:  # 关闭阶段的清理失败不应掩盖真正的结果
            logger.exception("error while closing runtime")
        self.bridge.close()


def build_runtime(
    cfg: OuterConfig,
    *,
    provider: ModelProvider | None = None,
    memory_client: AsyncMemoryClient | None = None,
    sleep=None,
    trace: TraceRecorder | None = None,
) -> ExecRuntime:
    """provider / memory_client 可注入(测试或自定义实现);缺省时按配置创建。"""
    ensure_importable(cfg.exec.src)
    from imact.context import ContextAssembler
    from imact.harness import ExecutionHarness
    from imact.store import SQLiteStore

    provider = provider or create_provider(cfg.model)
    if memory_client is None and cfg.memory.backend == "local":
        memory_client = LocalMemoryClient(cfg.memory.local, trace=trace)
    elif memory_client is None and cfg.memory.backend == "mps" and cfg.memory.gateway.enabled:
        memory_client = MpsMemoryClient(cfg.memory.gateway, retry=cfg.memory.retry.to_policy(), trace=trace)
    elif memory_client is not None and trace is not None:
        memory_client.trace = trace

    model_retry = cfg.retry.model_call
    timeout = cfg.model.timeout_s
    decision_timeout = cfg.exec.decision_timeout_s or max(120.0, timeout + 30.0)
    if decision_timeout <= timeout:
        raise ValueError("exec.decision_timeout_s 必须大于 model.timeout_s,否则模型超时前决策就先超时了")

    db_path = Path(cfg.exec.db_path)
    bridge = AsyncBridge()

    def make_harness() -> ExecutionHarness:
        return ExecutionHarness(
            SQLiteStore(db_path),
            ProviderActionModel(provider, params=cfg.model.params | cfg.model_call.params, response_format=cfg.model_call.response_format, trace=trace),
            decision_timeout=decision_timeout,
            context=ContextAssembler(cfg.exec.context_max_bytes),
            # 首次 + N 次重试 = N+1 次尝试;用尽后执行层返回 blocked,而不是让外层无限重试
            max_decision_attempts=model_retry.max_retries + 1,
        )

    harness = bridge.call(make_harness)
    router = _memory_router(memory_client, cfg) if memory_client is not None else None
    kwargs = {"sleep": sleep} if sleep is not None else {}
    outer = ExecutionOuter(
        harness, retry=model_retry.to_policy(), memory_router=router, max_memory_hops=cfg.max_memory_hops, trace=trace, **kwargs
    )
    return ExecRuntime(cfg, bridge, outer, memory_client, provider, trace)


def _memory_router(client: AsyncMemoryClient, cfg: OuterConfig):
    """执行层的 query_memory(request_id, question) → M 服务的 query({query_id, user_query, timestamp})。

    ``question`` 如何映射到 ``user_query`` 在 M 团队那边尚未对齐,所以做成配置:
    默认传模型提出的问题;设成 ``instruction`` 则沿用任务指令(与预取一致)。
    """

    async def route(req: MemoryRequest) -> dict:
        md = req.metadata
        user_query = md.get("instruction", req.question) if cfg.memory.active_query_text == "instruction" else req.question
        with trace_scope(task_id=req.task_id, query_id=md["query_id"], memory_request_id=req.request_id, phase="active_query"):
            return await client.query(
                user_id=md["user_id"], query_id=md["query_id"], user_query=user_query, timestamp=md.get("timestamp")
            )

    return route
