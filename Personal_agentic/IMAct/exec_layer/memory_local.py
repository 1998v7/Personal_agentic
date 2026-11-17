"""将同步 MemoryClient 接入异步外层;一个工作线程复用一个客户端。"""

from __future__ import annotations

import asyncio
import contextvars
import importlib.util
import json
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from ..trace import TraceRecorder

logger = logging.getLogger("imact.memory")
DEFAULT_MAYA = Path(__file__).resolve().parents[3] / "IMAct_memory" / "personal_memory_maya"


class AsyncMemoryClient(Protocol):
    trace: TraceRecorder | None

    async def query(self, *, user_id: str, query_id: str, user_query: str, timestamp: Any) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


class LocalMemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_path: str = str(DEFAULT_MAYA / "examples" / "query_memory.py")
    config_path: str = str(DEFAULT_MAYA / "harness_config.json")
    bench: str = "vitabench"


def normalize_local_response(payload: Any) -> dict[str, Any]:
    """building/error/未知状态必须保留为故障,不能伪装成没有记忆。"""
    if not isinstance(payload, dict):
        return {"status": "error", "context": "", "error": "local memory returned a non-object response"}
    status, context, error = (payload.get(k) for k in ("status", "context", "error"))
    if error or not isinstance(status, str) or status not in {"ok", "empty"}:
        return {"status": "error", "context": "", "error": str(error or f"local memory status={status!r}")[:500]}
    if not isinstance(context, str) or (status == "empty" and context.strip()):
        return {"status": "error", "context": "", "error": "local memory returned an invalid context"}
    return {"status": "ok" if context.strip() else "empty", "context": context if context.strip() else "", "error": None}


def _create_client(cfg: LocalMemoryConfig):
    path = Path(cfg.client_path).expanduser().resolve()
    config = Path(cfg.config_path).expanduser().resolve()
    if not path.is_file() or not config.is_file():
        raise FileNotFoundError(f"检查 memory.local.client_path / config_path: {path}, {config}")
    # 上游使用绝对导入 model / memory_harness;防止静默复用另一个 checkout。
    service_dir = path.parent.parent if path.parent.name == "examples" else path.parent / "personal_memory_maya"
    for name in ("model", "memory_harness"):
        module = sys.modules.get(name)
        if module is not None:
            origin = getattr(module, "__file__", None)
            if origin is None or service_dir not in Path(origin).resolve().parents:
                raise RuntimeError(f"{name} 已从其他目录导入;一个进程只能使用一份 Memory checkout")
    spec = importlib.util.spec_from_file_location("_imact_local_memory_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MemoryClient(config_path=config)


class LocalMemoryClient:
    """不加整轮重试/超时:模型请求的超时重试由 Memory 原生配置管理。

    单线程队列保证即使调用方取消等待,正在执行的同步查询也不会与下一次
    查询或 close 重叠。关闭时排空已接受的任务,随后释放同一个客户端。
    """

    def __init__(self, cfg: LocalMemoryConfig, *, trace: TraceRecorder | None = None, client_factory=None):
        self.cfg = cfg
        self.trace = trace
        self._factory = client_factory or _create_client
        self._client = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="imact-local-memory")
        self._closed = False
        self._close_future = None

    def _get_client(self):
        if self._client is None:
            self._client = self._factory(self.cfg)
        return self._client

    def _submit(self, fn, *args, **kwargs):
        if self._closed:
            raise RuntimeError("LocalMemoryClient is closed")
        return asyncio.wrap_future(self._executor.submit(contextvars.copy_context().run, fn, *args, **kwargs))

    async def start(self) -> None:
        """仅加载代码、配置和任务索引,不调用模型。"""
        await self._submit(self._get_client)

    async def query(self, *, user_id: str, query_id: str, user_query: str, timestamp: Any) -> dict[str, Any]:
        trace_id = uuid.uuid4().hex
        task = {"query_id": query_id, "user_query": user_query, "timestamp": timestamp}
        started = time.perf_counter()
        if self.trace:
            self.trace.emit("memory_request", trace_id=trace_id, transport="local",
                            body={"bench": self.cfg.bench, "user_id": user_id, "task": task})
        try:
            payload = await self._submit(self._query_sync, user_id, task, trace_id)
            result = normalize_local_response(payload)
            if self.trace:
                self.trace.emit("memory_response", trace_id=trace_id, transport="local",
                                elapsed_ms=(time.perf_counter() - started) * 1000,
                                body=json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            result = {"status": "error", "context": "", "error": f"local memory {type(exc).__name__}: {exc}"[:500]}
            if self.trace:
                self.trace.emit("memory_error", trace_id=trace_id, transport="local", error=result["error"])
        logger.info("memory local query=%s status=%s elapsed=%.1fs context_chars=%d error=%s",
                    query_id, result["status"], time.perf_counter() - started, len(result["context"]), result["error"])
        return result

    def _query_sync(self, user_id, task, trace_id):
        return self._get_client().query(
            bench=self.cfg.bench, user_id=user_id, query_id=task["query_id"],
            timestamp=task["timestamp"], query=task["user_query"], trace_id=trace_id,
        )

    def _close_sync(self):
        if self._client is not None:
            self._client.close()

    async def aclose(self) -> None:
        if self._close_future is None:
            self._closed = True
            self._close_future = self._executor.submit(self._close_sync)
            # 已排队的查询和 close 仍会完成;不阻塞 asyncio 循环等待线程退出。
            self._executor.shutdown(wait=False)
        await asyncio.shield(asyncio.wrap_future(self._close_future))
