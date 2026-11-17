"""Memory 网关(MPS)客户端。协议见 `memory接口.md`。

请求是 tensorFeatures 形式:每个字段是 ``{"shapes":[1],"stringValues":[<字符串>]}``,其中 task 是 JSON 字符串。
返回约定是 ``{"context": "..."}`` 或 ``{"error": "..."}``,实际网关在外层包了 MPS 信封:
成功时 ``{"success": true, "resultMap": {"context": ..., "status": "ok", "error": ""}}``,
网关侧 8s 超时时 ``{"success": false, "resultCode": "ERROR_REQUEST_TIMEOUT", "resultMap": {}}``(偶发,重试后通常成功)。
所以在整个响应里递归查找 ``context`` / ``error``,并把 ``success: false`` 当作可重试的调用错误。

任何失败都映射成 ``status="error"``,绝不当作 ``empty``(没有记忆和记忆服务挂了是两回事)。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import time
import uuid
from typing import Any, Awaitable, Callable

import httpx
from pydantic import BaseModel, ConfigDict

from ..retry import CallError, RetryPolicy, run_with_retry
from ..trace import TraceRecorder
from ..http_util import error_from_exception, error_from_status

logger = logging.getLogger("imact.memory")


class MpsMemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = ""  # 为空表示不启用 Memory
    bench: str = "vitabench"
    app_name: str = "imact"  # MPS-app-name
    http_version: str = "1.0"  # MPS-http-version
    api_key_env: str | None = None  # 网关需要鉴权时才配;文档示例里没有
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    timeout_s: float = 12.0  # 网关侧 8s 超时;客户端略大,才能收到网关返回的超时包
    max_concurrency: int = 16
    verify_tls: bool = True

    @property
    def enabled(self) -> bool:
        return bool(self.url)


def _sv(value: str) -> dict[str, Any]:
    return {"shapes": [1], "stringValues": [value]}


def build_body(action: str, bench: str, user_id: str, *, task: dict | None = None, data: Any = None) -> dict[str, Any]:
    feats: dict[str, Any] = {"action": _sv(action), "bench": _sv(bench), "user_id": _sv(str(user_id))}
    if task is not None:
        feats["task"] = _sv(json.dumps(task, ensure_ascii=False))
    if data is not None:  # build 的 data 编码方式文档未给出示例,按 task 同样的 JSON 字符串处理(未验证)
        feats["data"] = _sv(json.dumps(data, ensure_ascii=False))
    return {"features": {}, "tensorFeatures": feats}


def _find(obj: Any, key: str, depth: int = 0) -> Any:
    """在嵌套的 dict / list / JSON 字符串里找第一个非 None 的 key。"""
    if depth > 6:
        return None
    if isinstance(obj, str):
        s = obj.strip()
        if s[:1] in "{[":
            try:
                return _find(json.loads(s), key, depth + 1)
            except ValueError:
                return None
        return None
    if isinstance(obj, dict):
        if obj.get(key) is not None:
            return obj[key]
        for v in obj.values():
            found = _find(v, key, depth + 1)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find(v, key, depth + 1)
            if found is not None:
                return found
    return None


def parse_response(payload: Any) -> dict[str, Any]:
    """归一成执行层 ``MemoryPayload``:{status, context, error}。"""
    err = _find(payload, "error")
    if err:
        return {"status": "error", "context": "", "error": str(err)[:500]}
    ctx = _find(payload, "context")
    if isinstance(ctx, str):
        return {"status": "ok", "context": ctx, "error": None} if ctx.strip() else {"status": "empty", "context": "", "error": None}
    return {"status": "error", "context": "", "error": "memory 响应里没有 context 或 error 字段"}


def _log_result(action: str, trace: str, status: int, elapsed_ms: float, payload: Any) -> None:
    """context 全文按原样(真实换行)记录,便于直接阅读。"""
    parsed = parse_response(payload)
    head = f"memory {action} <- trace={trace} HTTP {status} in {elapsed_ms:.0f}ms status={parsed['status']}"
    if parsed["status"] == "ok":
        logger.info("%s context(%d chars):\n%s", head, len(parsed["context"]), parsed["context"])
    elif parsed["status"] == "empty":
        logger.info("%s (没有相关记忆)", head)
    else:
        logger.warning("%s error=%s raw=%s", head, parsed["error"], json.dumps(payload, ensure_ascii=False)[:500])


def curl_command(url: str, headers: dict[str, str], body: dict[str, Any], *, mask: tuple[str, ...] = ()) -> str:
    """可直接粘贴复现的 curl;``mask`` 里的头(鉴权)用 *** 代替。"""
    parts = ["curl", "-X", "POST"]
    for key, value in headers.items():
        parts += ["-H", f"{key}: {'***' if key.lower() in mask else value}"]
    parts += [url, "-d", json.dumps(body, ensure_ascii=False)]
    return " ".join(shlex.quote(p) for p in parts)


class MpsMemoryClient:
    def __init__(
        self,
        cfg: MpsMemoryConfig,
        *,
        retry: RetryPolicy | None = None,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        trace: TraceRecorder | None = None,
    ) -> None:
        self.cfg = cfg
        self.retry = retry or RetryPolicy(max_retries=1, backoff_s=0.5)
        self._key = api_key if api_key is not None else (os.environ.get(cfg.api_key_env) if cfg.api_key_env else None)
        if cfg.api_key_env and not self._key:
            raise ValueError(f"memory API key not found in environment variable {cfg.api_key_env!r}")
        self._sleep = sleep
        self._sem = asyncio.Semaphore(cfg.max_concurrency)
        self._client = client
        self._owns = client is None
        self.trace = trace

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.cfg.timeout_s), verify=self.cfg.verify_tls)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns:
            await self._client.aclose()
        self._client = None

    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json;charset=utf-8",
            "MPS-app-name": self.cfg.app_name,
            "MPS-http-version": self.cfg.http_version,
            "MPS-trace-id": uuid.uuid4().hex,
        }
        if self._key:
            h[self.cfg.auth_header] = f"{self.cfg.auth_scheme} {self._key}".strip()
        return h

    async def _post(self, body: dict[str, Any]) -> Any:
        async def once() -> Any:
            headers = self._headers()
            trace = headers["MPS-trace-id"]
            action = (body.get("tensorFeatures", {}).get("action", {}).get("stringValues") or ["?"])[0]
            logger.info(
                "memory %s -> request trace=%s\n%s", action, trace,
                curl_command(self.cfg.url, headers, body, mask=(self.cfg.auth_header.lower(),)),
            )
            started = time.perf_counter()
            if self.trace:
                self.trace.emit("memory_request", trace_id=trace, url=self.cfg.url, body=body)
            async with self._sem:
                try:
                    resp = await self._http().post(self.cfg.url, json=body, headers=headers)
                except httpx.HTTPError as exc:
                    if self.trace:
                        self.trace.emit("memory_error", trace_id=trace, error=str(exc))
                    logger.warning("memory %s <- trace=%s transport error after %.0fms: %s", action, trace, (time.perf_counter() - started) * 1000, exc)
                    raise error_from_exception(exc) from exc
            elapsed = (time.perf_counter() - started) * 1000
            if self.trace:
                self.trace.emit("memory_response", trace_id=trace, status_code=resp.status_code,
                                elapsed_ms=elapsed, body=resp.text)
            if resp.status_code >= 400:
                logger.warning("memory %s <- trace=%s HTTP %s in %.0fms: %s", action, trace, resp.status_code, elapsed, resp.text[:500])
                raise error_from_status(resp)
            try:
                payload = resp.json()
            except ValueError as exc:
                logger.warning("memory %s <- trace=%s HTTP %s in %.0fms, not JSON: %s", action, trace, resp.status_code, elapsed, resp.text[:500])
                raise CallError("bad_response", "memory 响应不是 JSON") from exc
            if isinstance(payload, dict) and payload.get("success") is False:
                code = str(payload.get("resultCode") or "")
                detail = str(payload.get("errorMessage") or code or "success=false")[:200]
                logger.warning("memory %s <- trace=%s gateway failure in %.0fms: %s", action, trace, elapsed, json.dumps(payload, ensure_ascii=False)[:500])
                raise CallError("timeout" if "TIMEOUT" in (code + detail).upper() else "server_error", detail)
            _log_result(action, trace, resp.status_code, elapsed, payload)
            return payload

        return await run_with_retry(once, self.retry, level="memory", sleep=self._sleep)

    async def query(self, *, user_id: str, query_id: str, user_query: str, timestamp: Any) -> dict[str, Any]:
        """返回执行层 MemoryPayload 形态的 dict。不抛异常。"""
        body = build_body(
            "query", self.cfg.bench, user_id, task={"query_id": query_id, "user_query": user_query, "timestamp": timestamp}
        )
        try:
            return parse_response(await self._post(body))
        except CallError as exc:
            return {"status": "error", "context": "", "error": f"memory 网关{exc.category}错误: {exc.message[:200]}"}

    async def build(self, *, user_id: str, data: Any) -> dict[str, Any]:
        """一次交付该用户的历史。网关目前只能测 query,build 尚未联调。"""
        try:
            payload = await self._post(build_body("build", self.cfg.bench, user_id, data=data))
        except CallError as exc:
            return {"status": "error", "error": f"memory 网关{exc.category}错误: {exc.message[:200]}"}
        err = _find(payload, "error")
        if err:
            return {"status": "error", "error": str(err)[:500]}
        return {"status": "complete"} if _find(payload, "status") == "complete" else {"status": "error", "error": "build 未返回 complete"}
