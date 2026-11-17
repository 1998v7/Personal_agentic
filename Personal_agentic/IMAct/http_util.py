"""模型客户端与 Memory 客户端共用的 HTTP 错误映射。"""

from __future__ import annotations

import httpx

from .retry import CallError


def category_for_status(status: int) -> str:
    if status == 429:
        return "rate_limit"
    if status == 408:
        return "timeout"
    if status >= 500:
        return "server_error"
    if status in (401, 403):
        return "auth"
    return "client"


def retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def error_from_status(response: httpx.Response) -> CallError:
    body = response.text[:500]
    return CallError(
        category_for_status(response.status_code),
        f"HTTP {response.status_code}: {body}",
        status=response.status_code,
        retry_after=retry_after_seconds(response),
    )


def error_from_exception(exc: Exception) -> CallError:
    if isinstance(exc, httpx.TimeoutException):
        return CallError("timeout", f"{type(exc).__name__}: {exc}")
    if isinstance(exc, httpx.TransportError):
        return CallError("transport", f"{type(exc).__name__}: {exc}")
    return CallError("transport", f"{type(exc).__name__}: {exc}")
