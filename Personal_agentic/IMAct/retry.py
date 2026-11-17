"""统一的重试策略(方案 §5.5)。三层重试都用同一个 RetryPolicy 描述。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 错误类别:transport / timeout / rate_limit / server_error / auth / client / bad_response
DEFAULT_RETRY_ON = frozenset({"transport", "timeout", "rate_limit", "server_error"})


class CallError(Exception):
    """一次外部调用(模型、Memory)失败。category 决定是否可重试。"""

    def __init__(self, category: str, message: str, *, status: int | None = None, retry_after: float | None = None):
        super().__init__(f"[{category}] {message}")
        self.category = category
        self.message = message
        self.status = status
        self.retry_after = retry_after


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3  # 首次失败后最多再试几次(不含首次)
    backoff_s: float = 2.0
    backoff_factor: float = 2.0
    max_backoff_s: float = 30.0
    retry_on: frozenset[str] = DEFAULT_RETRY_ON

    def should_retry(self, error: CallError, attempt: int) -> float | None:
        """返回等待秒数则重试,返回 None 则放弃。子类可覆盖实现自定义策略。attempt 从 0 开始。"""
        if attempt >= self.max_retries or error.category not in self.retry_on:
            return None
        delay = min(self.max_backoff_s, self.backoff_s * (self.backoff_factor**attempt))
        if error.retry_after is not None:
            delay = min(self.max_backoff_s, max(delay, error.retry_after))
        return delay


OnRetry = Callable[[str, int, CallError, float], None]


def _log_retry(level: str, attempt: int, error: CallError, delay: float) -> None:
    logger.warning("retry[%s] #%d after %s; sleeping %.1fs", level, attempt, error, delay)


async def run_with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    level: str = "model_call",
    on_retry: OnRetry | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    attempt = 0
    while True:
        try:
            return await fn()
        except CallError as exc:
            delay = policy.should_retry(exc, attempt)
            if delay is None:
                raise
            (on_retry or _log_retry)(level, attempt + 1, exc, delay)
            await sleep(delay)
            attempt += 1
