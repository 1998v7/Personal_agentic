"""把我们的 ModelProvider(OpenAI / OpenRouter / Azure / 任意兼容端点)接成执行层的 ``ActionModel``。

执行层自带的 ``ChatModel`` 写死了 ``max_tokens`` 且只认一种网关;这里按它公开的 ``ActionModel`` 协议
(``decide(context) -> dict | str | ModelResult``)注入我们的实现,不改它的代码。
行为与它的 ``ChatModel`` 对齐:一次请求、不在内部偷偷重试;失败抛 ``HarnessError`` 让外层用相同请求重试。
"""

from __future__ import annotations

import contextvars
from time import perf_counter
from typing import Any

from ..retry import CallError
from ..models.base import ChatRequest, ModelProvider
from ..trace import TraceRecorder
from .loader import ensure_importable

ensure_importable()
from imact.errors import HarnessError  # noqa: E402
from imact.model import ModelResponseError, ModelResult  # noqa: E402
from imact.store import dumps  # noqa: E402

# 每次 advance 内部可能调用多次模型(例如校验失败后的重问是外层重投,而不是内部循环);
# 用 ContextVar 把本次调用的 usage 收集起来,桥接层据此给 AssistantMessage 填 token / 成本。
usage_sink: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar("imact_usage_sink", default=None)

# 这些错误重试也不会好:配置 / 鉴权问题,应该立刻暴露而不是耗尽重试额度后返回 blocked。
NON_RETRYABLE_MODEL_CODES = frozenset({"model_auth_error"})


def to_harness_error(exc: CallError) -> HarnessError:
    if exc.category in ("transport", "timeout"):
        return HarnessError("model_transport_error", "模型请求失败,可使用相同事件重试", 502)
    if exc.category == "auth":
        return HarnessError("model_auth_error", f"模型鉴权失败(HTTP {exc.status or 401}),请检查密钥与 base_url", 502)
    if exc.category == "client":
        return HarnessError("model_http_error", f"模型服务 HTTP {exc.status or 400}: {exc.message[:200]}", 502)
    return HarnessError("model_http_error", f"模型服务 HTTP {exc.status or 500}", 502)


class ProviderActionModel:
    """``response_format="json_object"`` 需要网关支持;默认靠 prompt 约束输出联合 JSON。"""

    def __init__(self, provider: ModelProvider, *, params: dict[str, Any] | None = None, response_format: str = "text", trace: TraceRecorder | None = None) -> None:
        if response_format not in ("text", "json_object"):
            raise ValueError("response_format must be 'text' or 'json_object'")
        self.provider = provider
        self.params = dict(params or {})
        self.response_format = response_format
        self.model = getattr(provider, "model", None)  # 执行层把它记进 trace
        self.trace = trace

    async def decide(self, context: dict) -> ModelResult:
        messages = [{"role": "system", "content": context["system"]}]
        if context["task_policy"]:
            messages.append({"role": "system", "content": context["task_policy"]})
        messages.append({"role": "user", "content": dumps(context["input"])})
        params = dict(self.params)
        if self.response_format == "json_object":
            params["response_format"] = {"type": "json_object"}

        started = perf_counter()
        if self.trace:
            self.trace.emit("action_input", context=context, messages_before_provider=messages, params=params)
        try:
            resp = await self.provider.chat(ChatRequest(messages=messages, params=params))
        except CallError as exc:
            if self.trace:
                self.trace.emit("action_error", category=exc.category, status=exc.status, error=str(exc))
            raise to_harness_error(exc) from exc

        result = ModelResult(
            resp.content,
            {
                "model": resp.model or self.model,
                "request_id": (resp.raw or {}).get("id"),
                "finish_reason": resp.finish_reason,
                "usage": resp.usage or None,
                "latency_ms": round((perf_counter() - started) * 1000, 2),
                "request_parameters": params,
                "response_format": "native_tool_calls" if resp.tool_calls else "content",
            },
        )
        sink = usage_sink.get()
        if self.trace:
            self.trace.emit("action_output", content=resp.content, reasoning=resp.reasoning,
                            tool_calls=resp.tool_calls, metadata=result.metadata)
        if sink is not None and resp.usage:
            sink.append(dict(resp.usage))
        # 先判格式再判截断:返回了原生 tool_calls 时 finish_reason 也是 "tool_calls",此时真正的问题是格式
        if resp.tool_calls or not isinstance(resp.content, str):
            raise ModelResponseError("model_format", "模型需要在 content 返回联合 JSON", result)
        if resp.finish_reason not in (None, "stop"):
            raise ModelResponseError("model_incomplete", "模型输出未正常完成", result)
        return result
