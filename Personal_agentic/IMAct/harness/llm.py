"""同步调用一个 ``ModelProvider``,供用户模拟器与裁判使用。

这两个角色和被测 Action 模型走同一套 ``models`` 层(协议、鉴权、请求参数都来自 ``models.yaml``),
不再经过任何 benchmark 自带的 OpenAI SDK 客户端。重试沿用 ``RetryPolicy``;空正文与未正常结束按错误处理。
"""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from typing import Any

from ..exec_layer.bridge import AsyncBridge
from ..models.base import ChatRequest
from ..models.config import ModelConfig, create_provider
from ..retry import CallError, RetryPolicy, run_with_retry
from ..trace import TraceRecorder
from .messages import APICompatibleMessage, AssistantMessage, ToolCall, to_chat_messages

# 只有 Chat Completions 系协议认 ``seed``;Responses / Anthropic Messages 收到未知字段会 400。
SEED_PROVIDERS = frozenset({"openai", "openai_compatible", "openrouter", "azure_openai"})


class LLMCaller:
    def __init__(
        self,
        cfg: ModelConfig,
        bridge: AsyncBridge,
        *,
        policy: RetryPolicy | None = None,
        trace: TraceRecorder | None = None,
        trace_stage: str | None = None,
    ):
        """``trace_stage`` 给定时,每次请求写 ``<stage>_request / _response / _error`` 轨迹(如裁判的 ``judge``)。"""
        self.cfg = cfg
        self.bridge = bridge
        self.provider = create_provider(cfg)
        self.policy = policy or RetryPolicy()
        if "bad_response" not in self.policy.retry_on:  # 空正文/被截断也值得重试(见 chat 里的说明)
            self.policy = dataclasses.replace(self.policy, retry_on=self.policy.retry_on | {"bad_response"})
        self.trace = trace
        self.trace_stage = trace_stage
        self.calls = 0

    def chat(
        self,
        messages: list[APICompatibleMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        seed: int | None = None,
        allow_incomplete: bool = False,
        **params: Any,
    ) -> AssistantMessage:
        """``allow_incomplete``:空正文 / 未正常结束仍会按 ``bad_response`` 重试,但重试用尽后返回最后一次的结果而不是抛错。
        给"模型答成什么就按什么评分"的 benchmark 用(例如 PersonaMem-v3:空答案计 0 分,而不是让整道题失败)。"""
        merged = dict(params)
        if seed is not None and self.cfg.provider in SEED_PROVIDERS:
            merged["seed"] = seed
        request = ChatRequest(messages=to_chat_messages(messages), tools=tools, params=self.cfg.params | merged)
        stage = self.trace_stage if self.trace else None
        last: list[Any] = []  # allow_incomplete 时保留最后一次不完整的回复

        async def once():
            trace_id = uuid.uuid4().hex
            started = time.perf_counter()
            if stage:
                self.trace.emit(f"{stage}_request", trace_id=trace_id, model=self.cfg.model, messages_before_provider=request.messages,
                                params=request.params, extra_body=self.cfg.extra_body)
            try:
                response = await self.provider.chat(request)
            except Exception as exc:
                if stage:
                    self.trace.emit(f"{stage}_error", trace_id=trace_id, error=str(exc), elapsed_ms=(time.perf_counter() - started) * 1000)
                raise
            if stage:
                self.trace.emit(f"{stage}_response", trace_id=trace_id, model=response.model or self.cfg.model,
                                request_id=(response.raw or {}).get("id"), content=response.content,
                                finish_reason=response.finish_reason, usage=response.usage,
                                elapsed_ms=(time.perf_counter() - started) * 1000)
            # 推理模型偶尔把 max_tokens 全花在推理上,正文为空或被截断。这是可以重来的采样结果,放在重试之内。
            last[:] = [response]
            if not response.content.strip() and not response.tool_calls:
                raise CallError("bad_response", f"模型返回空正文(finish_reason={response.finish_reason!r})")
            if response.finish_reason not in {"stop", "tool_calls"}:
                raise CallError("bad_response", f"模型未完整结束: {response.finish_reason!r}")
            return response

        self.calls += 1
        try:
            response = self.bridge.run(run_with_retry(once, self.policy))
        except CallError as exc:
            if not (allow_incomplete and exc.category == "bad_response" and last):
                raise
            response = last[0]
        calls = [
            ToolCall(id=tc.get("id", ""), name=(tc.get("function") or {}).get("name", ""),
                     arguments=json.loads((tc.get("function") or {}).get("arguments") or "{}"))
            for tc in response.tool_calls
        ] or None
        usage = {"prompt_tokens": response.usage.get("prompt_tokens", 0), "completion_tokens": response.usage.get("completion_tokens", 0)}
        return AssistantMessage(
            role="assistant", content=response.content, tool_calls=calls, cost=0.0, usage=usage,
            raw_data={
                "id": (response.raw or {}).get("id"), "model": response.model or self.cfg.model,
                "choices": [{"finish_reason": response.finish_reason, "message": {"role": "assistant", "content": response.content,
                                                                                "tool_calls": response.tool_calls or None}}],
                "usage": response.usage,
            },
        )

    def close(self) -> None:
        self.bridge.run(self.provider.aclose())
