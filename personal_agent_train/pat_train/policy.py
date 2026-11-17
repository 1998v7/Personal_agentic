"""被训练的策略模型:把 uni-agent Gateway 的会话地址包装成 IMAct 的 ``LLMCaller``。

和评测时的区别(都是为了让"模型采样到什么就按什么打分",不让 harness 替模型兜底):

* 不重试:Gateway 上的每次请求都会成为一条训练样本,重试等于偷偷换掉了模型的输出。传输错误直接抛出,
  这一整条样本被框架丢弃(基础设施问题,不是模型的错)。
* 空正文 / 未正常结束(如超出 Gateway 的长度预算,``finish_reason="length"``)原样返回,交给对话循环按无效消息处理。
* 工具调用的参数不是合法 JSON 时,按无效消息(既无正文也无工具调用)返回,而不是抛错丢样本。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from IMAct.exec_layer.bridge import AsyncBridge
from IMAct.harness.llm import LLMCaller
from IMAct.harness.messages import AssistantMessage
from IMAct.models.config import ModelConfig
from IMAct.retry import RetryPolicy

logger = logging.getLogger(__name__)

NO_RETRY = RetryPolicy(max_retries=0)


def policy_model_config(model: Any, *, max_tokens: int | None = None, timeout_s: float = 900.0) -> ModelConfig:
    """uni-agent 的 ``agent.model``(base_url / api_key / model_name)→ IMAct 的 ``ModelConfig``。"""
    if not getattr(model, "base_url", None):
        raise ValueError("agent.model.base_url 为空:策略模型的地址应由 uni-agent 在运行时填入")
    return ModelConfig(
        provider="openai_compatible",
        model=model.model_name or "policy",
        base_url=model.base_url,
        api_key=model.api_key or "EMPTY",
        timeout_s=timeout_s,
        params={"max_tokens": max_tokens} if max_tokens else {},
    )


class PolicyLLM(LLMCaller):
    def __init__(self, cfg: ModelConfig, bridge: AsyncBridge, **kwargs: Any):
        super().__init__(cfg, bridge, policy=NO_RETRY, **kwargs)

    def chat(self, messages, *, tools=None, seed=None, allow_incomplete: bool = True, **params: Any) -> AssistantMessage:
        try:
            return super().chat(messages, tools=tools, seed=None, allow_incomplete=allow_incomplete, **params)
        except json.JSONDecodeError as exc:
            logger.warning("policy produced tool call arguments that are not JSON: %s", exc)
            return AssistantMessage(role="assistant", content="", tool_calls=None, cost=0.0, usage={})
