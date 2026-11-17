"""可替换的模型接口：普通事件一次请求；不在 SDK 内偷偷重试。"""

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import httpx2 as httpx

from .errors import HarnessError
from .store import dumps


@dataclass
class ModelResult:
    content: dict | str
    metadata: dict


class ModelResponseError(HarnessError):
    """仅在内部 trace 保存模型输出，不把供应商错误页返回调用方。"""

    def __init__(self, code, message, result):
        super().__init__(code, message, 502)
        self.result = result


class ActionModel(Protocol):
    async def decide(self, context: dict) -> dict | str | ModelResult: ...


class ChatModel:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 90,
        max_output_tokens: int = 4096,
        response_format: str = "text",
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.timeout, self.max_output_tokens = timeout, max_output_tokens
        self.response_format, self.transport = response_format, transport
        if response_format not in {"text", "json_object"}:
            raise ValueError("response_format must be text or json_object")

    @classmethod
    def from_env(cls):
        base, name = os.getenv("IMACT_MODEL_BASE_URL"), os.getenv("IMACT_MODEL_NAME")
        if not base or not name:
            raise HarnessError(
                "model_not_configured", "配置 IMACT_MODEL_BASE_URL 和 IMACT_MODEL_NAME", 503
            )
        return cls(
            base,
            name,
            os.getenv("IMACT_MODEL_API_KEY", ""),
            float(os.getenv("IMACT_MODEL_TIMEOUT", "90")),
            int(os.getenv("IMACT_MODEL_MAX_OUTPUT_TOKENS", "4096")),
            os.getenv("IMACT_MODEL_RESPONSE_FORMAT", "text"),
        )

    async def decide(self, context: dict) -> ModelResult:
        # 联合 JSON 协议：完整业务工具定义放在 input.available_tools 中。
        # 原生 tool_calls 留给外层输出适配；本次模型必须同时返回两个 memory 区。
        messages = [{"role": "system", "content": context["system"]}]
        if context["task_policy"]:
            messages.append({"role": "system", "content": context["task_policy"]})
        messages.append({"role": "user", "content": dumps(context["input"])})
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": self.max_output_tokens,
        }
        if self.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        started = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                response = await client.post(
                    self.base_url + "/chat/completions", json=body, headers=headers
                )
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                message = choice["message"]
                result = ModelResult(
                    message.get("content"),
                    {
                        "model": payload.get("model", self.model),
                        "request_id": payload.get("id"),
                        "finish_reason": choice.get("finish_reason"),
                        "usage": payload.get("usage"),
                        "latency_ms": round((perf_counter() - started) * 1000, 2),
                        "request_parameters": {k: v for k, v in body.items() if k != "messages"},
                        "response_format": "native_tool_calls"
                        if message.get("tool_calls")
                        else "content",
                    },
                )
                if choice.get("finish_reason") not in {None, "stop"}:
                    raise ModelResponseError("model_incomplete", "模型输出未正常完成", result)
                if message.get("tool_calls") or not isinstance(message.get("content"), str):
                    raise ModelResponseError(
                        "model_format", "模型需要在 content 返回联合 JSON", result
                    )
                return result
        except httpx.HTTPStatusError as exc:
            raise HarnessError(
                "model_http_error", f"模型服务 HTTP {exc.response.status_code}", 502
            ) from exc
        except httpx.RequestError as exc:
            raise HarnessError(
                "model_transport_error", "模型请求失败，可使用相同事件重试", 502
            ) from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise HarnessError("model_format", "模型服务响应格式无法解析", 502) from exc
