"""给外层 Harness 的接入工具；这里不执行业务工具、不调用 User Memory 服务。"""

import httpx2 as httpx

from .errors import HarnessError
from .protocol import AdvanceInput, AdvanceOutput
from .store import dumps


def to_chat_message(output: AdvanceOutput | dict) -> dict:
    """将公开动作转成 Chat assistant 消息；内部 Memory 动作需外层先处理。"""
    output = AdvanceOutput.model_validate(output)
    action = output.action.model_dump(exclude_none=True)
    if action["type"] == "reply_user":
        return {"role": "assistant", "content": action["content"]}
    if action["type"] == "call_tools":
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call["call_id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": dumps(call["arguments"])},
                }
                for call in action["calls"]
            ],
        }
    raise HarnessError("internal_action", "query_memory 应由外层处理后回送 memory_result")


class ExecutionClient:
    """可由外层工程直接使用的异步 HTTP 客户端。重试时保留原请求的 ID 和版本。"""

    def __init__(
        self,
        base_url: str,
        token: str = "",
        timeout: float = 130,
        trust_env: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url, self.token, self.timeout = base_url.rstrip("/"), token, timeout
        self.trust_env = trust_env
        self.transport = transport

    async def advance(self, event: AdvanceInput | dict) -> AdvanceOutput:
        event = AdvanceInput.model_validate(event)
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        async with httpx.AsyncClient(
            timeout=self.timeout, trust_env=self.trust_env, transport=self.transport
        ) as client:
            response = await client.post(
                self.base_url + "/v1/execution/advance",
                json=event.model_dump(mode="json"),
                headers=headers,
            )
            response.raise_for_status()
            return AdvanceOutput.model_validate(response.json())
