"""外层 HTTP 联调示例。只对显式 demo 服务工作，工具和 Memory 均为本地假实现。

先启动：IMACT_MODEL_MODE=demo .venv/bin/python -m imact.cli serve
再运行：.venv/bin/python examples/outer_harness_example.py
真实接入时，替换下面的模拟 Memory、工具回执和用户响应三个分支。
"""

import argparse
import asyncio
import json
import os
from uuid import uuid4

import httpx2 as httpx

from imact.adapters import ExecutionClient, to_chat_message
from imact.demo import STORE, fixture


async def run_roundtrip(base_url, token="", transport=None):
    client = ExecutionClient(base_url, token, trust_env=False, transport=transport)
    async with httpx.AsyncClient(trust_env=False, transport=transport) as http:
        health = await http.get(base_url.rstrip("/") + "/health")
        health.raise_for_status()
        if health.json().get("model_mode") != "demo":
            raise RuntimeError("此样例要求服务显式开启 IMACT_MODEL_MODE=demo")
    tid, rev = "http-demo-" + uuid4().hex[:10], 0
    event = {
        "type": "start",
        "query": "帮我点午餐外卖到单位",
        "tools": fixture("delivery_tools.json"),
        "system_prompt": "离线HTTP联调，演示账号 demo_user，地址为演示办公地址。",
    }
    observations = fixture("delivery_observations.json")
    outputs = []
    for _ in range(12):
        request = {"task_id": tid, "event_id": f"e{rev + 1}", "base_rev": rev, "event": event}
        response = await client.advance(request)
        # 网络重投验证：完全相同请求返回相同动作，不重复调用模型。
        assert await client.advance(request) == response
        rev = response.rev
        action = response.action.model_dump(exclude_none=True)
        outputs.append(response.model_dump(exclude_none=True))
        if action["type"] == "query_memory":
            event = {
                "type": "memory_result",
                "request_id": action["request_id"],
                "status": "ok",
                "context": "演示偏好：辛香，优先30分钟内配送。",
            }
        elif action["type"] == "call_tools":
            public_message = to_chat_message(response)
            tool = public_message["tool_calls"][0]
            name = tool["function"]["name"]
            receipts = {
                "delivery_store_search_recommand": observations["stores"],
                "delivery_product_search_recommand": observations["products"],
                "create_delivery_order": f"Order(order_id:HTTP_DEMO_ORDER, order_type:delivery, user_id:demo_user, store_id:{STORE}, status:unpaid, products:[])",
                "pay_delivery_order": "Payment successful",
            }
            event = {
                "type": "tool_results",
                "results": [
                    {
                        "call_id": tool["id"],
                        "name": name,
                        "delivery": "received",
                        "content": receipts[name],
                    }
                ],
            }
        elif action["mode"] != "final":
            event = {
                "type": "user_message",
                "content": "随便，你看着办就行。",
                "reply_to": f"e{rev}",
            }
        else:
            return {
                "mode": "local_http_with_demo_model",
                "task_id": tid,
                "committed_events": rev,
                "http_advance_requests": rev * 2,
                "outputs": outputs,
            }
    raise RuntimeError("演示轮次超过限制")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    args = parser.parse_args()
    report = asyncio.run(run_roundtrip(args.base_url, os.getenv("IMACT_SERVICE_TOKEN", "")))
    print(json.dumps(report, ensure_ascii=False, indent=2))
