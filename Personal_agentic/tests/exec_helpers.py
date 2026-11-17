"""执行层接入的测试公用件。

DemoProvider 把执行层自带的 DemoModel(固定决策)包成我们的 ModelProvider,
这样整条链路 —— provider → ProviderActionModel → ExecutionHarness → ExecutionOuter —— 不依赖真实模型就能跑通。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# VitaBench 在 import 时会展开 models.yaml 里的 ${OPENAI_API_KEY};这里只为让 import 通过,不会用来发请求。
os.environ.setdefault("OPENAI_API_KEY", "dummy-for-import-only")

from IMAct.exec_layer import build_runtime  # noqa: E402
from IMAct.exec_layer.config import OuterConfig  # noqa: E402
from IMAct.exec_layer.loader import ensure_importable  # noqa: E402
from IMAct.models.base import ChatRequest, ChatResponse, ModelProvider  # noqa: E402

ensure_importable()
from imact.demo import STORE, DemoModel, fixture  # noqa: E402

OBS = fixture("delivery_observations.json")
TOOLS = fixture("delivery_tools.json")
RECEIPTS = {
    "delivery_store_search_recommand": OBS["stores"],
    "delivery_product_search_recommand": OBS["products"],
    "create_delivery_order": f"Order(order_id:T_ORDER, order_type:delivery, user_id:demo_user, store_id:{STORE}, status:unpaid, products:[])",
    "pay_delivery_order": "Payment successful",
}


class DemoProvider(ModelProvider):
    """faults:按调用顺序先消耗的故障(异常实例会被抛出,ChatResponse 会被原样返回),之后恢复正常。"""

    name = "demo"
    model = "demo-model"

    def __init__(self, faults=None, usage=None):
        self.demo = DemoModel()
        self.faults = list(faults or [])
        self.requests: list[ChatRequest] = []
        self.usage = usage or {"prompt_tokens": 10, "completion_tokens": 5}

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self.faults:
            fault = self.faults.pop(0)
            if isinstance(fault, Exception):
                raise fault
            return fault
        decision = await self.demo.decide({"input": json.loads(request.messages[-1]["content"])})
        return ChatResponse(
            content=json.dumps(decision, ensure_ascii=False), finish_reason="stop", usage=dict(self.usage), model="demo-model",
            raw={"id": f"req-{len(self.requests)}"},
        )


def make_cfg(tmp: str | None = None, **overrides) -> OuterConfig:
    base = {
        "exec": {"db_path": str(Path(tmp or tempfile.mkdtemp()) / "exec.sqlite3")},
        "retry": {"model_call": {"max_retries": 3, "backoff_s": 0.0}},
        "memory": {"retry": {"max_retries": 1, "backoff_s": 0.0}},
    }
    for k, v in overrides.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return OuterConfig.model_validate(base)


async def no_sleep(_):
    return None


def make_runtime(provider=None, memory_client=None, **cfg_overrides):
    return build_runtime(make_cfg(**cfg_overrides), provider=provider or DemoProvider(), memory_client=memory_client, sleep=no_sleep)


def start_event(**extra):
    return {
        "type": "start",
        "query": "帮我点午餐外卖送到单位",
        "tools": TOOLS,
        "system_prompt": "离线测试。用户 demo_user,办公地址为演示办公地址。允许模拟下单与支付。",
        **extra,
    }


def run_demo_episode(rt, task_id: str, metadata=None, max_steps: int = 20):
    """按外层合同推进一整个 episode,返回所有 PublicAction。"""
    actions, rev, event = [], 0, start_event()
    for _ in range(max_steps):
        act = rt.bridge.run(rt.outer.drive(task_id, rev, event, metadata))
        actions.append(act)
        rev = act.rev
        if act.type == "reply_user":
            if act.mode == "final":
                return actions
            event = {"type": "user_message", "content": "随便,你看着办就行。", "reply_to": act.event_id}
        else:
            c = act.calls[0]
            event = {
                "type": "tool_results",
                "results": [{"call_id": c["call_id"], "name": c["name"], "delivery": "received", "content": RECEIPTS[c["name"]]}],
            }
    raise AssertionError("episode did not finish")
