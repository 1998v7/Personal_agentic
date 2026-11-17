"""固定决策的离线外卖联调样例。不是实际模型推理，也不是 VitaBench 得分。"""

import json
from importlib.resources import files
from pathlib import Path
from uuid import uuid4

from .harness import ExecutionHarness
from .store import SQLiteStore

STORE = "S17790990122987262_S00043"
PRODUCT = "S17790990122987262_P00103"


def fixture(name):
    return json.loads(files("imact").joinpath("fixtures/" + name).read_text())


def decision(action, changes=None, notes=None):
    return {"task_changes": changes or {}, "working_notes": notes or [], "next_action": action}


def call(name, **arguments):
    return {"type": "call_tools", "calls": [{"name": name, "arguments": arguments}]}


class DemoModel:
    """仅适用于本文件演示脚本；根据事件阶段返回固定输出。"""

    def __init__(self):
        self.calls = 0

    async def decide(self, context):
        self.calls += 1
        event = context["input"]["latest_event"]
        if event["type"] == "start":
            return decision(
                {"type": "query_memory", "question": "本次午餐相关的口味和配送偏好是什么？"},
                {
                    "goal": "订午餐外卖",
                    "meal_period": "lunch",
                    "destination": "work",
                    "completion": {"kind": "purchase", "count": 1},
                },
            )
        if event["type"] == "memory_result":
            return decision(
                call("delivery_store_search_recommand", keywords=["烧烤", "辣味"]),
                {"preferred_delivery_minutes": 30, "taste_preferences": ["辛香"]},
            )
        if event["type"] == "user_message":
            return decision(
                call("delivery_product_search_recommand", keywords=["牛肉", "蔬菜"]),
                {"store_choice": "agent", "product_choice": "agent"},
            )
        name = event["results"][0]["name"]
        if name == "delivery_store_search_recommand":
            return decision(
                {
                    "type": "reply_user",
                    "mode": "ask",
                    "content": "已经找到门店，需要我继续帮您选菜吗？",
                },
                notes=[
                    {
                        "kind": "issue",
                        "topic": "missing_product_id",
                        "status": "open",
                        "text": "目前只有门店候选，需要具体菜品与真实商品 ID。",
                    }
                ],
            )
        if name == "delivery_product_search_recommand":
            return decision(
                call(
                    "create_delivery_order",
                    user_id="demo_user",
                    store_id=STORE,
                    product_ids=[PRODUCT],
                    product_cnts=[1],
                    address="演示办公地址",
                    dispatch_time="2027-01-01 12:00:00",
                    note="离线联调演示",
                ),
                notes=[
                    {
                        "kind": "candidate",
                        "item_id": PRODUCT,
                        "assessment": "consider",
                        "text": "此商品已有真实 ID 和门店关联，可用于联调参数校验。",
                    },
                    {
                        "kind": "issue",
                        "topic": "missing_product_id",
                        "status": "resolved",
                        "text": "搜索回执提供了具体商品 ID。",
                    },
                ],
            )
        if name == "create_delivery_order":
            order_id = next(iter(context["input"]["business_state"]["orders"]))
            return decision(call("pay_delivery_order", order_id=order_id))
        return decision(
            {
                "type": "reply_user",
                "mode": "final",
                "outcome": "completed",
                "content": "离线演示：已收到合成订单的支付成功回执。",
            }
        )


async def run_demo(directory: str | Path = "data/demo") -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tid = "demo-" + uuid4().hex[:10]
    store, model = SQLiteStore(directory / "workspace.sqlite3"), DemoModel()
    harness = ExecutionHarness(store, model)
    observations = fixture("delivery_observations.json")
    event = {
        "type": "start",
        "query": "帮我点午餐外卖送到单位",
        "tools": fixture("delivery_tools.json"),
        "system_prompt": "离线演示。用户 demo_user，办公地址为演示办公地址。允许模拟下单与支付。",
    }
    rev = 0
    while True:
        response = await harness.advance(
            {"task_id": tid, "event_id": f"e{rev + 1}", "base_rev": rev, "event": event}
        )
        rev = response.rev
        action = response.action.model_dump(exclude_none=True)
        if action["type"] == "query_memory":
            event = {
                "type": "memory_result",
                "request_id": action["request_id"],
                "status": "ok",
                "context": "演示偏好：辛香，优先30分钟内配送。",
            }
        elif action["type"] == "reply_user":
            if action["mode"] == "final":
                break
            event = {
                "type": "user_message",
                "content": "随便，你看着办就行。",
                "reply_to": f"e{rev}",
            }
        else:
            c = action["calls"][0]
            responses = {
                "delivery_store_search_recommand": observations["stores"],
                "delivery_product_search_recommand": observations["products"],
                "create_delivery_order": f"Order(order_id:DEMO_ORDER, order_type:delivery, user_id:demo_user, store_id:{STORE}, status:unpaid, products:[])",
                "pay_delivery_order": "Payment successful",
            }
            event = {
                "type": "tool_results",
                "results": [
                    {
                        "call_id": c["call_id"],
                        "name": c["name"],
                        "delivery": "received",
                        "content": responses[c["name"]],
                    }
                ],
            }
    export = {
        "mode": "scripted_model_and_synthetic_order_receipts",
        "task_id": tid,
        "model_calls": model.calls,
        "result": response.model_dump(exclude_none=True),
        "snapshot": store.inspect(tid),
        "trace": store.trace(tid),
    }
    path = directory / f"{tid}.json"
    path.write_text(json.dumps(export, ensure_ascii=False, indent=2) + "\n")
    return {
        "task_id": tid,
        "model_calls": model.calls,
        "rev": rev,
        "export": str(path.resolve()),
        "notice": "固定决策 + 真实搜索回执 + 合成订单/支付回执；未调用模型 API 或业务服务。",
    }
