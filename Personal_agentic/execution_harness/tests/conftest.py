from copy import deepcopy

import pytest

from imact.demo import call, decision, fixture
from imact.harness import ExecutionHarness
from imact.store import SQLiteStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


class QueueModel:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.contexts = []

    async def decide(self, context):
        self.contexts.append(deepcopy(context))
        result = self.outputs.pop(0)
        if isinstance(result, Exception):
            raise result
        return deepcopy(result)


@pytest.fixture
def make_harness(tmp_path):
    def factory(*outputs):
        model = QueueModel(*outputs)
        store = SQLiteStore(tmp_path / "workspace.sqlite3")
        return ExecutionHarness(store, model), model, store

    return factory


def start(tid="test", eid="e1"):
    return {
        "task_id": tid,
        "event_id": eid,
        "base_rev": 0,
        "event": {
            "type": "start",
            "query": "帮我点午餐到单位",
            "tools": fixture("delivery_tools.json"),
            "memory": {"status": "ok", "context": "喜欢辛香，优先30分钟送达"},
        },
    }


def event(response, value, eid=None):
    return {
        "task_id": response.task_id,
        "event_id": eid or f"e{response.rev + 1}",
        "base_rev": response.rev,
        "event": value,
    }


def result(response, content, delivery="received"):
    c = response.action.calls[0]
    return event(
        response,
        {
            "type": "tool_results",
            "results": [
                {"call_id": c.call_id, "name": c.name, "delivery": delivery, "content": content}
            ],
        },
    )


def reply(text="请问是否继续？", mode="ask", **kwargs):
    return decision({"type": "reply_user", "mode": mode, "content": text, **kwargs})


def search():
    return decision(call("delivery_store_search_recommand", keywords=["辣味"]))
