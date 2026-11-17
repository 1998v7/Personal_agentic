import asyncio
from copy import deepcopy

import pytest
from conftest import event, reply, result, search, start

from imact.context import ContextAssembler
from imact.demo import PRODUCT, STORE, call, decision, fixture, run_demo
from imact.errors import HarnessError
from imact.harness import ExecutionHarness

pytestmark = pytest.mark.anyio


async def test_full_offline_flow_and_sources(tmp_path):
    report = await run_demo(tmp_path)
    assert report["rev"] == report["model_calls"] == 7
    from imact.store import SQLiteStore

    store = SQLiteStore(tmp_path / "workspace.sqlite3")
    task = store.inspect(report["task_id"])["workspace"]
    assert len(task["facts"]) == 143
    assert task["orders"]["DEMO_ORDER"]["status"] == "paid"
    assert task["task"]["store_choice"] == "agent"
    assert not task["pending_calls"]
    trace = store.trace(report["task_id"])
    assert len(trace["events"]) == len(trace["attempts"]) == 7
    assert trace["events"][-1]["output"]["action"]["evidence_refs"]
    assert trace["attempts"][3]["context"]["input"]["previous_reply"]["content"]
    assert "store_choice" not in trace["attempts"][3]["context"]["input"]["current_task"]


async def test_same_event_replays_even_after_restart_without_model(make_harness):
    harness, model, store = make_harness(search())
    request = start()
    first = await harness.advance(request)
    second = await ExecutionHarness(store, model).advance(request)
    assert first == second
    assert len(model.contexts) == 1
    changed = deepcopy(request)
    changed["event"]["query"] = "changed"
    with pytest.raises(HarnessError, match="输入发生变化"):
        await harness.advance(changed)


async def test_stale_revision_and_task_isolation(make_harness):
    harness, model, store = make_harness(reply(), reply())
    first = await harness.advance(start())
    stale = event(first, {"type": "user_message", "content": "继续"})
    stale["base_rev"] = 0
    with pytest.raises(HarnessError) as caught:
        await harness.advance(stale)
    assert caught.value.code == "revision_conflict"
    await harness.advance(start(tid="second-trial"))
    assert len(model.contexts) == 2
    assert store.inspect("test")["rev"] == 1
    assert store.inspect("second-trial")["rev"] == 1


async def test_invalid_incoming_call_does_not_poison_task(make_harness):
    harness, model, store = make_harness(search(), reply())
    first = await harness.advance(start())
    wrong = result(first, "receipt")
    wrong["event"]["results"][0]["call_id"] = "invented"
    with pytest.raises(HarnessError) as caught:
        await harness.advance(wrong)
    assert caught.value.code == "unknown_tool_call"
    correct = result(first, fixture("delivery_observations.json")["stores"])
    correct["event_id"] = "e2-corrected"
    await harness.advance(correct)
    assert len(store.inspect("test")["workspace"]["facts"]) == 43
    assert len(model.contexts) == 2


async def test_model_failure_keeps_raw_input_then_same_event_can_retry(make_harness):
    harness, model, store = make_harness("not-json", search())
    request = start()
    with pytest.raises(HarnessError) as caught:
        await harness.advance(request)
    assert caught.value.code == "invalid_decision"
    trace = store.trace("test")
    assert trace["events"][0]["input"]["event"]["query"] == request["event"]["query"]
    assert trace["attempts"][0]["decision"] == "not-json"
    response = await harness.advance(request)
    assert response.rev == 1
    assert len(store.trace("test")["attempts"]) == 2


async def test_concurrent_same_event_gets_one_decision(make_harness):
    entered, release = asyncio.Event(), asyncio.Event()

    class BlockingModel:
        calls = 0

        async def decide(self, context):
            self.calls += 1
            entered.set()
            await release.wait()
            return search()

    harness, _, store = make_harness()
    model = BlockingModel()
    one, two = ExecutionHarness(store, model), ExecutionHarness(store, model)
    task = asyncio.create_task(one.advance(start()))
    await entered.wait()
    with pytest.raises(HarnessError) as caught:
        await two.advance(start())
    assert caught.value.code == "event_in_progress"
    release.set()
    first = await task
    assert await two.advance(start()) == first
    assert model.calls == 1


async def test_memory_roundtrip_preserves_question_and_empty_status(make_harness):
    query = decision({"type": "query_memory", "question": "口味偏好？"})
    harness, model, store = make_harness(query, reply())
    response = await harness.advance(start())
    await harness.advance(
        event(
            response,
            {
                "type": "memory_result",
                "request_id": response.action.request_id,
                "status": "empty",
                "context": "",
            },
        )
    )
    memory = model.contexts[-1]["input"]["user_memory"][-1]
    assert memory["status"] == "empty" and memory["question"] == "口味偏好？"
    assert store.inspect("test")["workspace"]["pending_memory"] is None


async def test_memory_loop_is_bounded(make_harness):
    query = decision({"type": "query_memory", "question": "口味？"})
    harness, _, store = make_harness(query, query, query)
    r = await harness.advance(start())
    r = await harness.advance(
        event(
            r,
            {
                "type": "memory_result",
                "request_id": r.action.request_id,
                "status": "empty",
                "context": "",
            },
        )
    )
    with pytest.raises(HarnessError) as caught:
        await harness.advance(
            event(
                r,
                {
                    "type": "memory_result",
                    "request_id": r.action.request_id,
                    "status": "empty",
                    "context": "",
                },
            )
        )
    assert caught.value.code == "memory_query_limit"
    assert store.inspect("test")["rev"] == 2


@pytest.mark.parametrize(
    "bad",
    [
        {
            "task_changes": {"paid": True},
            "working_notes": [],
            "next_action": {"type": "reply_user", "mode": "ask", "content": "?"},
        },
        {
            "task_state_patch": [{"op": "set", "field": "goal", "value": "x"}],
            "working_notes": [],
            "next_action": {},
        },
        decision(call("invented_tool", anything=1)),
        decision(call("delivery_store_search_recommand", keywords="wrong-type")),
        decision(
            {"type": "reply_user", "mode": "final", "outcome": "completed", "content": "已完成"}
        ),
        decision(
            {"type": "reply_user", "mode": "ask", "content": "?"},
            notes=[
                {
                    "kind": "candidate",
                    "item_id": "invented",
                    "assessment": "consider",
                    "text": "guess",
                }
            ],
        ),
    ],
)
async def test_invalid_model_output_never_commits_action(make_harness, bad):
    harness, _, store = make_harness(bad)
    with pytest.raises(HarnessError):
        await harness.advance(start())
    assert store.trace("test")["events"][0]["output"] is None


async def test_clearing_requires_explicit_outer_annotation(make_harness):
    action = {"type": "reply_user", "mode": "ask", "content": "继续？"}
    harness, model, store = make_harness(
        decision(action, {"preferred_delivery_minutes": 30}),
        decision(action),
        decision(action, {"preferred_delivery_minutes": None}),
        decision(action),
        decision(action, {"preferred_delivery_minutes": None}),
    )
    r = await harness.advance(start())
    r = await harness.advance(event(r, {"type": "user_message", "content": "继续"}))
    assert store.inspect("test")["workspace"]["task"]["preferred_delivery_minutes"] == 30
    req = event(r, {"type": "user_message", "content": "继续"})
    with pytest.raises(HarnessError) as caught:
        await harness.advance(req)
    assert caught.value.code == "unconfirmed_clear"
    r = await harness.advance(req)
    r = await harness.advance(
        event(
            r,
            {
                "type": "user_message",
                "content": "取消配送时间偏好",
                "allowed_clears": ["preferred_delivery_minutes"],
            },
        )
    )
    assert "preferred_delivery_minutes" not in store.inspect("test")["workspace"]["task"]


async def test_real_invalid_product_is_rejected_before_dispatch(make_harness):
    create = call(
        "create_delivery_order",
        user_id="demo",
        store_id=STORE,
        product_ids=["P123456"],
        product_cnts=[1],
        address="示例地址",
        dispatch_time="2027-01-01 12:00:00",
    )
    harness, _, store = make_harness(search(), decision(create))
    r = await harness.advance(start())
    with pytest.raises(HarnessError) as caught:
        await harness.advance(result(r, fixture("delivery_observations.json")["stores"]))
    assert caught.value.code == "ungrounded_id"
    assert store.inspect("test")["rev"] == 1
    assert store.trace("test")["events"][-1]["input"]["event"]["results"][0]["content"]


async def test_unknown_write_cannot_be_reissued(make_harness):
    create = decision(
        call(
            "create_delivery_order",
            user_id="demo",
            store_id=STORE,
            product_ids=[PRODUCT],
            product_cnts=[1],
            address="示例地址",
            dispatch_time="2027-01-01 12:00:00",
        )
    )
    harness, _, store = make_harness(
        search(),
        decision(call("delivery_product_search_recommand", keywords=["牛肉"])),
        create,
        create,
    )
    r = await harness.advance(start())
    r = await harness.advance(result(r, fixture("delivery_observations.json")["stores"]))
    r = await harness.advance(result(r, fixture("delivery_observations.json")["products"]))
    with pytest.raises(HarnessError) as caught:
        await harness.advance(result(r, None, delivery="unknown"))
    assert caught.value.code == "unresolved_write"
    assert store.inspect("test")["rev"] == 3


async def test_context_overflow_is_explicit_and_input_kept(make_harness):
    harness, model, store = make_harness(search())
    harness.context = ContextAssembler(max_bytes=1)
    with pytest.raises(HarnessError) as caught:
        await harness.advance(start())
    assert caught.value.code == "context_overflow"
    assert not model.contexts
    assert store.trace("test")["events"][0]["input"]


async def test_model_timeout_releases_event_for_retry(make_harness):
    class Slow:
        async def decide(self, context):
            await asyncio.Event().wait()

    h, model, store = make_harness(search())
    h.model, h.decision_timeout = Slow(), 0.01
    with pytest.raises(HarnessError) as caught:
        await h.advance(start())
    assert caught.value.code == "decision_timeout"
    h.model = model
    assert (await h.advance(start())).rev == 1
