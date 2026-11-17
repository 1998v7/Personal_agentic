import pytest
from conftest import event, reply, result, search, start

from imact.demo import call, decision, fixture
from imact.domain import DeliveryDomain
from imact.errors import HarnessError
from imact.model import ModelResult
from imact.protocol import MemoryPayload

pytestmark = pytest.mark.anyio


async def test_retry_includes_rejection_and_has_a_bound(make_harness):
    h, model, store = make_harness("bad", "bad", "bad", reply("收到纠正"))
    request = start()
    for _ in range(2):
        with pytest.raises(HarnessError):
            await h.advance(request)
    r = await h.advance(request)
    assert r.action.outcome == "blocked" and r.rev == 1
    assert len(model.contexts[-1]["input"]["repair_feedback"]) == 2
    assert model.contexts[-1]["input"]["repair_feedback"][0]["error"]["code"] == "invalid_decision"
    assert await h.advance(request) == r
    assert len(model.contexts) == 3
    assert (await h.advance(event(r, {"type": "user_message", "content": "改成晚餐"}))).rev == 2
    assert len(store.trace("test")["attempts"]) == 4


async def test_correction_preserves_observed_facts_and_supersedes_failed_event(make_harness):
    h, model, store = make_harness(search(), "bad", reply("改为清淡"))
    r = await h.advance(start())
    failed = result(r, fixture("delivery_observations.json")["stores"])
    with pytest.raises(HarnessError):
        await h.advance(failed)
    correction = event(r, {"type": "user_message", "content": "不要辣了"}, eid="correction")
    r = await h.advance(correction)
    assert len(model.contexts[-1]["input"]["candidates"]) == 43
    assert not store.inspect("test")["workspace"]["pending_calls"]
    assert store.trace("test")["events"][1]["status"] == "superseded"
    with pytest.raises(HarnessError) as caught:
        await h.advance(failed)
    assert caught.value.code == "event_superseded"


async def test_terminate_failed_start_without_calling_model(make_harness):
    h, model, store = make_harness("bad", reply())
    with pytest.raises(HarnessError):
        await h.advance(start())
    stopped = await h.advance(
        dict(
            task_id="test",
            event_id="stop",
            base_rev=0,
            event={"type": "terminate", "reason": "cancelled"},
        )
    )
    assert stopped.rev == 1 and stopped.action.outcome == "partial"
    assert len(model.contexts) == 1
    assert store.inspect("test")["workspace"]["stop_reason"] == "cancelled"
    await h.advance(event(stopped, {"type": "resume"}))
    assert len(model.contexts) == 2


async def test_stop_keeps_pending_calls_and_accepts_late_observation(make_harness):
    h, model, store = make_harness(search())
    dispatched = await h.advance(start())
    stopped = await h.advance(event(dispatched, {"type": "terminate"}))
    late = result(dispatched, fixture("delivery_observations.json")["stores"])
    late.update(base_rev=stopped.rev, event_id="late")
    await h.advance(late)
    w = store.inspect("test")["workspace"]
    assert w["lifecycle"] == "stopped" and not w["pending_calls"] and len(w["facts"]) == 43
    assert len(model.contexts) == 1


async def test_max_turns_stops_before_another_model_call(make_harness):
    h, model, store = make_harness(reply())
    request = start()
    request["event"]["max_turns"] = 1
    r = await h.advance(request)
    r = await h.advance(event(r, {"type": "user_message", "content": "继续"}))
    assert r.action.outcome == "partial" and len(model.contexts) == 1
    assert store.inspect("test")["workspace"]["stop_reason"] == "max_turns"


def issue(text, status="open", **kw):
    return dict(kind="issue", topic="other_requirement", status=status, text=text, **kw)


async def test_multiple_issues_same_topic_have_independent_identity(make_harness):
    first = reply()
    first["working_notes"] = [issue("确认地址"), issue("确认预算")]
    h, model, store = make_harness(first)
    r = await h.advance(start())
    notes = store.inspect("test")["workspace"]["notes"]
    assert len(notes) == 2
    ids = {n["value"]["text"]: n["value"]["note_id"] for n in notes}
    close = reply("地址已确认")
    close["working_notes"] = [issue("送到单位", "resolved", note_id=ids["确认地址"])]
    model.outputs.append(close)
    await h.advance(event(r, {"type": "user_message", "content": "送单位，预算稍后说"}))
    notes = store.inspect("test")["workspace"]["notes"]
    assert sum(n["value"]["status"] == "open" for n in notes) == 1
    resolved = next(n for n in notes if n["value"]["status"] == "resolved")
    assert resolved["source_refs"] == ["e2"] and resolved["provenance"] == "model_judgment"


async def test_default_memory_tools_filtered_but_explicit_benchmark_mode_supported(make_harness):
    h, model, _ = make_harness(reply(), reply())
    await h.advance(start())
    names = {t["function"]["name"] for t in model.contexts[-1]["input"]["available_tools"]}
    assert "read_preference_memory" not in names and "query_preference_memory" not in names
    request = start(tid="baseline")
    request["event"]["memory_mode"] = "benchmark"
    await h.advance(request)
    assert len(model.contexts[-1]["input"]["available_tools"]) == 21


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"context": "喜欢辣"}, "ok"),
        ({"context": ""}, "empty"),
        ({"error": "timeout"}, "error"),
        ({}, "not_requested"),
    ],
)
def test_memory_service_shapes(payload, status):
    assert MemoryPayload.model_validate(payload).status == status


async def test_memory_cannot_overwrite_current_user_requirement(make_harness):
    h, _, store = make_harness(
        decision({"type": "query_memory", "question": "口味"}, {"required_delivery_minutes": 20}),
        decision(
            {"type": "reply_user", "mode": "inform", "content": "旧偏好30分钟"},
            {"required_delivery_minutes": 30},
        ),
    )
    r = await h.advance(start())
    with pytest.raises(HarnessError) as caught:
        await h.advance(
            event(
                r,
                {
                    "type": "memory_result",
                    "request_id": r.action.request_id,
                    "context": "平时30分钟",
                },
            )
        )
    assert caught.value.code == "task_requirement_overwrite"
    assert store.inspect("test")["workspace"]["task"]["required_delivery_minutes"] == 20


async def test_recommendation_can_complete_without_an_order(make_harness):
    output = reply("建议选择清淡菜品", mode="final", outcome="completed")
    output["task_changes"] = {"completion": {"kind": "answer"}}
    h, _, _ = make_harness(output)
    request = start()
    request["event"]["query"] = "给我午餐建议，不用下单"
    r = await h.advance(request)
    assert r.action.outcome == "completed"


def test_completion_requires_all_orders_and_goal_specific_evidence():
    d = DeliveryDomain()
    w = dict(
        task={"completion": {"kind": "purchase", "count": 2}},
        pending_calls={},
        pending_memory=None,
        orders={
            "O1": dict(status="paid", created_in_task=True, source_refs=["payment1"]),
            "O2": dict(status="unpaid", created_in_task=True, source_refs=["create2"]),
        },
    )
    assert not d.completion_evidence(w)
    w["orders"]["O2"]["status"] = "paid"
    assert d.completion_evidence(w) == ["payment1", "create2"]
    w["orders"]["O2"]["created_in_task"] = False
    assert not d.completion_evidence(w)
    w["task"]["completion"] = {"kind": "cancel", "entity_ids": ["O1", "O2"]}
    w["orders"]["O1"]["status"] = "cancelled"
    assert not d.completion_evidence(w)
    w["orders"]["O2"]["status"] = "cancelled"
    assert d.completion_evidence(w)


async def test_model_metadata_stored_without_changing_decision_protocol(make_harness):
    h, _, store = make_harness(
        ModelResult(
            reply(), {"model": "test", "usage": {"total_tokens": 123}, "request_id": "req1"}
        )
    )
    await h.advance(start())
    meta = store.trace("test")["attempts"][0]["metadata"]
    assert meta["usage"]["total_tokens"] == 123 and meta["request_id"] == "req1"
    assert meta["latency_ms"] >= 0


async def test_existing_order_lookup_can_be_followed_by_cancel(make_harness):
    h, _, store = make_harness(
        decision(
            call("search_delivery_orders", user_id="demo", status="paid"),
            {"completion": {"kind": "cancel"}},
        ),
        decision(
            call("get_delivery_order_detail", order_id="EXISTING"),
            {"completion": {"kind": "cancel", "entity_ids": ["EXISTING"]}},
        ),
        decision(call("cancel_delivery_order", order_id="EXISTING")),
        reply("已取消", mode="final", outcome="completed"),
    )
    r = await h.advance(start())
    r = await h.advance(result(r, "Order(order_id:EXISTING, order_type:delivery, status:paid, "))
    r = await h.advance(
        result(r, "Order(order_id:EXISTING, order_type:delivery, status:paid, products:[])")
    )
    r = await h.advance(result(r, "Order EXISTING has been cancelled."))
    assert r.action.outcome == "completed"
    assert store.inspect("test")["workspace"]["orders"]["EXISTING"]["created_in_task"] is False


async def test_tool_cannot_weaken_completion(make_harness):
    output = search()
    output["task_changes"] = {"completion": {"kind": "purchase", "count": 2}}
    bad = reply("完成", mode="final", outcome="completed")
    bad["task_changes"] = {"completion": {"kind": "answer"}}
    h, _, _ = make_harness(output, bad)
    r = await h.advance(start())
    with pytest.raises(HarnessError) as caught:
        await h.advance(result(r, fixture("delivery_observations.json")["stores"]))
    assert caught.value.code == "completion_scope"


async def test_evidence_lookup_is_scoped_and_returns_original_receipt(make_harness):
    h, _, store = make_harness(search(), reply())
    r = await h.advance(start())
    cid = r.action.calls[0].call_id
    raw = fixture("delivery_observations.json")["stores"]
    await h.advance(result(r, raw))
    assert store.evidence("test", "e2/" + cid)["result"]["content"] == raw
    assert store.evidence("test", "e1")["event"]["type"] == "start"
    with pytest.raises(HarnessError) as caught:
        store.evidence("other-task", "e2/" + cid)
    assert caught.value.code == "evidence_not_found"


async def test_context_budget_archives_old_memory_without_dropping_task_constraints(make_harness):
    from imact.context import ContextAssembler
    from imact.harness import fresh_workspace
    from imact.protocol import AdvanceInput

    request = AdvanceInput.model_validate(start()).model_dump(mode="json")
    w = fresh_workspace(request["event"], None)
    w["task"] = {"budget_amount": 100, "dietary_constraints": ["不吃花生"]}
    w["progress"] = {}
    w["user_memory"] = [
        {"status": "ok", "context": "初始资料"},
        {"status": "ok", "context": "历史补充" * 10000, "source_event": "e2"},
        {"status": "ok", "context": "本轮资料", "source_event": "e3"},
    ]
    packet = ContextAssembler(max_bytes=50000).build(
        w, {"type": "user_message", "content": "继续"}, "e4", DeliveryDomain().decision_schema
    )
    assert packet["manifest"]["omitted"]["user_memory"] == 1
    assert packet["input"]["current_task"] == w["task"]
    assert packet["input"]["archived_evidence"] == [{"type": "memory", "ref": "e2"}]
    assert len(w["user_memory"]) == 3
