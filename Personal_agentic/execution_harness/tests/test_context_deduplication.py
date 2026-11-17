from copy import deepcopy

import pytest
from conftest import event, reply, result, search, start

from imact.context import ContextAssembler, compact
from imact.demo import call, decision, fixture
from imact.errors import HarnessError
from imact.harness import fresh_workspace
from imact.protocol import AdvanceInput
from imact.vitabench import parse_records


def workspace():
    request = start()
    request["event"].update(system_prompt="任务策略：用户在单位用餐", history="用户：之前喜欢清淡")
    initial = AdvanceInput.model_validate(request).model_dump(mode="json")["event"]
    value = fresh_workspace(initial, None)
    value["progress"] = {}
    return value, initial


def observe_candidates(w, content, source="e2/c1"):
    for record in parse_records(content):
        w["facts"][record["item_id"]] = {**record, "source_refs": [source]}


def tool_event(content, call_id="c1"):
    return {
        "type": "tool_results",
        "results": [
            {
                "call_id": call_id,
                "name": "delivery_product_search_recommand",
                "delivery": "received",
                "content": content,
            }
        ],
    }


PRODUCTS = (
    "StoreProduct(product_id=P1, store_id=S1, product_name=午餐甲, "
    "attributes=少油, 辛香, price=20.0, tags=['川菜', '午餐'])\n"
    "StoreProduct(product_id=P2, store_id=S2, product_name=午餐乙, "
    "price=30.0, seasonal_notice=供应到12点)"
)


@pytest.mark.anyio
async def test_start_material_has_one_location_across_three_decisions(make_harness):
    h, model, store = make_harness(search(), reply(), reply())
    request = start()
    request["event"].update(system_prompt="唯一任务策略", history="完整历史对话")
    first = await h.advance(request)
    second = await h.advance(result(first, fixture("delivery_observations.json")["stores"]))
    await h.advance(event(second, {"type": "user_message", "content": "继续", "reply_to": "e2"}))
    for context in model.contexts:
        body = context["input"]
        assert context["task_policy"] == "唯一任务策略"
        assert body["visible_history"] == "完整历史对话"
        assert body["original_query"] == request["event"]["query"]
        assert body["user_memory"][0]["context"] == request["event"]["memory"]["context"]
        assert body["available_tools"]
        events = [body["latest_event"], *(entry["event"] for entry in body["recent_events"])]
        for projected in events:
            if projected["type"] == "start":
                assert (
                    not {"system_prompt", "tools", "memory", "history", "query"} & projected.keys()
                )
    assert model.contexts[-1]["input"]["latest_event"]["reply_to"] == "e2"
    original = store.evidence("test", "e1")["event"]
    assert original["system_prompt"] == "唯一任务策略"
    assert original["tools"] and original["memory"]["context"]


@pytest.mark.anyio
async def test_memory_result_full_text_is_preserved_once_with_status(make_harness):
    h, model, store = make_harness(
        decision({"type": "query_memory", "question": "忌口？"}), reply(), reply()
    )
    first = await h.advance(start())
    incoming = {
        "type": "memory_result",
        "request_id": first.action.request_id,
        "status": "ok",
        "context": "不能吃花生\n请保留全部来源和约束",
    }
    second = await h.advance(event(first, incoming))
    await h.advance(event(second, {"type": "user_message", "content": "继续"}))
    for context in model.contexts[1:]:
        body = context["input"]
        assert body["user_memory"][-1]["context"] == incoming["context"]
        entries = [body["latest_event"], *(entry["event"] for entry in body["recent_events"])]
        projected = next(e for e in entries if e["type"] == "memory_result")
        assert projected["status"] == "ok" and "context" not in projected
        assert projected["context_in"] == "user_memory" and projected["source_ref"] == "e2"
    assert store.evidence("test", "e2")["event"]["context"] == incoming["context"]


@pytest.mark.anyio
async def test_exact_candidate_receipt_uses_visible_fields_and_keeps_raw_evidence(make_harness):
    h, model, store = make_harness(
        decision(call("delivery_product_search_recommand", keywords=["午餐"])), reply()
    )
    first = await h.advance(start())
    await h.advance(result(first, PRODUCTS))
    body = model.contexts[-1]["input"]
    receipt = body["latest_event"]["results"][0]
    assert "content" not in receipt and receipt["record_count"] == 2
    assert receipt["content_in"] == "candidates"
    candidates = {f["item_id"]: f for f in body["candidates"]}
    assert candidates["P1"]["attributes"] == "少油, 辛香"
    assert candidates["P2"]["seasonal_notice"] == "供应到12点"
    assert all(receipt["content_ref"] in f["source_refs"] for f in candidates.values())
    assert store.evidence("test", receipt["content_ref"])["result"]["content"] == PRODUCTS


@pytest.mark.parametrize(
    "content",
    [
        PRODUCTS + "\n配送范围暂时缩小，需另行确认。",
        "提示：套餐需要预约\n" + PRODUCTS,
        "StoreProduct(product_id=P1, price=20.0",  # permissive parser accepts incomplete repr
        "StoreProduct(product_id=P1, price=10.0, price=20.0)",
        "StoreProduct(product_id=P1, products=未能解析的附加套餐)",
        "Error: 上游写入可能已成功，请核验",
        "[121.123, 41.456]",
        {"error": "权限失败", "retry_after": 30},
    ],
)
def test_partial_unknown_and_error_receipts_keep_exact_content(content):
    w, _ = workspace()
    if isinstance(content, str):
        observe_candidates(w, content)
    incoming = tool_event(content)
    before_w, before_event = deepcopy(w), deepcopy(incoming)
    packet = ContextAssembler().build(w, incoming, "e2", {})
    assert packet["input"]["latest_event"]["results"][0]["content"] == content
    assert w == before_w and incoming == before_event


def test_historical_receipt_keeps_old_values_after_candidate_changes():
    w, _ = workspace()
    observe_candidates(w, PRODUCTS)
    w["facts"]["P1"]["price"] = 99.0
    w["recent_events"] = [{"event_id": "e2", "event": tool_event(PRODUCTS), "action": {}}]
    packet = ContextAssembler().build(w, {"type": "user_message", "content": "继续"}, "e3", {})
    assert packet["input"]["recent_events"][0]["event"]["results"][0]["content"] == PRODUCTS


def test_matching_values_without_the_receipt_source_do_not_elide_content():
    w, _ = workspace()
    observe_candidates(w, PRODUCTS, source="different/call")
    packet = ContextAssembler().build(w, tool_event(PRODUCTS), "e2", {})
    assert packet["input"]["latest_event"]["results"][0]["content"] == PRODUCTS


def test_search_order_differing_from_candidate_order_retains_receipt():
    w, _ = workspace()
    observe_candidates(w, PRODUCTS)
    w["facts"] = dict(reversed(list(w["facts"].items())))
    packet = ContextAssembler().build(w, tool_event(PRODUCTS), "e2", {})
    assert packet["input"]["latest_event"]["results"][0]["content"] == PRODUCTS


def test_candidate_numeric_representation_must_roundtrip_exactly():
    w, _ = workspace()
    observe_candidates(w, PRODUCTS)
    w["facts"]["P1"]["price"] = 20  # numerically equal to 20.0, different representation
    packet = ContextAssembler().build(w, tool_event(PRODUCTS), "e2", {})
    assert packet["input"]["latest_event"]["results"][0]["content"] == PRODUCTS


def test_history_receipt_projection_does_not_modify_workspace_or_candidates():
    w, initial = workspace()
    observe_candidates(w, PRODUCTS)
    w["recent_events"] = [
        {"event_id": "e1", "event": initial, "action": {}},
        {"event_id": "e2", "event": tool_event(PRODUCTS), "action": {}},
    ]
    before = deepcopy(w)
    packet = ContextAssembler().build(w, {"type": "user_message", "content": "继续"}, "e3", {})
    assert "content" not in packet["input"]["recent_events"][1]["event"]["results"][0]
    assert packet["input"]["candidates"] == compact(list(w["facts"].values()))
    assert w == before


def test_budget_never_leaves_current_receipt_without_its_candidates():
    w, _ = workspace()
    observe_candidates(w, PRODUCTS)
    packet = ContextAssembler().build(w, tool_event(PRODUCTS), "e2", {})
    required_bytes = packet["manifest"]["input_bytes"]
    with pytest.raises(HarnessError, match="必需上下文"):
        ContextAssembler(max_bytes=required_bytes - 1).build(w, tool_event(PRODUCTS), "e2", {})
    assert len(w["facts"]) == 2
