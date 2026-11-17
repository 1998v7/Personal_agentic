import pytest
from conftest import event, reply, result, search, start

from imact.demo import PRODUCT, STORE, call, decision, fixture
from imact.domain import DeliveryDomain
from imact.errors import HarnessError

pytestmark = pytest.mark.anyio


def test_entity_name_preserves_parentheses_and_type():
    adapter = DeliveryDomain()
    assert adapter._field("Store(name=门店(分店), store_id=S1, score=4.8)", "name") == "门店(分店)"
    with pytest.raises(HarnessError) as caught:
        adapter.validate_call(
            {"name": "get_delivery_store_info", "arguments": {"store_id": "P1"}},
            {"facts": {"P1": {"kind": "product"}}},
            {"P1"},
        )
    assert caught.value.code == "entity_type_mismatch"


async def test_failed_read_is_grounded_and_not_order_success(make_harness):
    get_product = decision(call("get_delivery_product_info", product_id=PRODUCT))
    harness, model, store = make_harness(
        decision(call("delivery_product_search_recommand", keywords=["牛肉"])), get_product, reply()
    )
    r = await harness.advance(start())
    r = await harness.advance(result(r, fixture("delivery_observations.json")["products"]))
    failed_call = r.action.calls[0].call_id
    r = await harness.advance(result(r, f"Error: {PRODUCT} not found"))
    workspace = store.inspect("test")["workspace"]
    assert not workspace["orders"]
    assert not workspace["pending_calls"]
    failure = workspace["failed_attempts"][0]
    assert failure["attempt"] == failed_call
    assert failure["source_refs"] == ["e3/" + failed_call]
    assert model.contexts[-1]["input"]["failed_attempts"][0] == failure


async def test_wrong_product_store_pair_is_rejected(make_harness):
    wrong_store = "S17790990122987262_S00034"
    bad = decision(
        call(
            "create_delivery_order",
            user_id="demo",
            store_id=wrong_store,
            product_ids=[PRODUCT],
            product_cnts=[1],
            address="演示地址",
            dispatch_time="2027-01-01 12:00:00",
        )
    )
    h, _, _ = make_harness(
        search(), decision(call("delivery_product_search_recommand", keywords=["牛肉"])), bad
    )
    r = await h.advance(start())
    r = await h.advance(result(r, fixture("delivery_observations.json")["stores"]))
    with pytest.raises(HarnessError) as caught:
        await h.advance(result(r, fixture("delivery_observations.json")["products"]))
    assert caught.value.code == "store_product_mismatch"


async def test_unknown_write_can_receive_definite_result_later(make_harness):
    create = decision(
        call(
            "create_delivery_order",
            user_id="demo",
            store_id=STORE,
            product_ids=[PRODUCT],
            product_cnts=[1],
            address="演示地址",
            dispatch_time="2027-01-01 12:00:00",
        )
    )
    h, _, store = make_harness(
        search(),
        decision(call("delivery_product_search_recommand", keywords=["牛肉"])),
        create,
        reply("订单结果待核验", mode="inform"),
        reply("已建单，尚未支付", mode="inform"),
    )
    r = await h.advance(start())
    r = await h.advance(result(r, fixture("delivery_observations.json")["stores"]))
    r = await h.advance(result(r, fixture("delivery_observations.json")["products"]))
    c = r.action.calls[0]
    r = await h.advance(result(r, None, delivery="unknown"))
    assert store.inspect("test")["workspace"]["pending_calls"][c.call_id]["delivery"] == "unknown"
    receipt = f"Order(order_id:LATE_ORDER, store_id:{STORE}, status:unpaid, products:[])"
    r = await h.advance(
        event(
            r,
            {
                "type": "tool_results",
                "results": [
                    {
                        "call_id": c.call_id,
                        "name": c.name,
                        "delivery": "received",
                        "content": receipt,
                    }
                ],
            },
        )
    )
    workspace = store.inspect("test")["workspace"]
    assert not workspace["pending_calls"]
    assert workspace["orders"]["LATE_ORDER"]["status"] == "unpaid"
    assert not DeliveryDomain().completion_evidence(workspace)


async def test_memory_failure_is_distinct_from_empty(make_harness):
    h, model, _ = make_harness(decision({"type": "query_memory", "question": "偏好？"}), reply())
    r = await h.advance(start())
    await h.advance(
        event(
            r,
            {
                "type": "memory_result",
                "request_id": r.action.request_id,
                "status": "error",
                "context": "",
                "error": "upstream timeout",
            },
        )
    )
    memory = model.contexts[-1]["input"]["user_memory"][-1]
    assert memory["status"] == "error"
    assert memory["error"] == "upstream timeout"


async def test_reference_extension_is_explicit_and_context_has_contents(make_harness):
    h, model, _ = make_harness(reply())

    class Resolver:
        def resolve(self, reference):
            return {
                "trusted-tools": fixture("delivery_tools.json"),
                "trusted-policy": "模拟时间与账号",
            }[reference]

    h.resolver = Resolver()
    request = start()
    del request["event"]["tools"]
    request["event"].update(tools_ref="trusted-tools", system_ref="trusted-policy")
    await h.advance(request)
    assert len(model.contexts[0]["input"]["available_tools"]) == 19
    assert model.contexts[0]["task_policy"] == "模拟时间与账号"


async def test_explicit_crash_recovery_and_retry(make_harness):
    h, model, store = make_harness(search())
    from imact.protocol import AdvanceInput

    request = start()
    store.claim(AdvanceInput.model_validate(request).model_dump(mode="json"))
    with pytest.raises(HarnessError) as caught:
        await h.advance(request)
    assert caught.value.code == "event_in_progress"
    store.recover_interrupted("test", "e1")
    assert (await h.advance(request)).rev == 1
    assert len(model.contexts) == 1
