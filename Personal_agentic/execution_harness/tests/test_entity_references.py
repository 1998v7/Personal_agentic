from copy import deepcopy

import pytest
from conftest import result, start

from imact.context import ContextAssembler
from imact.demo import call, decision
from imact.domain import VitaDomain
from imact.errors import HarnessError
from imact.harness import fresh_workspace
from imact.protocol import AdvanceInput

PRODUCT = "StoreProduct(product_id=P1, store_id=S1, product_name=午餐, price=26.0)"
SEARCH = {"name": "delivery_product_search_recommand", "arguments": {"keywords": ["午餐"]}}


def workspace():
    event = AdvanceInput.model_validate(start()).model_dump(mode="json")["event"]
    value = fresh_workspace(event, None)
    value["progress"] = {}
    return value


def create(store_id="S1"):
    return {
        "name": "create_delivery_order",
        "arguments": {
            "user_id": "u1",
            "store_id": store_id,
            "product_ids": ["P1"],
            "product_cnts": [1],
            "address": "单位",
            "dispatch_time": "2026-03-02 11:35:00",
            "note": "",
        },
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    "next_call",
    [
        {"name": "get_delivery_store_info", "arguments": {"store_id": "S1"}},
        create(),
    ],
)
async def test_product_search_alone_grounds_detail_or_matching_order(next_call, make_harness):
    h, model, store = make_harness(
        decision(call(SEARCH["name"], **SEARCH["arguments"])),
        decision(call(next_call["name"], **next_call["arguments"])),
    )
    first = await h.advance(start())
    second = await h.advance(result(first, PRODUCT))
    assert second.action.calls[0].name == next_call["name"]
    assert all(a["error"] is None for a in store.trace("test")["attempts"])
    body = model.contexts[-1]["input"]
    parent = next(f for f in body["candidates"] if f["item_id"] == "S1")
    assert parent["identity_only"] is True and parent["kind"] == "store"
    assert parent["source_refs"] == [f"e2/{first.action.calls[0].call_id}"]
    assert not {"score", "shipping_time", "location", "price"} & parent.keys()
    receipt = body["latest_event"]["results"][0]
    assert receipt["record_count"] == 1 and "content" not in receipt


def test_detail_enriches_reference_and_later_product_cannot_erase_details():
    domain, w = VitaDomain(), workspace()
    domain.observe(SEARCH, PRODUCT, w, "search1")
    domain.observe(SEARCH, PRODUCT, w, "search2")
    assert w["facts"]["S1"]["source_refs"] == ["search1", "search2"]
    domain.observe(
        {"name": "get_delivery_store_info", "arguments": {"store_id": "S1"}},
        "Store(name=午餐店, store_id=S1, score=4.7)",
        w,
        "detail",
    )
    parent = w["facts"]["S1"]
    assert parent["record_type"] == "Store" and not parent.get("identity_only")
    assert parent["score"] == 4.7 and "detail" in parent["source_refs"]
    before = deepcopy(parent)
    domain.observe(SEARCH, PRODUCT, w, "search3")
    assert w["facts"]["S1"] == before


def test_invented_ids_wrong_types_and_cross_store_order_stay_rejected():
    domain, w = VitaDomain(), workspace()
    domain.observe(SEARCH, PRODUCT + "\nStoreProduct(product_id=P2, store_id=S2)", w, "search")
    for action, code in [
        (
            {"name": "get_delivery_store_info", "arguments": {"store_id": "invented"}},
            "ungrounded_id",
        ),
        (
            {"name": "get_delivery_store_info", "arguments": {"store_id": "P1"}},
            "entity_type_mismatch",
        ),
        (create("S2"), "store_product_mismatch"),
    ]:
        with pytest.raises(HarnessError) as caught:
            domain.validate_call(action, w, set(w["facts"]))
        assert caught.value.code == code
    with pytest.raises(HarnessError) as caught:
        domain.validate_call(create(), w, {"P1"})
    assert caught.value.code == "ungrounded_id"  # 已归档且不可见的实体仍不放行


@pytest.mark.parametrize(
    "content",
    [
        "店铺 ID 是 S1",
        "Error: StoreProduct(product_id=P1, store_id=S1)",
        "StoreProduct(product_id=P1, attributes=store_id:S1)",
        "Order(order_id=O1, status=paid, products=[StoreProduct(product_id=P1, store_id=S1)])",
    ],
)
def test_free_text_error_and_transaction_line_items_do_not_register_parents(content):
    w = workspace()
    VitaDomain().observe(SEARCH, content, w, "search")
    assert "S1" not in w["facts"]


@pytest.mark.parametrize(
    "record_type,key,kind",
    [
        ("ShopProduct", "shop_id", "shop"),
        ("HotelProduct", "hotel_id", "hotel"),
        ("FlightProduct", "flight_id", "flight"),
        ("TrainProduct", "train_id", "train"),
        ("TaxiProduct", "taxi_id", "taxi"),
    ],
)
def test_other_product_parents_use_explicit_typed_foreign_keys(record_type, key, kind):
    w = workspace()
    product_key = {
        "HotelProduct": "room_id",
        "FlightProduct": "seat_id",
        "TrainProduct": "seat_id",
    }.get(record_type, "product_id")
    VitaDomain("delivery,instore,ota").observe(
        SEARCH,
        f"{record_type}({product_key}=P1, {key}=E1)",
        w,
        "search",
    )
    assert w["facts"]["E1"]["kind"] == kind
    assert w["facts"]["E1"][key] == "E1" and w["facts"]["E1"]["identity_only"]


def test_attraction_ticket_alias_does_not_become_an_attraction():
    w, domain = workspace(), VitaDomain("ota")
    tool = {"name": "get_ota_attraction_info", "arguments": {"attraction_id": "A1"}}
    domain.observe(tool, "AttractionProduct(attraction_id=T1, price=30.0)", w, "ticket")
    assert set(w["facts"]) == {"T1"} and w["facts"]["T1"]["kind"] == "product"
    domain.observe(
        tool,
        "Attraction(attraction_id=A1, products=[AttractionProduct(attraction_id=T1)])",
        w,
        "detail",
    )
    assert w["facts"]["T1"]["attraction_id"] == "A1"
    assert w["facts"]["A1"]["kind"] == "attraction"


def test_reference_never_overwrites_a_conflicting_entity_type():
    w, domain = workspace(), VitaDomain()
    domain.observe(SEARCH, "StoreProduct(product_id=S1)", w, "first")
    domain.observe(SEARCH, PRODUCT, w, "second")
    assert w["facts"]["S1"]["kind"] == "product"
    with pytest.raises(HarnessError) as caught:
        domain.validate_call(create(), w, set(w["facts"]))
    assert caught.value.code == "entity_type_mismatch"


@pytest.mark.parametrize("pinned", [True, False])
def test_context_budget_keeps_parent_of_visible_product(pinned):
    w, domain = workspace(), VitaDomain()
    domain.observe(SEARCH, PRODUCT, w, "old")
    # 父实体先于商品排列，确保它不会成为第一个被裁剪的候选。
    w["facts"] = {k: w["facts"][k] for k in ["S1", "P1"]}
    w["notes"] = {
        "candidate:P1": {
            "value": {
                "kind": "candidate",
                "item_id": "P1",
                "assessment": "consider",
                "text": "用户已选",
            }
        }
    }
    if not pinned:
        w["notes"] = {}
    event = {"type": "user_message", "content": "继续"}
    baseline = ContextAssembler().build(w, event, "new", {})
    w["facts"]["unused"] = {
        "item_id": "unused",
        "kind": "store",
        "source_refs": ["old"],
        "name": "长" * 5000,
    }
    w["facts"] = {k: w["facts"][k] for k in ["S1", "unused", "P1"]}
    packet = ContextAssembler(baseline["manifest"]["input_bytes"] + 100).build(w, event, "new", {})
    assert {f["item_id"] for f in packet["input"]["candidates"]} == {"S1", "P1"}
    domain.validate_call(create(), w, {f["item_id"] for f in packet["input"]["candidates"]})
