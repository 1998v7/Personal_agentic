import json
from pathlib import Path

import pytest
from conftest import reply, result, start

from imact.demo import call, decision
from imact.domain import VitaDomain
from imact.errors import HarnessError
from imact.vitabench import CATALOG, TOOLS, parse_records

FIXTURES = Path(__file__).parent / "fixtures/vitabench"
REPORT = json.loads((FIXTURES / "receipts.json").read_text())
RECEIPTS = REPORT["receipts"]
SCHEMAS = json.loads((FIXTURES / "tools.json").read_text())


def workspace():
    return dict(task={}, orders={}, facts={}, pending_calls={}, pending_memory=None)


def test_catalog_covers_actual_tool_schemas_and_pinned_source():
    assert len(TOOLS) == 72
    assert {s["function"]["name"] for s in SCHEMAS} == set(TOOLS)
    assert REPORT["files"] == CATALOG["files"] and REPORT["revision"] == CATALOG["revision"]
    assert {r["name"] for r in RECEIPTS} == set(TOOLS)


@pytest.mark.anyio
@pytest.mark.parametrize("domain", ["delivery", "instore", "ota"])
async def test_actual_receipts_through_harness(domain, make_harness):
    rows = [r for r in RECEIPTS if r["domain"] == domain]
    # taxi 是上游真实失败；未知写结果后不再继续发写操作。
    if domain == "ota":
        rows = rows[: next(i for i, r in enumerate(rows) if r["name"] == "create_taxi_order") + 1]
    decisions = [decision(call(r["name"], **r["arguments"])) for r in rows]
    decisions.append(reply("已处理回执", mode="inform"))
    h, model, store = make_harness(*decisions)
    request = start()
    request["event"].update(domain=domain, tools=SCHEMAS)
    r = await h.advance(request)
    for receipt in rows:
        assert r.action.calls[0].name == receipt["name"]
        r = await h.advance(result(r, receipt["content"]))
    w = store.inspect("test")["workspace"]
    assert len(w["observations"]) == len(rows)
    assert all(o["status"] == "cancelled" for o in w["orders"].values())
    if domain == "ota":
        pending = list(w["pending_calls"].values())
        assert len(pending) == 1 and pending[0]["name"] == "create_taxi_order"
        assert pending[0]["delivery"] == "unknown"
    else:
        assert not w["pending_calls"]


@pytest.mark.parametrize(
    "tool,entity,product,key",
    [
        ("get_ota_hotel_info", "H1", "HP0", "hotel_id"),
        ("get_ota_attraction_info", "A1", "AP0", "attraction_id"),
        ("get_ota_flight_info", "F1", "FP0", "flight_id"),
        ("get_ota_train_info", "T1", "TP0", "train_id"),
        ("get_taxi_info", "X1", "XP1", "taxi_id"),
    ],
)
def test_ota_product_aliases_and_parent_relation(tool, entity, product, key):
    row = next(r for r in RECEIPTS if r["name"] == tool)
    parsed = {r["item_id"]: r for r in parse_records(row["content"])}
    assert parsed[product][key] == entity
    assert parsed[product]["product_id"] == product
    assert parsed[product]["price"] > 0
    assert parsed[product]["parent_kind"] != "order"


@pytest.mark.parametrize("kind", ["flight", "train"])
def test_change_needs_detail_then_additional_payment(kind):
    d, w = VitaDomain("ota"), workspace()
    oid = None
    for index, row in enumerate(RECEIPTS):
        name = row["name"]
        if name not in {
            f"create_{kind}_order",
            f"pay_{kind}_order",
            f"modify_{kind}_order",
            f"get_{kind}_order_detail",
        }:
            continue
        cid = str(index)
        pending = {"name": name, "arguments": row["arguments"]}
        w["pending_calls"][cid] = pending
        status = d.observe(pending, row["content"], w, cid)
        if status != "unknown":
            w["pending_calls"].pop(cid)
        w["pending_calls"] = {
            k: v for k, v in w["pending_calls"].items() if not v.get("resolved_by")
        }
        if name.startswith("create_"):
            oid = next(iter(w["orders"]))
            w["task"]["completion"] = {
                "kind": "modify",
                "entity_ids": [oid],
                "expected_fields": {"date": "2026-11-07"},
            }
        if name.startswith("modify_"):
            assert status == "unknown" and w["orders"][oid]["status"] == "unpaid"
            assert not d.completion_evidence(w)
        if w["orders"][oid].get("modification_source"):
            if name.startswith("get_"):
                assert not w["pending_calls"] and not d.completion_evidence(w)
            elif name.startswith("pay_"):
                assert d.completion_evidence(w)


def test_unclosed_list_and_dictionary_order_products():
    text = (
        "BookInfo(book_id=B1,shop_id=S1, status=unpaid\nBookInfo(book_id=B2,shop_id=S2, status=paid"
    )
    assert [r["item_id"] for r in parse_records(text)] == ["B1", "B2"]
    text = "Order(order_id:O1, status:paid, products:[{'product_id':'P1','date':'2026-11-07'}])"
    records = parse_records(text)
    assert records[0]["products"][0]["date"] == "2026-11-07"


def test_taxi_failure_after_inventory_change_is_not_retryable():
    row = next(r for r in RECEIPTS if r["name"] == "create_taxi_order")
    assert row["state_changed"] and row["exception"]
    d, w = VitaDomain("ota"), workspace()
    call_ = {"name": row["name"], "arguments": row["arguments"]}
    w["pending_calls"]["taxi"] = call_
    assert d.observe(call_, row["content"], w, "receipt") == "unknown"
    assert not w["orders"]


def test_unregistered_tool_not_silently_treated_as_safe_read():
    with pytest.raises(HarnessError) as caught:
        VitaDomain("delivery").validate_call(
            {"name": "transfer_money", "arguments": {}}, workspace(), set()
        )
    assert caught.value.code == "unsupported_tool"


def test_wrong_parent_rejected_before_ota_create():
    w = workspace()
    w["facts"] = {"H2": {"kind": "hotel"}, "HP0": {"kind": "product", "hotel_id": "H1"}}
    with pytest.raises(HarnessError) as caught:
        VitaDomain("ota").validate_call(
            {
                "name": "create_hotel_order",
                "arguments": {"hotel_id": "H2", "room_id": "HP0", "user_id": "u"},
            },
            w,
            set(w["facts"]),
        )
    assert caught.value.code == "store_product_mismatch"
