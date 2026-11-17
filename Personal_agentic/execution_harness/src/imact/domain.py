"""VitaBench 的领域校验与回执归一化；不执行工具，不读取 benchmark 答案。"""

import json
from importlib.resources import files
from itertools import permutations
from typing import Protocol

from jsonschema import Draft202012Validator

from .errors import HarnessError
from .vitabench import (
    MEMORY_TOOLS,
    TOOLS,
    TRANSACTIONS,
    fields,
    parse_records,
    product_parent,
    response_failure,
)

PAID = {"paid", "in-progress", "delivered", "consumed"}
IDS = {
    "store_id": "store",
    "shop_id": "shop",
    "hotel_id": "hotel",
    "attraction_id": "attraction",
    "flight_id": "flight",
    "train_id": "train",
    "taxi_id": "taxi",
    "product_id": "product",
    "room_id": "product",
    "ticket_id": "product",
    "seat_id": "product",
}
TX_IDS = {"order_id": "order", "book_id": "book", "reservation_id": "reservation"}


class DomainAdapter(Protocol):
    name: str
    decision_schema: dict

    def is_write(self, tool: str) -> bool: ...
    def observe(self, call: dict, content: object, workspace: dict, source: str) -> str: ...
    def validate_call(self, call: dict, workspace: dict, visible_ids: set[str]) -> None: ...
    def completion_evidence(self, workspace: dict) -> list[str]: ...


def require(condition: bool, code: str, message: str):
    if not condition:
        raise HarnessError(code, message)


class VitaDomain:
    def __init__(self, name="delivery"):
        self.name = name
        self.tools = {n: t for n, t in TOOLS.items() if t["domain"] in {*name.split(","), "common"}}
        self.writes = {n for n, t in self.tools.items() if t["type"] == "WRITE"}
        self.decision_schema = json.loads(
            files("imact").joinpath("schemas/action-turn-v0.3.json").read_text()
        )
        Draft202012Validator.check_schema(self.decision_schema)

    def is_write(self, tool: str) -> bool:
        return tool in self.writes

    @staticmethod
    def _field(text, name, separator="="):
        # 保留旧扩展测试入口；实际回执统一经 parse_records。
        return fields(text.split("(", 1)[-1].removesuffix(")")).get(name)

    @staticmethod
    def _matches(record, expected):
        for key, value in expected.items():
            actual = record.get(key)
            if key == "date" and actual is None and record.get("products"):
                if all(p.get("date") == value for p in record["products"]):
                    continue
            if actual != value:
                return False
        return True

    def _reconcile(self, w, record, source):
        for pending in w["pending_calls"].values():
            args, name = pending["arguments"], pending["name"]
            if not any(args.get(k) == record["entity_id"] for k in TX_IDS):
                continue
            if (
                name.startswith("pay_")
                and record.get("status") in PAID
                or "cancel" in name
                and record.get("status") == "cancelled"
            ):
                pending["resolved_by"] = source
            elif "modify" in name:
                expected = {k: v for k, v in args.items() if k not in {*TX_IDS, "user_id"}}
                expected = {
                    ({"new_date": "date", "time": "reservation_time"}.get(k, k)): v
                    for k, v in expected.items()
                }
                if expected and self._matches(record, expected):
                    pending["resolved_by"] = source
                    record["modification_source"] = source
                    record.pop("verification_required", None)

    @staticmethod
    def _register_parent_reference(workspace, record, source):
        parent = product_parent(record)
        if parent is None:
            return
        kind, key, item_id = parent
        old = workspace["facts"].get(item_id)
        if old is not None:
            # 已有详情不降级；类型冲突不覆盖，后续调用仍会被类型校验拦截。
            if old.get("kind") == kind and old.get("identity_only"):
                old["source_refs"] = list(dict.fromkeys(old["source_refs"] + [source]))
            return
        workspace["facts"][item_id] = {
            "item_id": item_id,
            "kind": kind,
            key: item_id,
            "record_type": "EntityReference",
            "identity_only": True,
            "source_refs": [source],
        }

    def observe(self, call, content, workspace, source):
        name, args, w = call["name"], call["arguments"], workspace
        w.setdefault("observations", {})[source] = {
            "name": name,
            "arguments": args,
            "content": content,
            "source_refs": [source],
        }
        if response_failure(content):
            if self.is_write(name) and (
                isinstance(content, dict)
                or str(content).startswith(("Error:", "Modification failed", "Cancellation failed"))
                or content == "Order already exists"
            ):
                # 沙箱可先改库存/订单再在序列化时抛错；Error 不是无副作用证明。
                return "unknown"
            return "failed"
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        created = name.startswith("create_") or name in {"instore_book", "instore_reservation"}
        structured = (
            created
            or name == "instore_modify_reservation"
            or self.tools.get(name, {}).get("type") == "READ"
        )
        records = parse_records(text) if structured else []
        transactions = []
        for record in records:
            item_id = record["item_id"]
            if record["kind"] in TRANSACTIONS:
                if not (created or any(part in name for part in ("order", "book", "reservation"))):
                    continue
                expected_type = next(
                    (
                        d
                        for d in (
                            "delivery",
                            "instore",
                            "hotel",
                            "attraction",
                            "flight",
                            "train",
                            "taxi",
                        )
                        if d in name
                    ),
                    None,
                )
                if expected_type and record.get("order_type") not in {None, expected_type}:
                    continue
                requested_id = next((args[k] for k in TX_IDS if args.get(k)), None)
                if requested_id is not None and item_id != requested_id:
                    continue
                if record.get("status") not in {
                    "unpaid",
                    "paid",
                    "in-progress",
                    "delivered",
                    "cancelled",
                    "unconsumed",
                    "consumed",
                    "processed",
                }:
                    continue
                old = w["orders"].get(item_id, {})
                merged = {
                    **old,
                    **record,
                    "source_refs": list(dict.fromkeys(old.get("source_refs", []) + [source])),
                }
                if created:
                    merged.update(created_in_task=True, creation_source=source)
                    # create 的简略字符串不含全部实参；仅作为请求参数保存，不伪装回执。
                    merged["creation_arguments"] = args
                    for key in ("store_id", "shop_id", "hotel_id", "product_ids"):
                        if key in args:
                            merged.setdefault(key, args[key])
                else:
                    merged.setdefault("created_in_task", False)
                w["orders"][item_id] = merged
                transactions.append(merged)
                self._reconcile(w, merged, source)
            elif record.get("parent_kind") not in TRANSACTIONS:
                old = w["facts"].get(item_id, {})
                w["facts"][item_id] = {
                    **old,
                    **record,
                    "source_refs": list(dict.fromkeys(old.get("source_refs", []) + [source])),
                }
                w["facts"][item_id].pop("identity_only", None)
                self._register_parent_reference(w, record, source)
        if created:
            expected_kind = {"instore_book": "book", "instore_reservation": "reservation"}.get(
                name, "order"
            )
            return (
                "succeeded"
                if len(transactions) == 1 and transactions[0]["kind"] == expected_kind
                else "unknown"
            )
        entity_id = next((args[k] for k in TX_IDS if k in args), None)
        order = w["orders"].get(entity_id)
        if order:
            status = None
            if name.startswith("pay_") and text == "Payment successful":
                status = "paid"
            elif "cancel" in name and (
                text == f"Order {entity_id} has been cancelled."
                or text.startswith("Cancellation successful, refund amount:")
                or text
                in {
                    f"{cls} {entity_id} is {prefix}cancelled."
                    for cls in ("Order", "BookInfo", "ReservationInfo")
                    for prefix in ("", "already ")
                }
            ):
                status = "cancelled"
            elif name == "get_delivery_order_status" and text in PAID | {"unpaid", "cancelled"}:
                status = text
            if status:
                order.update(
                    status=status, source_refs=list(dict.fromkeys(order["source_refs"] + [source]))
                )
                self._reconcile(w, order, source)
                return "succeeded" if self.is_write(name) else "observed"
            if name == "modify_delivery_order" and text == f"Order {entity_id} has been modified.":
                order.update(note=args["note"], modification_source=source)
                order["source_refs"].append(source)
                return "succeeded"
            if name == "instore_modify_reservation" and transactions:
                return "succeeded" if order.get("modification_source") == source else "unknown"
            if name in {"modify_train_order", "modify_flight_order"} and text.startswith(
                "Modification successful"
            ):
                # 改签 ack 只证实操作被受理；详情核对日期，补款需要独立支付。
                order["verification_required"] = source
                if "need to pay additional amount:" in text:
                    order["status"] = "unpaid"
                return "unknown"
        return "unknown" if self.is_write(name) else "observed"

    def validate_call(self, call, workspace, visible_ids):
        name, args, w = call["name"], call["arguments"], workspace
        if name in MEMORY_TOOLS:
            require(
                w.get("memory_mode") == "benchmark", "memory_route", "用户记忆请走 query_memory"
            )
            return
        require(name in self.tools, "unsupported_tool", "此领域没有注册该工具的读写与回执规则")
        for key, kind in IDS.items():
            if key not in args:
                continue
            value = args[key]
            require(value in visible_ids, "ungrounded_id", f"{key} 不在本轮候选事实中")
            require(w["facts"][value]["kind"] == kind, "entity_type_mismatch", "业务 ID 类型不匹配")
        if name == "create_delivery_order":
            products, counts = args["product_ids"], args["product_cnts"]
            require(
                bool(products)
                and len(products) == len(counts)
                and len(products) == len(set(products))
                and all(type(n) is int and n > 0 for n in counts),
                "invalid_quantity",
                "商品与正整数数量必须一一对应，不可重复商品 ID",
            )
            for pid in products:
                require(pid in visible_ids, "ungrounded_id", "商品 ID 没有可见工具事实")
                require(
                    w["facts"][pid].get("store_id") == args["store_id"],
                    "store_product_mismatch",
                    "商品不属于所选门店",
                )
        for key in ("quantity", "customer_count"):
            if key in args:
                require(
                    type(args[key]) is int and args[key] > 0, "invalid_quantity", "数量必须为正整数"
                )
        if name.startswith("create_"):
            parent = next(
                ((k, v) for k, v in args.items() if k in IDS and IDS[k] != "product"), None
            )
            product = next((v for k, v in args.items() if k in IDS and IDS[k] == "product"), None)
            if parent and product:
                require(
                    w["facts"][product].get(parent[0]) == parent[1],
                    "store_product_mismatch",
                    "商品不属于所选服务实体；请先读取详情",
                )
        for key, kind in TX_IDS.items():
            if key in args and args[key] is not None:
                require(args[key] in w["orders"], "ungrounded_id", "订单/预订/预约 ID 尚未核实")
                order = w["orders"][args[key]]
                require(
                    order.get("kind", "order") == kind, "entity_type_mismatch", "交易实体类型不匹配"
                )
                expected_type = next(
                    (
                        d
                        for d in (
                            "delivery",
                            "instore",
                            "hotel",
                            "attraction",
                            "flight",
                            "train",
                            "taxi",
                        )
                        if d in name
                    ),
                    None,
                )
                if key == "order_id" and expected_type and order.get("order_type"):
                    require(
                        order["order_type"] == expected_type,
                        "entity_type_mismatch",
                        "订单业务类型不匹配",
                    )
                if name.startswith("pay_"):
                    require(
                        order["status"] == "unpaid", "invalid_order_state", "只可支付待付款交易"
                    )
        if self.is_write(name):
            require(
                not w["pending_calls"],
                "unresolved_write",
                "先收齐在途回执并核验未知写操作，再提交新的写操作",
            )

    def completion_evidence(self, workspace):
        spec = workspace["task"].get("completion")
        if not spec or workspace["pending_calls"] or workspace["pending_memory"] is not None:
            return []
        orders = workspace["orders"]
        created = [o for o in orders.values() if o.get("created_in_task")]
        kind = spec["kind"]
        if kind == "answer":
            # 仅引用可见用户任务，不把它伪装成答案质量或业务执行成功的证据。
            return (
                [workspace["task_origin"]] if not created and workspace.get("task_origin") else []
            )
        if kind in {"purchase", "reserve"} and any(o["status"] == "unpaid" for o in created):
            return []
        ids = spec.get("entity_ids")
        if kind in {"cancel", "modify"} and not ids:
            return []
        selected = [orders.get(i) for i in ids] if ids else created
        if len(selected) < spec.get("count", 1) or any(o is None for o in selected):
            return []
        for order in selected:
            if order.get("verification_required"):
                return []
            if kind == "purchase" and order["status"] not in PAID:
                return []
            if kind == "reserve" and order["status"] not in PAID | {"unconsumed"}:
                return []
            if kind == "cancel" and order["status"] != "cancelled":
                return []
            if kind == "modify" and (
                not order.get("modification_source")
                or not spec.get("expected_fields")
                or order["status"] == "unpaid"
            ):
                return []
            if not self._matches(order, spec.get("expected_fields", {})):
                return []
        return list(dict.fromkeys(ref for o in selected for ref in o["source_refs"]))


class DeliveryDomain(VitaDomain):
    def __init__(self):
        super().__init__("delivery")


def builtin_domains():
    names = ("delivery", "instore", "ota")
    return {
        ",".join(group): VitaDomain(",".join(group))
        for n in (1, 2, 3)
        for group in permutations(names, n)
    }
