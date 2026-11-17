"""规则式上下文组装，不额外调用模型。预算单位为 UTF-8 字节，不冒充 tokenizer。"""

from copy import deepcopy
from importlib.resources import files

from .errors import HarnessError
from .store import dumps
from .vitabench import TRANSACTIONS, fields, parse_records, product_parent, response_failure


def compact(value):
    """模型输入不重复携带已结构化实体的 raw；完整原文仍在事件与工作区。"""
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k != "raw"}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def represented_records(content, source, facts):
    """Only elide receipts that can be reconstructed from visible candidate fields.

    The parser intentionally accepts imperfect/mixed receipts. Parsing alone is
    therefore insufficient: every record must round-trip exactly, all intervening
    text must be whitespace, and every field and its source must still be visible.
    Transaction receipts and unstructured results retain their original content.
    """
    if not isinstance(content, str) or response_failure(content):
        return 0
    records = parse_records(content)
    if not records:
        return 0
    cursor = 0
    for record in records:
        raw, kind = record["raw"], record["kind"]
        if kind in TRANSACTIONS or not raw.endswith(")"):
            return 0
        offset = content.find(raw, cursor)
        if offset < 0 or content[cursor:offset].strip():
            return 0
        values = fields(raw[len(record["record_type"]) + 1 : -1])
        reconstructed = (
            record["record_type"]
            + "("
            + ", ".join(f"{key}={value}" for key, value in values.items())
            + ")"
        )
        if reconstructed != raw:
            return 0
        fact = facts.get(record["item_id"])
        if (
            fact is None
            or fact.get("kind") != kind
            or fact.get("record_type") != record["record_type"]
            or source not in fact["source_refs"]
            or any(key not in fact or fact[key] != value for key, value in values.items())
        ):
            return 0
        visible_text = (
            record["record_type"] + "(" + ", ".join(f"{key}={fact[key]}" for key in values) + ")"
        )
        if visible_text != raw:
            return 0
        cursor = offset + len(raw)
    # Keep search ordering and duplicate occurrences, not just the set of IDs.
    # 商品外键另外登记的父实体不是这条回执中的独立记录，不参与原始记录顺序比较。
    record_types = {record["record_type"] for record in records}
    if [
        key
        for key, fact in facts.items()
        if source in fact["source_refs"] and fact.get("record_type") in record_types
    ] != [record["item_id"] for record in records]:
        return 0
    return len(records) if not content[cursor:].strip() else 0


def event_view(event, event_id, workspace, facts):
    """Project an event for the model; the persisted event remains untouched."""
    view = deepcopy(event)
    if event["type"] == "start":
        # These fields already have canonical locations in the model context.
        for key in (
            "system_prompt",
            "system_ref",
            "tools",
            "tools_ref",
            "memory",
            "history",
            "history_ref",
        ):
            view.pop(key, None)
        if view.get("query") == workspace["query"]:
            view.pop("query", None)
    elif event["type"] == "memory_result":
        if any(
            memory.get("source_event") == event_id and memory.get("context") == event.get("context")
            for memory in workspace["user_memory"]
        ):
            view.pop("context", None)
            view["context_in"] = "user_memory"
            view["source_ref"] = event_id
    elif event["type"] == "tool_results":
        for result in view["results"]:
            source = event_id + "/" + result["call_id"]
            if result["delivery"] != "received":
                continue
            count = represented_records(result.get("content"), source, facts)
            if count:
                result.pop("content")
                result.update(content_in="candidates", content_ref=source, record_count=count)
    return view


class ContextAssembler:
    def __init__(self, max_bytes: int = 240_000):
        self.max_bytes = max_bytes
        self.prompt = files("imact").joinpath("prompts/action.txt").read_text()

    def build(self, workspace: dict, event: dict, event_id: str, schema: dict) -> dict:
        notes = [
            n["value"]
            for n in workspace["notes"].values()
            if n["value"].get("status") != "resolved"
        ]
        facts = list(workspace["facts"].values())
        candidates = compact(facts)
        visible_facts = {fact["item_id"]: fact for fact in candidates}
        body = {
            "original_query": workspace["query"],
            "current_task": workspace["task"],
            "business_state": {
                "progress": workspace["progress"],
                "orders": compact(workspace["orders"]),
                "pending_calls": workspace["pending_calls"],
                "pending_memory": workspace["pending_memory"],
            },
            "latest_event": event_view(event, event_id, workspace, visible_facts),
            "previous_reply": workspace["last_reply"],
            "recent_events": [
                {
                    **entry,
                    "event": event_view(
                        entry["event"], entry["event_id"], workspace, visible_facts
                    ),
                }
                for entry in workspace["recent_events"][-6:]
            ],
            "visible_history": workspace["history"],
            "user_memory": workspace["user_memory"],
            "candidates": candidates,
            "repair_feedback": workspace.get("repair_feedback", []),
            "latest_observations": [
                {k: x for k, x in v.items() if k != "content"}
                for ref, v in workspace.get("observations", {}).items()
                if ref.startswith(event_id + "/")
            ],
            "failed_attempts": workspace["failed_attempts"],
            "active_notes": notes,
            "available_tools": workspace["tools"],
            "memory_query_budget_remaining": max(0, 2 - workspace["memory_queries_since_external"]),
            "output_schema": schema,
            "archived_evidence": [],
        }
        packet = deepcopy(
            {"system": self.prompt, "task_policy": workspace["system_prompt"], "input": body}
        )
        selected = packet["input"]
        omitted = {"recent_events": 0, "candidates": 0, "user_memory": 0, "failed_attempts": 0}
        # 最新事件/前一句/状态/活跃笔记/Memory/工具 schema 不被截断。
        while len(dumps(packet).encode()) > self.max_bytes and selected["recent_events"]:
            selected["recent_events"].pop(0)
            omitted["recent_events"] += 1
        required_ids = {n.get("item_id") for n in notes if n.get("assessment") != "exclude"}
        required_ids.update(
            f["item_id"]
            for f in facts
            if any(ref.startswith(event_id + "/") for ref in f["source_refs"])
        )
        # 保留活跃商品时也保留其已登记的父实体，避免裁剪后再次拒绝合法详情查询。
        required_ids.update(
            parent[2]
            for f in facts
            if f["item_id"] in required_ids and (parent := product_parent(f)) is not None
        )
        while len(dumps(packet).encode()) > self.max_bytes:
            referenced_parents = {
                parent[2]
                for candidate in selected["candidates"]
                if (parent := product_parent(candidate)) is not None
            }
            index = next(
                (
                    i
                    for i, c in enumerate(selected["candidates"])
                    if c["item_id"] not in required_ids | referenced_parents
                ),
                None,
            )
            if index is None:
                # 只在预算压力下移出老材料；保留首轮 Memory、最近 Memory 和未决状态。
                if len(selected["user_memory"]) > 2:
                    removed = selected["user_memory"].pop(1)
                    selected["archived_evidence"].append(
                        {"type": "memory", "ref": removed.get("source_event")}
                    )
                    omitted["user_memory"] += 1
                    continue
                pinned_attempts = {n.get("attempt") for n in notes}
                removable = next(
                    (
                        i
                        for i, a in enumerate(selected["failed_attempts"][:-2])
                        if a["attempt"] not in pinned_attempts
                    ),
                    None,
                )
                if removable is not None:
                    removed = selected["failed_attempts"].pop(removable)
                    selected["archived_evidence"].append(
                        {"type": "failure", "refs": removed["source_refs"]}
                    )
                    omitted["failed_attempts"] += 1
                    continue
                raise HarnessError(
                    "context_overflow",
                    "必需上下文超过字节预算，请扩大预算或在上游提供更精确材料",
                    413,
                )
            selected["candidates"].pop(index)
            omitted["candidates"] += 1
        return {
            **packet,
            "manifest": {
                "input_bytes": len(dumps(packet).encode()),
                "budget_bytes": self.max_bytes,
                "omitted": omitted,
                "evidence_refs": sorted(
                    {
                        r
                        for f in facts
                        if f["item_id"] in {c["item_id"] for c in selected["candidates"]}
                        for r in f["source_refs"]
                    }
                ),
            },
        }
