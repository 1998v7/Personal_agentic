"""执行循环：事件 → 事实 → 上下文 → 一次模型 → 校验 → 原子提交。"""

import asyncio
import json
from copy import deepcopy
from hashlib import sha256
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from jsonschema import Draft202012Validator, SchemaError, ValidationError

from .context import ContextAssembler
from .domain import DomainAdapter, builtin_domains, require
from .errors import HarnessError
from .model import ActionModel, ModelResponseError, ModelResult
from .protocol import AdvanceInput, AdvanceOutput
from .store import SQLiteStore
from .vitabench import CATALOG, MEMORY_TOOLS


class ReferenceResolver(Protocol):
    """由外层注入已授权材料读取器；默认不接受任意路径/URL读取。"""

    def resolve(self, reference: str) -> object: ...


def fresh_workspace(event: dict, resolver: ReferenceResolver | None) -> dict:
    def resolve(key, default):
        ref_key = {"system_prompt": "system_ref", "tools": "tools_ref", "history": "history_ref"}[
            key
        ]
        if event.get(ref_key) is not None:
            require(
                resolver is not None,
                "reference_unavailable",
                f"{ref_key} 需要配置 ReferenceResolver",
            )
            return resolver.resolve(event[ref_key])
        return event.get(key) if event.get(key) is not None else default

    tools = resolve("tools", [])
    from .protocol import ToolDefinition

    tools = [ToolDefinition.model_validate(t).model_dump(exclude_none=True) for t in tools]
    if event.get("memory_mode", "external") == "external":
        tools = [t for t in tools if t["function"]["name"] not in MEMORY_TOOLS]
    names = [t["function"]["name"] for t in tools]
    require(len(names) == len(set(names)), "duplicate_tool", "工具名必须唯一")
    for tool in tools:
        schema = tool["function"]["parameters"]
        Draft202012Validator.check_schema(schema)

        # 不允许工具 schema 触发外部资源读取。
        def check_refs(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in {"$ref", "$dynamicRef"}:
                        require(
                            isinstance(value, str) and value.startswith("#"),
                            "external_schema_ref",
                            "只支持 schema 内部引用",
                        )
                    check_refs(value)
            elif isinstance(node, list):
                for value in node:
                    check_refs(value)

        check_refs(schema)
    policy, history = resolve("system_prompt", ""), resolve("history", "")
    require(
        isinstance(policy, str) and isinstance(history, str),
        "invalid_reference",
        "系统策略和历史必须是文本",
    )
    return {
        "domain": event["domain"],
        "query": event["query"],
        "system_prompt": policy,
        "history": history,
        "tools": tools,
        "memory_mode": event.get("memory_mode", "external"),
        "lifecycle": "active",
        "turns": 0,
        "max_turns": event.get("max_turns", 100),
        "task": {},
        "task_sources": {},
        "notes": {},
        "facts": {},
        "observations": {},
        "orders": {},
        "pending_calls": {},
        "pending_memory": None,
        "user_memory": [event["memory"]],
        "failed_attempts": [],
        "recent_events": [],
        "last_reply": None,
        "memory_queries_since_external": 0,
    }


class ExecutionHarness:
    def __init__(
        self,
        store: SQLiteStore,
        model: ActionModel,
        domains: dict[str, DomainAdapter] | None = None,
        context: ContextAssembler | None = None,
        resolver: ReferenceResolver | None = None,
        decision_timeout: float = 120,
        max_decision_attempts: int = 3,
    ):
        self.store, self.model = store, model
        self.domains = domains if domains is not None else builtin_domains()
        self.context = context or ContextAssembler()
        self.resolver, self.decision_timeout = resolver, decision_timeout
        self.max_decision_attempts = max_decision_attempts
        require(max_decision_attempts >= 1, "invalid_config", "决策尝试上限必须为正整数")

    async def advance(self, request: AdvanceInput | dict) -> AdvanceOutput:
        request = AdvanceInput.model_validate(request)
        raw = request.model_dump(mode="json")
        workspace, cached = self.store.claim(raw)
        if cached is not None:
            return AdvanceOutput.model_validate(cached)
        attempt_id = None
        event_valid = False
        observed = None
        started = perf_counter()
        try:
            event, eid = raw["event"], request.event_id
            if workspace is None:
                workspace = fresh_workspace(event, self.resolver)
                workspace["task_origin"] = eid
            # 兼容 v0.1 的外卖订单快照；不替旧任务臆造完成条件。
            for oid, order in workspace["orders"].items():
                order.setdefault("entity_id", oid)
                order.setdefault("item_id", oid)
                order.setdefault("kind", "order")
            require(
                workspace["domain"] in self.domains, "unsupported_domain", "尚未注册此领域适配器"
            )
            domain = self.domains[workspace["domain"]]
            already_observed = workspace.pop("_observed_event", None) == eid
            if not already_observed:
                self._observe(workspace, event, eid, domain)
            self._refresh_progress(workspace)
            event_valid = True
            self.store.stage_observation(request.task_id, eid, workspace)
            observed = deepcopy(workspace)
            if event["type"] == "resume":
                workspace["lifecycle"] = "active"
                workspace["max_turns"] = workspace.get("turns", 0) + event["additional_turns"]
            if event["type"] == "terminate":
                workspace["lifecycle"] = "stopped"
                workspace["stop_reason"] = event["reason"]
            if workspace.get("turns", 0) >= workspace.get("max_turns", 100):
                workspace.update(lifecycle="stopped", stop_reason="max_turns")
            if workspace.get("lifecycle") == "stopped":
                return self._commit_reply(
                    raw,
                    workspace,
                    "任务已停止；已发出的工具请求仍需回传结果，停止任务不会撤销业务订单。",
                    "partial",
                )
            workspace["repair_feedback"] = self.store.repair_feedback(request.task_id, eid)
            if len(workspace["repair_feedback"]) >= self.max_decision_attempts:
                return self._commit_reply(
                    raw, workspace, "本轮决策已达到尝试上限，请补充或纠正要求后继续。", "blocked"
                )
            context = self.context.build(workspace, event, eid, domain.decision_schema)
            attempt_id = self.store.begin_attempt(
                request.task_id,
                eid,
                context,
                {
                    "model_adapter": type(self.model).__name__,
                    "configured_model": getattr(self.model, "model", None),
                    "prompt_sha256": sha256(context["system"].encode()).hexdigest(),
                    "decision_schema": domain.decision_schema.get("title"),
                    "vitabench_revision": CATALOG["revision"],
                },
            )
            async with asyncio.timeout(self.decision_timeout):
                decision = await self.model.decide(deepcopy(context))
            metadata = {"latency_ms": round((perf_counter() - started) * 1000, 2)}
            if isinstance(decision, ModelResult):
                metadata.update(decision.metadata)
                decision = decision.content
            self.store.record_decision(attempt_id, decision, metadata)
            if isinstance(decision, str):
                decision = json.loads(decision)
            self._validate(decision, context, workspace, event, domain)
            action = self._apply(decision, workspace, eid, event, domain)
            self._refresh_progress(workspace, action["type"])
            workspace["turns"] = workspace.get("turns", 0) + 1
            result = AdvanceOutput.model_validate(
                {
                    "task_id": request.task_id,
                    "event_id": eid,
                    "rev": request.base_rev + 1,
                    "action": action,
                }
            )
            output = result.model_dump(mode="json", exclude_none=True)
            workspace["recent_events"].append({"event_id": eid, "event": event, "action": action})
            # 全量轨迹在 events；快照只保留近期片段，上一条回复独立保留。
            workspace["recent_events"] = workspace["recent_events"][-12:]
            self.store.commit(raw, workspace, output)
            return result
        except (ValidationError, SchemaError, json.JSONDecodeError) as exc:
            detail = exc.message if isinstance(exc, (ValidationError, SchemaError)) else str(exc)
            error = HarnessError(
                "invalid_decision", "模型 JSON / schema 校验失败：" + detail[:600], 422
            )
            exhausted = self._fail(raw, error, attempt_id, event_valid, observed)
            if exhausted:
                return exhausted
            raise error from exc
        except HarnessError as exc:
            if isinstance(exc, ModelResponseError) and attempt_id is not None:
                self.store.record_decision(attempt_id, exc.result.content, exc.result.metadata)
            exhausted = self._fail(raw, exc, attempt_id, event_valid, observed)
            if exhausted:
                return exhausted
            raise
        except BaseException as exc:
            code = "decision_timeout" if isinstance(exc, TimeoutError) else "decision_interrupted"
            error = HarnessError(code, "本轮未提交，可使用同一事件重试", 503)
            exhausted = self._fail(
                raw,
                error,
                attempt_id,
                event_valid,
                observed,
                interrupted=isinstance(
                    exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)
                ),
            )
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise
            if exhausted:
                return exhausted
            raise error from exc

    def _commit_reply(self, raw, workspace, content, outcome, failed=False):
        action = {
            "type": "reply_user",
            "mode": "final",
            "content": content,
            "outcome": outcome,
            "evidence_refs": [],
        }
        result = AdvanceOutput(
            task_id=raw["task_id"], event_id=raw["event_id"], rev=raw["base_rev"] + 1, action=action
        )
        workspace["last_reply"] = {"event_id": raw["event_id"], "mode": "final", "content": content}
        self.store.commit(
            raw, workspace, result.model_dump(mode="json", exclude_none=True), failed=failed
        )
        return result

    def _fail(self, raw, error, attempt_id, event_valid, observed, interrupted=False):
        exhausted = (
            event_valid
            and observed is not None
            and not interrupted
            and len(self.store.repair_feedback(raw["task_id"], raw["event_id"]))
            + (attempt_id is not None)
            >= self.max_decision_attempts
        )
        self.store.fail(
            raw["task_id"],
            raw["event_id"],
            error.as_dict(),
            attempt_id,
            rejected=not event_valid,
            keep_processing=exhausted,
        )
        failures = self.store.repair_feedback(raw["task_id"], raw["event_id"])
        if exhausted:
            observed["repair_feedback"] = failures
            return self._commit_reply(
                raw,
                observed,
                "本轮模型决策连续失败，已保留收到的事实。请补充或纠正要求后继续。",
                "blocked",
            )
        return None

    def _observe(self, w: dict, event: dict, eid: str, domain: DomainAdapter):
        kind = event["type"]
        if kind != "memory_result":
            w["memory_queries_since_external"] = 0
        if kind == "user_message":
            if event.get("reply_to"):
                require(
                    w["last_reply"] is not None
                    and event["reply_to"] == w["last_reply"]["event_id"],
                    "unknown_reply",
                    "reply_to 未指向最近一次助手回复",
                )
        elif kind == "memory_result":
            pending = w["pending_memory"]
            require(
                pending is not None and pending["request_id"] == event["request_id"],
                "unknown_memory_request",
                "Memory 结果没有对应的请求",
            )
            w["user_memory"].append({**event, "question": pending["question"], "source_event": eid})
            w["pending_memory"] = None
        elif kind == "tool_results":
            ids = [r["call_id"] for r in event["results"]]
            require(len(ids) == len(set(ids)), "duplicate_result", "同一批回执不能重复 call_id")
            for result in event["results"]:
                call = w["pending_calls"].get(result["call_id"])
                require(
                    call is not None and call["name"] == result["name"],
                    "unknown_tool_call",
                    "回执 call_id / name 与未决调用不匹配",
                )
                if result["delivery"] == "unknown":
                    call["delivery"] = "unknown"
                    continue
                source = eid + "/" + result["call_id"]
                status = domain.observe(call, result["content"], w, source)
                if status == "failed":
                    w["failed_attempts"].append(
                        {
                            "attempt": result["call_id"],
                            "name": call["name"],
                            "arguments": call["arguments"],
                            "error": result["content"],
                            "source_refs": [source],
                        }
                    )
                if status == "unknown":
                    call["delivery"] = "unknown"
                    call["last_observation"] = {"source_ref": source, "status": "unknown"}
                else:
                    del w["pending_calls"][result["call_id"]]
            w["pending_calls"] = {
                cid: c for cid, c in w["pending_calls"].items() if not c.get("resolved_by")
            }

    def _validate(self, decision, context, w, event, domain):
        Draft202012Validator(domain.decision_schema).validate(decision)
        if "completion" in decision["task_changes"]:
            old = w["task"].get("completion")
            new = decision["task_changes"]["completion"]
            binding = (
                old is not None
                and "entity_ids" not in old
                and "entity_ids" in new
                and {k: v for k, v in new.items() if k != "entity_ids"} == old
                and all(i in w["orders"] for i in new["entity_ids"])
            )
            require(
                event["type"] in {"start", "user_message"} or binding,
                "completion_scope",
                "完成条件只能依据用户任务改变；工具事件只允许绑定已核实的目标 ID",
            )
        for field, value in decision["task_changes"].items():
            if (
                event["type"] not in {"start", "user_message"}
                and field in w["task"]
                and field != "completion"
            ):
                require(
                    value == w["task"][field],
                    "task_requirement_overwrite",
                    "工具或记忆事件不能覆盖已保存的当前任务要求",
                )
            if value is None:
                require(
                    field in w["task"] and field in event.get("allowed_clears", []),
                    "unconfirmed_clear",
                    "清空字段需外层明确给出 allowed_clears",
                )
        visible = {f["item_id"] for f in context["input"]["candidates"]}
        attempts = {a["attempt"] for a in context["input"]["failed_attempts"]}
        seen = set()
        for note in decision["working_notes"]:
            if note["kind"] == "issue":
                if "note_id" in note:
                    require(
                        "issue:" + note["note_id"] in w["notes"],
                        "unknown_issue",
                        "新问题省略 note_id；更新时复制已存在的问题 ID",
                    )
                if "note_id" not in note:
                    matches = [
                        n["value"]
                        for n in w["notes"].values()
                        if n["value"].get("kind") == "issue"
                        and n["value"].get("topic") == note["topic"]
                        and n["value"].get("status") == "open"
                    ]
                    if note["status"] == "resolved":
                        require(
                            len(matches) == 1, "ambiguous_issue", "关闭问题请提供输入中的 note_id"
                        )
                        note["note_id"] = matches[0].get("note_id", note["topic"])
                    else:
                        note["note_id"] = (
                            "issue_"
                            + sha256((note["topic"] + ":" + note["text"]).encode()).hexdigest()[:16]
                        )
            key = self._note_key(note)
            if note["kind"] == "issue" and key in w["notes"]:
                require(
                    w["notes"][key]["value"]["topic"] == note["topic"],
                    "issue_topic_mismatch",
                    "更新问题时 topic 必须保持一致",
                )
            require(key not in seen, "duplicate_note", "同轮同一笔记只能更新一次")
            seen.add(key)
            if note["kind"] == "candidate":
                require(note["item_id"] in visible, "ungrounded_note", "候选笔记没有可见实体")
            elif note["kind"] == "failed_attempt":
                require(note["attempt"] in attempts, "ungrounded_note", "失败笔记没有已知失败尝试")
            elif note["status"] == "resolved":
                require(
                    key in w["notes"] and w["notes"][key]["value"]["status"] == "open",
                    "unknown_issue",
                    "只能关闭已存在的未决问题",
                )
                require(
                    w["notes"][key]["value"]["topic"] == note["topic"],
                    "issue_topic_mismatch",
                    "更新问题时 topic 必须保持一致",
                )
                # 仅自动认可有确定业务事实的两类关闭；其余继续保留开放问题。
                condition = (
                    note["topic"] == "missing_product_id"
                    and any(f.get("kind") == "product" for f in w["facts"].values())
                    or note["topic"] == "unverified_order"
                    and bool(w["orders"])
                    and not any(domain.is_write(c["name"]) for c in w["pending_calls"].values())
                    or note["topic"] in {"unclear_authorization", "other_requirement"}
                    and event["type"] == "user_message"
                )
                require(condition, "unverified_resolution", "本版尚无足够事实自动确认此问题已解决")
        action = decision["next_action"]
        if action["type"] == "call_tools":
            schemas = {t["function"]["name"]: t["function"]["parameters"] for t in w["tools"]}
            require(
                len(action["calls"]) == 1
                or not any(domain.is_write(c["name"]) for c in action["calls"]),
                "dependent_tool_batch",
                "写操作每轮独立提交，读取可并行",
            )
            for call in action["calls"]:
                require(call["name"] in schemas, "unknown_tool", "工具不在本轮允许列表中")
                Draft202012Validator(schemas[call["name"]]).validate(call["arguments"])
                domain.validate_call(call, w, visible)
        elif action["type"] == "query_memory":
            require(w["pending_memory"] is None, "memory_pending", "已有未返回的 Memory 请求")
            require(
                context["input"]["memory_query_budget_remaining"] > 0,
                "memory_query_limit",
                "连续 Memory 查询已达上限",
            )
        elif action["mode"] == "final" and action["outcome"] == "completed":
            prospective = {**w, "task": {**w["task"], **decision["task_changes"]}}
            require(
                bool(domain.completion_evidence(prospective))
                and not w["pending_calls"]
                and w["pending_memory"] is None,
                "unverified_completion",
                "当前证据不足以确认任务完成",
            )

    @staticmethod
    def _refresh_progress(w, next_action_type=None):
        stores = sum(f["kind"] == "store" for f in w["facts"].values())
        products = sum(f["kind"] == "product" for f in w["facts"].values())
        phase = "gathering_requirements"
        if stores:
            phase = "store_candidates_available"
        if products:
            phase = "product_candidates_available"
        if w["orders"]:
            phase = "order_observed"
        if any(o["status"] in {"paid", "in-progress", "delivered"} for o in w["orders"].values()):
            phase = "payment_confirmed"
        w["progress"] = {
            "phase": phase,
            "store_count": stores,
            "product_count": products,
            "failed_tool_calls": len(w["failed_attempts"]),
            "pending_tool_calls": len(w["pending_calls"]),
            "waiting_for_memory": w["pending_memory"] is not None,
            "entity_counts": {
                kind: sum(f["kind"] == kind for f in w["facts"].values())
                for kind in {f["kind"] for f in w["facts"].values()}
            },
        }
        if next_action_type:
            w["progress"]["next_action_type"] = next_action_type

    @staticmethod
    def _note_key(note):
        return (
            note["kind"]
            + ":"
            + str(note.get("item_id", note.get("attempt", note.get("note_id", note.get("topic")))))
        )

    def _apply(self, decision, w, eid, event, domain):
        for field, value in decision["task_changes"].items():
            if value is None:
                w["task"].pop(field, None)
            else:
                w["task"][field] = value
            w["task_sources"][field] = {"proposal_event": eid, "provenance": "model_interpretation"}
        for note in decision["working_notes"]:
            refs = []
            if note["kind"] == "candidate":
                refs = w["facts"][note["item_id"]]["source_refs"]
            elif note["kind"] == "failed_attempt":
                refs = next(
                    a["source_refs"]
                    for a in w["failed_attempts"]
                    if a["attempt"] == note["attempt"]
                )
            elif note["kind"] == "issue":
                refs = [eid]
            w["notes"][self._note_key(note)] = {
                "value": note,
                "proposal_event": eid,
                "source_refs": refs,
                "provenance": "model_judgment",
            }
        action = deepcopy(decision["next_action"])
        if action["type"] == "call_tools":
            for call in action["calls"]:
                call["call_id"] = "call_" + uuid4().hex
                w["pending_calls"][call["call_id"]] = {**call, "delivery": "pending"}
        elif action["type"] == "query_memory":
            action["request_id"] = "mem_" + uuid4().hex
            w["pending_memory"] = deepcopy(action)
            w["memory_queries_since_external"] += 1
        else:
            w["last_reply"] = {
                "event_id": eid,
                "mode": action["mode"],
                "content": action["content"],
            }
            if action["mode"] == "final" and action["outcome"] == "completed":
                action["evidence_refs"] = domain.completion_evidence(w)
        return action
