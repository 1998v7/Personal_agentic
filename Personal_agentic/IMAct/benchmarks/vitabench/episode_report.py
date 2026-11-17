"""把真实 trace 中的提案、校验与提交快照关联成逐轮可视化数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .trace_report import dump, export_trace


def difference(before, after):
    """保留新增、修改、删除以及缺失和 null 的区别。"""
    changes = []
    for key in sorted(before.keys() | after.keys()):
        if key not in before or key not in after or before[key] != after[key]:
            changes.append(
                {
                    "field": key,
                    "before_present": key in before,
                    "after_present": key in after,
                    "before": before.get(key),
                    "after": after.get(key),
                }
            )
    return changes


def notes_by_key(notes):
    if isinstance(notes, dict):
        return notes
    result = {}
    for entry in notes or []:
        note = entry["value"]
        key = (
            note["kind"]
            + ":"
            + str(note.get("item_id", note.get("attempt", note.get("note_id", note.get("topic")))))
        )
        result[key] = entry
    return result


def parse_decision(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def audit_context_injection(records):
    """核对提交状态、Harness 上下文和序列化模型消息,不推断模型是否采用。"""
    previous, result = {}, []
    for record in records:
        task_id = record["correlation"].get("task_id")
        if record["stage"] == "workspace_committed":
            previous[task_id] = record
        if record["stage"] != "action_input":
            continue
        data = record["data"]
        context = data["context"]["input"]
        messages = data.get("messages_before_provider", [])
        message_index = next(
            (i for i in reversed(range(len(messages))) if messages[i]["role"] == "user"),
            None,
        )
        try:
            body = (
                json.loads(messages[message_index]["content"]) if message_index is not None else {}
            )
        except (ValueError, TypeError):
            body = {}
        if not isinstance(body, dict):
            body = {}
        source = previous.get(task_id)
        workspace = source["data"]["workspace"] if source else {}
        source_available = (
            source is not None or context.get("latest_event", {}).get("type") == "start"
        )
        notes = list(notes_by_key(workspace.get("notes")).values())
        active = [entry["value"] for entry in notes if entry["value"].get("status") != "resolved"]
        result.append(
            {
                "input_seq": record["seq"],
                "event_id": record["correlation"].get("event_id"),
                "user_message_index": message_index,
                "task_state_path": "messages[user].content(JSON).current_task",
                "working_memory_path": "messages[user].content(JSON).active_notes",
                "serialized_fields_match_context": all(
                    key in body and key in context and body[key] == context[key]
                    for key in ("current_task", "active_notes")
                ),
                "source_commit_seq": source["seq"] if source else None,
                "source_rev": source["data"]["rev"]
                if source
                else (0 if source_available else None),
                "task_matches_committed_state": context.get("current_task")
                == workspace.get("task", {})
                if source_available
                else None,
                "notes_match_committed_active_notes": context.get("active_notes") == active
                if source_available
                else None,
                "task_field_count": len(context.get("current_task", {})),
                "active_note_count": len(context.get("active_notes", [])),
                "resolved_notes_not_in_active_notes": sum(
                    entry["value"].get("status") == "resolved" for entry in notes
                ),
            }
        )
    return result


def build_episode(payload):
    records, database = payload["records"], payload["sqlite"]
    snapshots = {
        (r["correlation"].get("task_id"), r["correlation"].get("event_id")): r
        for r in records
        if r["stage"] == "workspace_committed"
    }
    events = {(r["task_id"], r["event_id"]): r for r in database["events"]}
    outputs = []
    for record in records:
        if record["stage"] == "action_input":
            outputs.append(None)
        elif record["stage"] in {"action_output", "action_error"} and outputs:
            outputs[-1] = record
    previous = {}
    rounds = []
    attempts = database["attempts"]
    for index, attempt in enumerate(attempts):
        key = attempt["task_id"], attempt["event_id"]
        event = events[key]
        raw = outputs[index] if index < len(outputs) else None
        # Only the successful attempt or final blocked fallback owns the commit.
        last_for_event = not any(
            (a["task_id"], a["event_id"]) == key for a in attempts[index + 1 :]
        )
        snapshot = snapshots.get(key) if last_for_event else None
        before = previous.get(attempt["task_id"], {})
        after = snapshot["data"]["workspace"] if snapshot else before
        decision = parse_decision(attempt["decision"])
        accepted = (
            attempt["error"] is None and event["status"] == "committed" and snapshot is not None
        )
        incoming = event["input"]["event"]
        if incoming["type"] == "start":
            incoming = {"type": "start", "query": incoming["query"]}
        context = (attempt.get("context") or {}).get("input", {})
        rounds.append(
            {
                "index": index + 1,
                "attempt_id": attempt["id"],
                "event_id": attempt["event_id"],
                "time": raw["time"] if raw else attempt["created_at"],
                "incoming": incoming,
                "accepted": accepted,
                "error": attempt["error"],
                "decision": decision,
                "raw_output": attempt["decision"],
                "reasoning": raw["data"].get("reasoning") if raw else None,
                "metadata": attempt.get("metadata"),
                "candidate_count": len(context.get("candidates", [])),
                "memory": context.get("user_memory", []),
                "repair_feedback": context.get("repair_feedback", []),
                "task_changes": difference(before.get("task", {}), after.get("task", {}))
                if snapshot
                else [],
                "note_changes": difference(
                    notes_by_key(before.get("notes")), notes_by_key(after.get("notes"))
                )
                if snapshot
                else [],
                "committed": snapshot is not None,
                "committed_rev": snapshot["data"]["rev"] if snapshot else None,
                "committed_action": event["output"]["action"] if snapshot else None,
                "workspace": after if snapshot else None,
            }
        )
        if snapshot:
            previous[attempt["task_id"]] = after
    business_events = [
        e
        for e in database["events"]
        if e.get("output") and e["input"]["event"]["type"] != "terminate"
    ]
    completed = [
        e
        for e in business_events[-1:]
        if e["output"].get("action", {}).get("outcome") == "completed"
    ]
    final = payload["summary"].get("final_workspaces", [])
    return {
        "manifest": payload["manifest"],
        "summary": payload["summary"],
        "memory_events": [r for r in records if r["stage"].startswith("memory_")],
        "user_messages": [r for r in records if r["stage"] == "user_reply"],
        "rounds": rounds,
        "completion_verified_by_harness": bool(completed),
        "completion_actions": [e["output"]["action"] for e in completed],
        "final_workspaces": final,
        "snapshot_coverage": {
            "committed_events": sum(e["status"] == "committed" for e in database["events"]),
            "snapshots": len(snapshots),
        },
    }


def export_episode(out, *, refresh=True):
    out = Path(out)
    if refresh:
        export_trace(out)
    payload = json.loads((out / "full_trace.json").read_text())
    episode = build_episode(payload)
    dump(out / "episode.json", episode)
    context_injection = audit_context_injection(payload["records"])
    dump(out / "context-injection-verification.json", context_injection)
    template = Path(__file__).with_name("episode_view.html").read_text()
    data = json.dumps(episode, ensure_ascii=False).replace("<", "\\u003c")
    fragment = template.replace("__EPISODE_DATA__", data)
    (out / "episode-fragment.html").write_text(fragment)
    # Standalone viewer uses the same data and interaction as the inline fragment.
    style = Path(__file__).with_name("episode_standalone.css").read_text()
    (out / "rounds.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>IMAct 任务逐轮回放</title><style>"
        + style
        + "</style><body><main>"
        + fragment
        + "</main></body></html>"
    )
    chain = {
        "manifest": payload["manifest"],
        "summary": payload["summary"],
        "records": payload["records"],
        "attempts": [
            {"id": a["id"], "event_id": a["event_id"], "error": a["error"]}
            for a in payload["sqlite"]["attempts"]
        ],
        "completed": episode["completion_verified_by_harness"],
        "completion_actions": episode["completion_actions"],
        "context_injection": context_injection,
    }
    full_template = Path(__file__).with_name("full_chain_view.html").read_text()
    (out / "episode.html").write_text(
        full_template.replace(
            "__CHAIN_DATA__", json.dumps(chain, ensure_ascii=False).replace("<", "\\u003c")
        )
    )
    return episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    args = parser.parse_args()
    episode = export_episode(args.out)
    print(
        json.dumps(
            {
                "rounds": len(episode["rounds"]),
                "completed": episode["completion_verified_by_harness"],
                "snapshot_coverage": episode["snapshot_coverage"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
