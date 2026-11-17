import json
import tempfile
import unittest
from pathlib import Path

from IMAct.benchmarks.vitabench.episode_report import (
    audit_context_injection,
    build_episode,
    difference,
    export_episode,
)
from IMAct.exec_layer import build_runtime
from IMAct.models.base import ChatResponse
from IMAct.trace import TraceRecorder

from tests.exec_helpers import DemoProvider, make_cfg, no_sleep, run_demo_episode


class EpisodeReportTests(unittest.TestCase):
    def test_context_injection_checks_serialized_message_and_resolved_note_filter(self):
        active = {"kind": "issue", "note_id": "n1", "status": "open", "text": "待确认"}
        resolved = {"kind": "issue", "note_id": "n2", "status": "resolved", "text": "已确认"}
        state = {"task": {"goal": "午餐"}, "notes": [{"value": active}, {"value": resolved}]}
        body = {"current_task": state["task"], "active_notes": [active]}
        records = [
            {
                "seq": 1,
                "stage": "workspace_committed",
                "correlation": {"task_id": "t", "event_id": "e1"},
                "data": {"rev": 1, "workspace": state},
            },
            {
                "seq": 2,
                "stage": "action_input",
                "correlation": {"task_id": "t", "event_id": "e2"},
                "data": {
                    "context": {"input": body},
                    "messages_before_provider": [{"role": "user", "content": json.dumps(body)}],
                },
            },
        ]
        verified = audit_context_injection(records)[0]
        self.assertTrue(verified["serialized_fields_match_context"])
        self.assertTrue(verified["task_matches_committed_state"])
        self.assertTrue(verified["notes_match_committed_active_notes"])
        self.assertEqual(verified["source_commit_seq"], 1)
        self.assertEqual(verified["resolved_notes_not_in_active_notes"], 1)
        # 仅在数据库/上下文里存在,而序列化时丢掉笔记,不能报告已传入模型。
        records[1]["data"]["messages_before_provider"][0]["content"] = json.dumps(
            {**body, "active_notes": []}
        )
        self.assertFalse(audit_context_injection(records)[0]["serialized_fields_match_context"])

    def test_diff_distinguishes_null_missing_and_delete(self):
        changes = difference({"old": 1, "same": 2}, {"new": None, "same": 2})
        self.assertEqual([x["field"] for x in changes], ["new", "old"])
        self.assertFalse(changes[0]["before_present"])
        self.assertTrue(changes[0]["after_present"])
        self.assertIsNone(changes[0]["after"])
        self.assertFalse(changes[1]["after_present"])

    def test_failed_proposal_has_no_committed_changes_and_complete_trace_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            provider = DemoProvider(faults=[ChatResponse(content="not-json", finish_reason="stop")])
            rt = build_runtime(
                make_cfg(directory),
                provider=provider,
                sleep=no_sleep,
                trace=TraceRecorder(out / "trace.jsonl"),
            )
            try:
                run_demo_episode(rt, "episode")
            finally:
                rt.close()
            episode = export_episode(out)
            failed = episode["rounds"][0]
            self.assertFalse(failed["accepted"])
            self.assertFalse(failed["committed"])
            self.assertEqual(failed["task_changes"], [])
            self.assertEqual(failed["note_changes"], [])
            self.assertTrue(episode["completion_verified_by_harness"])
            self.assertEqual(
                episode["snapshot_coverage"]["committed_events"],
                episode["snapshot_coverage"]["snapshots"],
            )
            self.assertTrue(any(r["task_changes"] for r in episode["rounds"]))
            self.assertTrue(any(r["note_changes"] for r in episode["rounds"]))
            self.assertTrue((out / "episode.html").is_file())
            injection = json.loads((out / "context-injection-verification.json").read_text())
            self.assertTrue(all(row["serialized_fields_match_context"] for row in injection))
            self.assertTrue(all(row["task_matches_committed_state"] for row in injection))
            self.assertTrue(all(row["notes_match_committed_active_notes"] for row in injection))
            payload = json.loads((out / "full_trace.json").read_text())
            # A missing response from a timed-out attempt must not shift later raw responses.
            payload["records"] = [
                r
                for r in payload["records"]
                if not (r["stage"] == "action_output" and r["data"]["content"] == "not-json")
            ]
            replay = build_episode(payload)
            self.assertIsNone(replay["rounds"][0]["reasoning"])
            self.assertEqual(replay["rounds"][1]["raw_output"], episode["rounds"][1]["raw_output"])
