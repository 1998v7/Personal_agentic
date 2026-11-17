import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from tests.exec_helpers import DemoProvider, make_cfg, no_sleep, run_demo_episode
from IMAct.exec_layer import build_runtime
from IMAct.exec_layer.memory_mps import MpsMemoryClient, MpsMemoryConfig
from IMAct.models.base import ChatResponse
from IMAct.trace import TraceRecorder
from IMAct.benchmarks.vitabench.diagnose import select_cases, run_case
from IMAct.benchmarks.vitabench.trace_report import export_trace


class TraceTests(unittest.TestCase):
    def test_user_completion_guard_is_opt_in_and_preserves_base_prompt(self):
        from IMAct.benchmarks.vitabench.data_model.personalization_task import SubTask
        from IMAct.benchmarks.vitabench.personalization import COMPLETION_GUARD, user_system_prompt

        st = SubTask(subtask_id="test", domain="ota", instruction="订两晚房", environment={})
        original = user_system_prompt(st, "测试用户", "chinese")
        guarded = user_system_prompt(st, "测试用户", "chinese", completion_guard=True)
        self.assertNotIn(COMPLETION_GUARD, original)
        self.assertIn("订两晚房", original)  # 指令进了 prompt
        self.assertEqual(guarded, original + "\n\n" + COMPLETION_GUARD)

    def test_diagnostic_initialization_preserves_time_and_never_evaluates(self):
        from IMAct.models.catalog import ModelCatalog
        from IMAct.benchmarks.vitabench.data_model.personalization_task import PersonalizationTask, SubTask
        from IMAct.benchmarks.vitabench.personalization import PersonalizationRunner, SubtaskResult
        from IMAct.harness.conversation import ConversationResult, TerminationReason

        st = SubTask(subtask_id="test-subtask", domain="ota", instruction="订酒店",
                     environment={"time": "2026-03-28 10:00:00"})
        task = PersonalizationTask(id="test", user_id="test", user_profile={"user_id": "test"}, subtasks=[st])
        cfg = make_cfg(model={"provider": "openai_compatible", "model": "q", "base_url": "https://unused"})
        catalog = ModelCatalog.model_validate({"roles": {"action": "q", "user": "q", "judge": "q"}, "models": {"q": cfg.model.model_dump()}})
        result = SubtaskResult("test-subtask", 0, ConversationResult(messages=[], termination_reason=TerminationReason.USER_STOP, duration=0.0))
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(PersonalizationRunner, "run_subtask", return_value=result), \
             patch.object(PersonalizationRunner, "evaluate_subtask", side_effect=AssertionError("must not score")):
            audit = run_case(task, st, 0, cfg=cfg, catalog=catalog, out=Path(temp)/"case", max_steps=3, seed=300)
            self.assertEqual(audit["status"], "conversation_finished")
            self.assertIsNone(audit["reward"])
            self.assertEqual(st.environment["time"], "2026-03-28 10:00:00")

    def test_complete_trace_includes_failed_decision_memory_and_full_context(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            trace = TraceRecorder(out / "trace.jsonl")
            memory_text = "</script><img data-test>" + "完整记忆材料。" * 1600
            client = MpsMemoryClient(MpsMemoryConfig(url="https://memory.test", api_key_env=None), api_key="SECRET-NEVER-LOG",
                                     client=httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"context": memory_text}))), trace=trace)
            provider = DemoProvider(faults=[ChatResponse(content="not-json", finish_reason="stop", raw={"id": "failed-but-received"})])
            cfg = make_cfg(directory)
            cfg.exec.db_path = str(out / "exec.sqlite3")
            rt = build_runtime(cfg, provider=provider, memory_client=client, sleep=no_sleep, trace=trace)
            try:
                actions = run_demo_episode(rt, "trace-task", metadata={"user_id": "demo", "query_id": "q", "timestamp": "2026-01-01"})
            finally:
                rt.close()
            self.assertEqual(actions[-1].outcome, "completed")
            records = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
            self.assertEqual([r["seq"] for r in records], list(range(1, len(records) + 1)))
            inputs = [r for r in records if r["stage"] == "action_input"]
            outputs = [r for r in records if r["stage"] == "action_output"]
            self.assertEqual(len(inputs), len(outputs))
            self.assertEqual(outputs[0]["data"]["content"], "not-json")
            self.assertTrue(inputs[0]["data"]["context"]["system"])
            self.assertEqual(inputs[0]["correlation"]["event_id"], "e1")
            self.assertTrue(any(r["stage"] == "harness_error" for r in records))
            snapshots = [r for r in records if r["stage"] == "workspace_committed"]
            commits = [r for r in records if r["stage"] == "harness_result"]
            self.assertEqual(len(snapshots), len(commits))
            self.assertEqual([r["data"]["rev"] for r in snapshots],
                             [r["data"]["output"]["rev"] for r in commits])
            self.assertTrue(all("notes" in r["data"]["workspace"] for r in snapshots))
            self.assertTrue(snapshots[-1]["data"]["workspace"]["orders"])
            first_error = next(r["seq"] for r in records if r["stage"] == "harness_error")
            self.assertFalse(any(r["seq"] < first_error for r in snapshots))
            self.assertTrue(any(r["stage"] == "memory_response" and memory_text in json.loads(r["data"]["body"])["context"] for r in records))
            summary = export_trace(out)
            self.assertEqual(summary["attempts"], len(inputs))
            self.assertEqual(summary["model_outputs"], len(outputs))
            self.assertEqual(len(summary["validation_errors"]), 1)
            self.assertEqual(summary["memory_calls"], 1)
            self.assertNotIn("SECRET-NEVER-LOG", (out / "full_trace.json").read_text())
            self.assertNotIn("<img data-test>", (out / "trace.html").read_text())
            self.assertEqual(summary["sqlite_integrity_check"], "ok")

    def test_select_subtasks_preserves_original_position_and_rejects_typos(self):
        task = SimpleNamespace(subtasks=[SimpleNamespace(subtask_id="a"), SimpleNamespace(subtask_id="b")])
        cases = select_cases([task], ["b"])
        self.assertEqual(cases[0][2], 1)
        self.assertEqual(len(task.subtasks), 2)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            select_cases([task], ["missing"])
        with self.assertRaisesRegex(ValueError, "unique"):
            select_cases([task], ["b", "b"])
