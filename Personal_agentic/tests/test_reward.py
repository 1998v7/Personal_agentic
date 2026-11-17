import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.exec_helpers import DemoProvider
from IMAct.benchmarks.vitabench.data_model.personalization_task import SubTask
from IMAct.benchmarks.vitabench.diagnose import grade_existing_case
from IMAct.benchmarks.vitabench.reward import grade_simulation
from IMAct.evaluation.base import JudgeError
from IMAct.evaluation.vitabench import apply_judgement
from IMAct.models.base import ChatResponse
from IMAct.models.catalog import ModelCatalog
from IMAct.trace import TraceRecorder

PROVIDER = "IMAct.harness.llm.create_provider"


def fixtures():
    catalog = ModelCatalog.model_validate({"roles": {"action": "q", "user": "q", "judge": "j"},
        "models": {"q": {"model": "unused"}, "j": {"model": "DeepSeek-V4.1-Flash-1017", "base_url": "https://judge.invalid", "api_key": "secret-must-not-leak", "params": {"max_tokens": 12000}}}})
    task = types.SimpleNamespace(id="t", user_id="u", user_profile={})
    subtask = SubTask(subtask_id="s", domain="delivery", instruction="点午餐", environment={"time": "2026-03-02 11:35:00"},
                      evaluation_criteria={"overall_rubrics": ["含肉", "配送及时"]})
    simulation = {"subtask_id": "s", "duration": 1, "termination_reason": "user_stop", "states": {"old_states": [], "new_states": []},
                  "messages": [{"role": "user", "content": f"消息{i}"} for i in range(20)]}
    return catalog, task, subtask, simulation


def judge_rows():
    return [{"rubric_idx": "rubric_0", "justification": "工具回执支持", "meetExpectation": True},
            {"rubric_idx": "rubric_1", "justification": "尚未满足", "meetExpectation": False}]


class Judge(DemoProvider):
    async def chat(self, request):
        self.requests.append(request)
        return ChatResponse(content=json.dumps(judge_rows(), ensure_ascii=False), model="DeepSeek-V4.1-Flash-1017",
                            finish_reason="stop", usage={"prompt_tokens": 10}, raw={"id": f"judge-{len(self.requests)}"})


class RewardTests(unittest.TestCase):
    def test_real_evaluator_windows_binary_reward_and_api_trace(self):
        catalog, task, subtask, simulation = fixtures()
        provider = Judge()
        with tempfile.TemporaryDirectory() as tmp, patch(PROVIDER, return_value=provider):
            out = Path(tmp)
            result = grade_simulation(task, subtask, simulation, catalog=catalog, out=out, trace=TraceRecorder(out / "trace.jsonl"))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["reward"], 0.0)
            self.assertEqual(result["rubric_pass_rate"], 0.5)
            self.assertEqual(result["window_count"], 3)
            self.assertTrue(result["judge_called"])
            self.assertEqual(len(provider.requests), 3)
            records = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
            responses = [r for r in records if r["stage"] == "judge_response"]
            self.assertEqual(len(responses), 3)
            self.assertEqual(responses[-1]["data"]["request_id"], "judge-3")
            self.assertTrue(all(r["correlation"]["query_id"] == "s" for r in responses))
            self.assertNotIn("secret-must-not-leak", (out / "trace.jsonl").read_text())

    def test_judge_transport_failure_is_null_not_zero(self):
        catalog, task, subtask, simulation = fixtures()

        class Failing(Judge):
            async def chat(self, request):
                raise RuntimeError("API unavailable")

        with tempfile.TemporaryDirectory() as tmp, patch(PROVIDER, return_value=Failing()):
            out = Path(tmp)
            result = grade_simulation(task, subtask, simulation, catalog=catalog, out=out, trace=TraceRecorder(out / "trace.jsonl"))
            self.assertEqual(result["status"], "error")
            self.assertIsNone(result["reward"])
            self.assertIn("API unavailable", result["error"])

    def test_malformed_judge_response_is_an_error_and_keeps_the_raw_reply(self):
        catalog, task, subtask, simulation = fixtures()

        class Sloppy(Judge):
            async def chat(self, request):
                self.requests.append(request)
                rows = [dict(judge_rows()[0], meetExpectation="true"), judge_rows()[1]]  # 布尔被写成了字符串
                return ChatResponse(content=json.dumps(rows), finish_reason="stop", usage={}, raw={"id": "sloppy"})

        with tempfile.TemporaryDirectory() as tmp, patch(PROVIDER, return_value=Sloppy()):
            out = Path(tmp)
            result = grade_simulation(task, subtask, simulation, catalog=catalog, out=out, trace=TraceRecorder(out / "trace.jsonl"))
            self.assertEqual(result["status"], "error")
            self.assertIsNone(result["reward"])
            self.assertIn("meetExpectation", result["error"])
            raw = json.loads((out / "reward_info.raw.json").read_text())
            self.assertEqual(len(raw["window_evaluations"]), 1)  # 出错窗口的原始响应留下来排查

    def test_incomplete_or_mistyped_judgements_are_rejected(self):
        states = {f"rubric_{i}": {"rubric": r, "justification": "", "meetExpectation": False} for i, r in enumerate(["含肉", "配送及时"])}
        for rows in [[], judge_rows()[:1], judge_rows() + judge_rows()[:1],
                     [dict(judge_rows()[0], meetExpectation="false"), judge_rows()[1]],
                     [dict(judge_rows()[0], rubric_idx="rubric_9"), judge_rows()[1]]]:
            with self.subTest(rows=rows):
                with self.assertRaises(JudgeError):
                    apply_judgement(states, json.dumps(rows))

    def test_valid_judgement_updates_states_without_mutating_the_input(self):
        states = {f"rubric_{i}": {"rubric": r, "justification": "Not evaluated yet", "meetExpectation": False} for i, r in enumerate(["含肉", "配送及时"])}
        updated = apply_judgement(states, json.dumps(judge_rows()))
        self.assertTrue(updated["rubric_0"]["meetExpectation"])
        self.assertEqual(updated["rubric_1"]["justification"], "尚未满足")
        self.assertFalse(states["rubric_0"]["meetExpectation"])

    def test_regrading_preserves_source_and_changes_only_evaluation_outputs(self):
        catalog, task, subtask, simulation = fixtures()
        with tempfile.TemporaryDirectory() as tmp, patch(PROVIDER, return_value=Judge()):
            root = Path(tmp); source = root / "source"; source.mkdir(); out = root / "graded"
            (source / "case.json").write_text(json.dumps({"user_id": "u", "subtask_id": "s", "status": "conversation_finished", "domain": "delivery", "instruction": "点午餐", "evaluation": "skipped", "reward": None}))
            (source / "simulation.json").write_text(json.dumps(simulation))
            (source / "trace.jsonl").write_text("")
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            summary = grade_existing_case(source, task, subtask, catalog=catalog, out=out)
            self.assertEqual(summary["reward"], 0)
            self.assertEqual(summary["evaluation"], "completed")
            self.assertEqual(hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()})
            self.assertEqual(json.loads((out / "simulation.json").read_text())["messages"], simulation["messages"])
            self.assertTrue((out / "grading.html").exists())
            self.assertEqual(summary["model_outputs"], 0)

    def test_regrading_rejects_wrong_user(self):
        catalog, task, subtask, simulation = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"; source.mkdir()
            (source / "case.json").write_text(json.dumps({"user_id": "someone_else", "subtask_id": "s"}))
            (source / "simulation.json").write_text(json.dumps(simulation))
            with self.assertRaisesRegex(ValueError, "不一致"):
                grade_existing_case(source, task, subtask, catalog=catalog, out=Path(tmp) / "graded")


if __name__ == "__main__":
    unittest.main()
