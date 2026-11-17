"""pat_train 的离线测试:不连 Gateway / vLLM / OpenRouter / MCP 网关。

    scripts/wsl/py.sh -m unittest discover -s /path/to/personal_agent_train/tests -t /path/to/personal_agent_train
"""

from __future__ import annotations

import asyncio
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from IMAct.exec_layer.bridge import AsyncBridge
from IMAct.harness.messages import UserMessage
from pat_train import runner as R  # noqa: F401  登记 Task 与 agent
from pat_train.agent import ImactAgentConfig, sample_fields
from pat_train.policy import PolicyLLM, policy_model_config
from pat_train.preprocess import build
from uni_agent.agents import ModelConfig
from uni_agent.tasks import TaskConfigResolver, get_task

ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG = ROOT / "configs" / "task_config.yaml"


class FakeChat(BaseHTTPRequestHandler):
    """按队列返回 Chat Completions 响应,记录收到的请求。"""

    replies: list[dict] = []
    requests: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "body": body})
        status, payload = type(self).replies.pop(0)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def completion(content="", tool_calls=None, finish="stop"):
    return 200, {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeChat)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}/sessions/s1/v1"
        cls.bridge = AsyncBridge("test")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.bridge.close()

    def setUp(self):
        FakeChat.replies, FakeChat.requests = [], []
        cfg = policy_model_config(ModelConfig(base_url=self.base, model_name="Qwen3.5-2B"), max_tokens=64)
        self.llm = PolicyLLM(cfg, self.bridge)

    def tearDown(self):
        self.llm.close()

    def chat(self):
        return self.llm.chat([UserMessage(role="user", content="hi")], seed=7)

    def test_request_goes_to_session_url_with_max_tokens_and_no_seed(self):
        FakeChat.replies = [completion("hello")]
        self.assertEqual(self.chat().content, "hello")
        req = FakeChat.requests[0]
        self.assertEqual(req["path"], "/sessions/s1/v1/chat/completions")
        self.assertEqual(req["body"]["model"], "Qwen3.5-2B")
        self.assertEqual(req["body"]["max_tokens"], 64)
        self.assertNotIn("seed", req["body"])

    def test_length_exhausted_reply_is_returned_not_retried(self):
        FakeChat.replies = [completion("", finish="length")]
        msg = self.chat()
        self.assertEqual(msg.content, "")
        self.assertIsNone(msg.tool_calls)
        self.assertEqual(len(FakeChat.requests), 1)

    def test_server_error_is_not_retried(self):
        FakeChat.replies = [(500, {"error": {"message": "boom"}})]
        with self.assertRaises(Exception):
            self.chat()
        self.assertEqual(len(FakeChat.requests), 1)

    def test_non_json_tool_arguments_become_an_invalid_message(self):
        call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{not json"}}
        FakeChat.replies = [completion("", tool_calls=[call], finish="tool_calls")]
        msg = self.chat()
        self.assertEqual(msg.content, "")
        self.assertIsNone(msg.tool_calls)

    def test_missing_base_url_is_rejected(self):
        with self.assertRaises(ValueError):
            policy_model_config(ModelConfig(model_name="x"))


class TaskConfigTests(unittest.TestCase):
    def resolve(self, sample):
        return TaskConfigResolver.from_file(str(TASK_CONFIG)).resolve(
            sample, runtime_model={"base_url": "http://gw/sessions/a/v1", "api_key": "EMPTY", "model_name": "Qwen3.5-2B"})

    def test_parquet_rows_resolve_into_tasks_with_runtime_model(self):
        for row in build(["tau2", "mcp_universe"], "train", seed=0, limit=2):
            sample = row["extra_info"]["tools_kwargs"]["task"]
            task = get_task(self.resolve(sample))
            self.assertEqual(task.name, sample["name"])
            self.assertIsInstance(task.config.agent, ImactAgentConfig)
            self.assertEqual(task.config.agent.model.base_url, "http://gw/sessions/a/v1")
            self.assertEqual(task.config.agent.max_tokens_per_turn, 1024)
            self.assertTrue(all(sample_fields(task.config, "domain", "task_id")))

    def test_sample_fields_rejects_missing_keys(self):
        task = get_task(self.resolve({"name": "imact_tau2", "metadata": {"domain": "retail", "split": "", "task_id": ""}}))
        with self.assertRaises(ValueError):
            sample_fields(task.config, "domain", "split", "task_id")


class PreprocessTests(unittest.TestCase):
    def test_tau2_train_excludes_frozen_eval_subsets(self):
        from IMAct.benchmarks.tau2.multilingual import frozen_canonical_ids

        rows = build(["tau2"], "train", seed=0, limit=None)
        by_domain: dict[str, set[str]] = {}
        for r in rows:
            md = r["extra_info"]["tools_kwargs"]["task"]["metadata"]
            by_domain.setdefault(md["domain"], set()).add(md["task_id"])
        self.assertEqual(set(by_domain), {"retail", "telecom"})
        for domain, ids in by_domain.items():
            self.assertFalse(ids & set(frozen_canonical_ids(domain)), domain)

    def test_mcp_rows_are_read_only(self):
        from pat_train.tasks.mcp_universe import WRITE_SERVERS, task_by_id, DEFAULT_ENV_FILE

        rows = build(["mcp_universe"], "train", seed=0, limit=None)
        self.assertEqual(len(rows), 135)
        for r in rows[:20]:
            md = r["extra_info"]["tools_kwargs"]["task"]["metadata"]
            task = task_by_id(md["domain"], md["task_id"], str(DEFAULT_ENV_FILE))
            self.assertFalse({s["name"] for s in task.mcp_servers} & WRITE_SERVERS)
            self.assertFalse(task.cleanups)

    def test_rows_share_one_metadata_schema(self):
        rows = build(["tau2", "mcp_universe"], "train", seed=0, limit=3)
        keys = {tuple(sorted(r["extra_info"]["tools_kwargs"]["task"]["metadata"])) for r in rows}
        self.assertEqual(keys, {("domain", "split", "subtask", "task_id")})


class Tau2TaskTests(unittest.TestCase):
    def test_evaluation_error_drops_the_sample(self):
        from IMAct.evaluation.base import RewardInfo
        from pat_train.tasks import tau2 as T

        sample = {"name": "imact_tau2", "metadata": {"domain": "retail", "split": "base", "task_id": T.train_cell("retail", "base").canonical_ids[0]}}
        task = get_task(TaskConfigResolver.from_file(str(TASK_CONFIG)).resolve(sample, runtime_model={"base_url": "http://127.0.0.1:9/v1", "model_name": "m"}))
        fake_run = mock.Mock(reward_info=RewardInfo(reward=0.0, info={"evaluation_error": True, "note": "judge down"}), messages=[], termination_reason="user_stop")
        with mock.patch.object(T.Tau2Runner, "run_task", return_value=fake_run), mock.patch.object(T.shared, "role", return_value=mock.Mock()):
            with self.assertRaisesRegex(RuntimeError, "评分出错"):
                asyncio.run(task.run())

    def test_reward_is_tau2_reward(self):
        from IMAct.evaluation.base import RewardInfo
        from pat_train.tasks import tau2 as T

        sample = {"name": "imact_tau2", "metadata": {"domain": "retail", "split": "base", "task_id": T.train_cell("retail", "base").canonical_ids[0]}}
        task = get_task(TaskConfigResolver.from_file(str(TASK_CONFIG)).resolve(sample, runtime_model={"base_url": "http://127.0.0.1:9/v1", "model_name": "m"}))
        fake_run = mock.Mock(reward_info=RewardInfo(reward=1.0, info={}), messages=[1, 2, 3], termination_reason="user_stop")
        with mock.patch.object(T.Tau2Runner, "run_task", return_value=fake_run) as run, mock.patch.object(T.shared, "role", return_value=mock.Mock()):
            result = asyncio.run(task.run())
        self.assertEqual((result.reward, result.accuracy, result.finished), (1.0, 1.0, True))
        runner = run.call_args.args[0] if run.call_args.args else None  # noqa: F841
        self.assertEqual(result.extra_info["num_messages"], 3)


class VitaBenchTaskTests(unittest.TestCase):
    """执行层拒绝 → 整个 episode 0 分且不调裁判;基础设施错误 → 丢样本;否则 rubric 分。不连 Memory / 模型。"""

    @classmethod
    def setUpClass(cls):
        from pat_train.tasks import vitabench as V

        cls.V = V
        tasks = V._tasks("chinese")
        if not tasks:
            raise unittest.SkipTest("没有 data/vitabench/tasks.json")
        task = next(iter(tasks.values()))
        index = next(i for i, s in enumerate(task.subtasks) if s.evaluation_criteria is not None)
        cls.sample = {"name": V.NAME, "metadata": {"domain": task.subtasks[index].domain, "task_id": task.id, "subtask": str(index)}}

    def make_task(self, **overrides):
        sample = {**self.sample, **overrides}
        return get_task(TaskConfigResolver.from_file(str(TASK_CONFIG)).resolve(sample, runtime_model={"base_url": "http://127.0.0.1:9/v1", "model_name": "m"}))

    def run_with(self, codes, *, reward=0.75, eval_info=None):
        from IMAct.evaluation.base import RewardInfo
        from IMAct.harness.conversation import TerminationReason

        V = self.V
        conversation = mock.Mock(termination_reason=TerminationReason.USER_STOP, messages=[1, 2])
        runner = mock.Mock()
        runner.run_subtask.return_value = mock.Mock(conversation=conversation)
        runner.evaluate_subtask.return_value = RewardInfo(reward=reward, info=eval_info or {})
        rt = mock.Mock()
        with mock.patch.object(V, "build_runtime", return_value=rt) as build, \
                mock.patch.object(V, "PersonalizationRunner", return_value=runner), \
                mock.patch.object(V, "attempt_errors", return_value=codes), \
                mock.patch.object(V.shared, "role", return_value=mock.Mock()):
            try:
                return asyncio.run(self.make_task().run()), runner, build.call_args.args[0]
            finally:
                rt.close.assert_called_once()

    def test_rejected_decision_zeroes_the_episode_without_judge(self):
        result, runner, _ = self.run_with(["ungrounded_id"])
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.extra_info["rejections"], ["ungrounded_id"])
        runner.evaluate_subtask.assert_not_called()

    def test_infra_error_drops_the_sample(self):
        with self.assertRaisesRegex(RuntimeError, "基础设施"):
            self.run_with(["model_transport_error"])

    def test_clean_episode_gets_rubric_reward_and_training_settings(self):
        result, runner, outer = self.run_with([], reward=0.75)
        self.assertEqual((result.reward, result.accuracy), (0.75, 0.75))
        self.assertEqual(outer.retry.model_call.max_retries, 0)
        self.assertTrue(outer.memory.require_prefetch)
        self.assertEqual(outer.model.base_url, "http://127.0.0.1:9/v1")
        self.assertTrue(outer.exec.db_path.endswith("exec.sqlite3"))

    def test_judge_error_drops_the_sample(self):
        with self.assertRaisesRegex(RuntimeError, "评分出错"):
            self.run_with([], eval_info={"evaluation_error": True, "note": "judge down"})

    def test_memory_is_required(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write("memory: {backend: none}\n")
        task = get_task(TaskConfigResolver.from_file(str(TASK_CONFIG)).resolve(
            {**self.sample, "outer_config": f.name}, runtime_model={"base_url": "http://127.0.0.1:9/v1", "model_name": "m"}))
        with self.assertRaisesRegex(ValueError, "必须带 Memory"):
            asyncio.run(task.run())

    def test_attempt_errors_reads_codes_in_order(self):
        import sqlite3
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "exec.sqlite3"
            with sqlite3.connect(db) as c:
                c.execute("CREATE TABLE attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, event_id TEXT, context TEXT, decision TEXT, error TEXT)")
                for err in (None, '{"code": "invalid_decision", "message": "x"}', "not json", '{"code": "decision_timeout"}'):
                    c.execute("INSERT INTO attempts(task_id,event_id,error) VALUES('t','e',?)", (err,))
            self.assertEqual(self.V.attempt_errors(db), ["invalid_decision", "unknown", "decision_timeout"])


if __name__ == "__main__":
    unittest.main()
