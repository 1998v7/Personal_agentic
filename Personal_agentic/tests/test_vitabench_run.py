"""入口脚本的测试。真正的模拟需要用户模拟器与评测器两个 LLM,这里用替身验证外围逻辑:
任务级重试、断点续跑、单任务失败不拖垮整批、汇总、--check 的诊断。"""

import json
import tempfile
import types
import unittest
from pathlib import Path

from tests.exec_helpers import DemoProvider, make_runtime  # noqa: F401  (设置 OPENAI_API_KEY 占位值)

from IMAct.benchmarks.vitabench import run as vrun  # noqa: E402
from IMAct.exec_layer.config import OuterConfig  # noqa: E402
from IMAct.models.catalog import ModelCatalog  # noqa: E402


class FakeSim:
    def __init__(self, reward):
        self.reward_info = types.SimpleNamespace(reward=reward) if reward is not None else None
        self._reward = reward

    def model_dump(self, mode="json"):  # 与真实结果模型同形:落盘的记录里带 reward_info
        return {"reward_info": None if self._reward is None else {"reward": self._reward, "info": None}}


def args(out, **kw):
    ns = dict(out=out, run_id="r1", num_trials=1, seed=None, rerun=False, max_steps=5,
              language="chinese", max_task_retries=2, max_concurrency=1)
    ns.update(kw)
    return types.SimpleNamespace(**ns)


def task(i):
    return types.SimpleNamespace(id=f"task{i}", subtasks=[1, 2])


CFG = OuterConfig()


class RunAllTests(unittest.TestCase):
    def test_summary_and_result_files(self):
        out = tempfile.mkdtemp()
        rewards = {"task1": 1.0, "task2": 0.5}
        summary = vrun.run_all(args(out), CFG, None, [task(1), task(2)], simulate_fn=lambda t, **kw: FakeSim(rewards[t.id]))
        self.assertEqual(summary["avg_reward"], 0.75)
        self.assertEqual(sorted(r["task_id"] for r in summary["results"]), ["task1", "task2"])
        self.assertTrue((Path(out) / "task1.trial0.json").exists())
        self.assertEqual(json.loads((Path(out) / "summary.json").read_text(encoding="utf-8"))["avg_reward"], 0.75)

    def test_task_level_retry_then_success(self):
        calls = []

        def flaky(t, **kw):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("transient")
            return FakeSim(1.0)

        summary = vrun.run_all(args(tempfile.mkdtemp(), max_task_retries=3), CFG, None, [task(1)], simulate_fn=flaky)
        self.assertEqual(len(calls), 3)
        self.assertEqual(summary["failed"], [])

    def test_failure_after_retries_is_isolated(self):
        def sim(t, **kw):
            if t.id == "task1":
                raise RuntimeError("always")
            return FakeSim(1.0)

        orig = vrun.time.sleep
        vrun.time.sleep = lambda s: None  # 不真睡
        try:
            summary = vrun.run_all(args(tempfile.mkdtemp(), max_task_retries=1), CFG, None, [task(1), task(2)], simulate_fn=sim)
        finally:
            vrun.time.sleep = orig
        self.assertEqual([f["task_id"] for f in summary["failed"]], ["task1"])
        self.assertIn("always", summary["failed"][0]["error"])
        self.assertEqual([r["task_id"] for r in summary["results"]], ["task2"])

    def test_resume_skips_finished_results_unless_rerun(self):
        out = tempfile.mkdtemp()
        vrun.run_all(args(out), CFG, None, [task(1)], simulate_fn=lambda t, **kw: FakeSim(1.0))

        def must_not_run(t, **kw):
            raise AssertionError("已完成的任务不应重跑")

        again = vrun.run_all(args(out), CFG, None, [task(1)], simulate_fn=must_not_run)
        self.assertEqual(again["skipped"], [{"task_id": "task1", "trial": 0}])
        rerun = vrun.run_all(args(out, rerun=True), CFG, None, [task(1)], simulate_fn=lambda t, **kw: FakeSim(0.0))
        self.assertEqual(rerun["avg_reward"], 0.0)

    def test_a_resumed_run_still_summarizes_the_earlier_results(self):
        """断点续跑:以前跑完的 job 不重跑,但它们的分数要算进这次的汇总,而不是只算新跑的。"""
        out = tempfile.mkdtemp()
        vrun.run_all(args(out), CFG, None, [task(1)], simulate_fn=lambda t, **kw: FakeSim(1.0))
        again = vrun.run_all(args(out), CFG, None, [task(1), task(2)], simulate_fn=lambda t, **kw: FakeSim(0.0))
        self.assertEqual(sorted(r["task_id"] for r in again["results"]), ["task1", "task2"])
        self.assertEqual(again["avg_reward"], 0.5)
        self.assertEqual([r.get("resumed", False) for r in sorted(again["results"], key=lambda r: r["task_id"])], [True, False])
        self.assertEqual(again["skipped"], [{"task_id": "task1", "trial": 0}])

    def test_an_unreadable_earlier_result_is_run_again(self):
        out = tempfile.mkdtemp()
        vrun.run_all(args(out), CFG, None, [task(1)], simulate_fn=lambda t, **kw: FakeSim(1.0))
        from IMAct.harness.batch import result_path

        result_path(Path(out), "task1", 0).write_text("{truncated", encoding="utf-8")
        again = vrun.run_all(args(out), CFG, None, [task(1)], simulate_fn=lambda t, **kw: FakeSim(0.0))
        self.assertEqual((again["skipped"], again["avg_reward"]), ([], 0.0))

    def test_trials_and_seeds(self):
        seen = []
        vrun.run_all(args(tempfile.mkdtemp(), num_trials=3, seed=7), CFG, None, [task(1)], simulate_fn=lambda t, **kw: (seen.append((kw["run_id"], kw["seed"])), FakeSim(1.0))[1])
        self.assertEqual([r for r, _ in sorted(seen)], ["r1-t0", "r1-t1", "r1-t2"])
        self.assertEqual(len({s for _, s in seen}), 3)  # 每个 trial 的种子不同

    def test_concurrency(self):
        import threading

        names = set()
        vrun.run_all(args(tempfile.mkdtemp(), max_concurrency=4), CFG, None, [task(i) for i in range(6)],
                     simulate_fn=lambda t, **kw: (names.add(threading.current_thread().name), FakeSim(1.0))[1])
        self.assertGreaterEqual(len(names), 1)

    def test_missing_reward_is_not_counted_as_zero(self):
        summary = vrun.run_all(args(tempfile.mkdtemp()), CFG, None, [task(1), task(2)], simulate_fn=lambda t, **kw: FakeSim(None if t.id == "task1" else 1.0))
        self.assertEqual(summary["avg_reward"], 1.0)


class CheckTests(unittest.TestCase):
    def test_check_reports_missing_everything(self):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = vrun.check(OuterConfig.model_validate({"model": {"provider": "openai", "model": "gpt-5.5", "api_key_env": "IMACT_DEFINITELY_UNSET_KEY"}}),
                              args(tempfile.mkdtemp()), tasks=[])
        text = buf.getvalue()
        self.assertEqual(code, 2)
        self.assertIn("IMACT_DEFINITELY_UNSET_KEY", text)  # 说清楚缺哪个环境变量
        self.assertIn("data/vitabench/tasks.json", text)  # 说清楚数据集放在哪
        self.assertIn("未配置", text)  # Memory 未启用要明说,而不是静默

    def test_check_passes_when_everything_is_there(self):
        import contextlib
        import io
        import os

        os.environ["IMACT_TEST_OK_KEY"] = "k"
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = vrun.check(OuterConfig.model_validate({"model": {"provider": "openai", "model": "m", "api_key_env": "IMACT_TEST_OK_KEY"},
                                                             "memory": {"gateway": {"url": "https://gw/x"}}}),
                                  args(tempfile.mkdtemp()), tasks=[task(1)])
        finally:
            del os.environ["IMACT_TEST_OK_KEY"]
        self.assertEqual(code, 0, msg=buf.getvalue())

    def test_missing_user_or_judge_credentials_are_flagged_before_any_request(self):
        import contextlib
        import io
        import os

        catalog = ModelCatalog.model_validate({
            "roles": {"action": "a", "user": "u", "judge": "j"},
            "models": {"a": {"provider": "openai", "model": "m", "api_key": "k"},
                       "u": {"provider": "openai", "model": "m", "api_key_env": "IMACT_DEFINITELY_UNSET_USER_KEY"},
                       "j": {"provider": "openai", "model": "m", "api_key_env": "IMACT_DEFINITELY_UNSET_JUDGE_KEY"}}})
        os.environ.pop("IMACT_DEFINITELY_UNSET_USER_KEY", None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = vrun.check(OuterConfig(model=catalog.selected("action")[1]), args(tempfile.mkdtemp()), tasks=[task(1)], catalog=catalog)
        text = buf.getvalue()
        self.assertEqual(code, 2)
        self.assertIn("IMACT_DEFINITELY_UNSET_USER_KEY", text)
        self.assertIn("IMACT_DEFINITELY_UNSET_JUDGE_KEY", text)


class ParserTests(unittest.TestCase):
    def test_defaults_follow_vitabench_conventions(self):
        a = vrun.build_parser().parse_args(["--config", "c.yaml"])
        self.assertEqual((a.max_task_retries, a.language, a.num_trials, a.max_concurrency), (3, "chinese", 1, 1))


if __name__ == "__main__":
    unittest.main()
