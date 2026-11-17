"""结果表(IMAct/table.py、各 bench 的 tables.py)与 ``python -m IMAct.report`` 的测试。离线、不调用模型。"""

import json
import tempfile
import unittest
from pathlib import Path

from IMAct.table import display_width, fmt, render_table


class RenderTableTests(unittest.TestCase):
    def test_cjk_characters_count_as_two_columns_so_borders_line_up(self):
        out = render_table(["领域", "SR"], [["web_search", 12.5], ["中文名字", None]])
        lines = [l for l in out.splitlines() if l.startswith(("|", "+"))]
        self.assertEqual(len({display_width(l) for l in lines}), 1)  # 每一行显示宽度相同 → 竖线对齐
        self.assertIn("12.50", out)
        self.assertIn(" - ", out)  # None → -

    def test_formatting_rules(self):
        self.assertEqual((fmt(None), fmt(""), fmt(3), fmt(1 / 3), fmt(True)), ("-", "-", "3", "0.33", "是"))
        self.assertEqual(fmt(1 / 3, digits=1), "0.3")

    def test_title_separators_and_footer(self):
        out = render_table(["a", "b"], [[1, 2], [3, 4]], title="T", separators=[1], footer=["note"])
        lines = out.splitlines()
        self.assertEqual(lines[0], "T")
        self.assertEqual(sum(1 for l in lines if l.startswith("+-")), 3)  # 顶、分隔、底
        self.assertEqual(lines[-1], "  note")


class BenchTablesTests(unittest.TestCase):
    def test_mcp_universe_has_the_paper_columns_in_paper_order(self):
        from IMAct.benchmarks.mcp_universe.tables import render

        d = lambda sr: {"completed": 2, "failed": 0, "evaluation_errors": 0, "num_successful": 1, "SR": sr, "AE": 60.0, "AS": 5.0, "avg_iterations": 7.0}  # noqa: E731
        out = render({"agent_model": "m", "SR": 50.0, "AE": 60.0, "AS": 5.0, "completed": 4, "failed": 0, "evaluation_errors": 0,
                      "num_successful": 2, "avg_iterations": 7.0, "domains": {"web_search": d(50.0), "location_navigation": d(50.0)}})
        header = next(l for l in out.splitlines() if "Model" in l)
        self.assertLess(header.index("Location Navigation"), header.index("Web Searching"))
        self.assertIn("Overall SR", header)
        self.assertNotIn("Repository Management", header)  # 没跑的领域不出现

    def test_tau2_is_a_language_by_domain_matrix_weighted_by_task_counts(self):
        from IMAct.benchmarks.tau2.tables import render

        cell = lambda r, n: {"completed": n, "scored": n, "failed": 0, "skipped": 0, "evaluation_errors": 0, "avg_reward": r, "failed_tasks": []}  # noqa: E731
        out = render({"benchmark": "text-v1", "agent_model": "m", "completed": 30, "failed": 0, "evaluation_errors": 0,
                      "cells": {"en-airline": cell(1.0, 10), "zh-airline": cell(0.5, 10), "en-retail": cell(0.0, 10)}})
        en = next(l for l in out.splitlines() if l.startswith("| en "))
        total = next(l for l in out.splitlines() if l.startswith("| 全部"))
        self.assertIn("100.00", en)
        self.assertIn("50.00", en)  # en 两个 domain 的加权平均
        self.assertIn("75.00", total)  # airline 全部语言
        self.assertTrue(total.rstrip("| ").endswith("50.00"))  # 30 个 task 的总平均

    def test_personamem_lists_headline_classes_and_every_task(self):
        from IMAct.benchmarks.personamem_v3.tables import render

        out = render({"agent_model": "m", "judge_model": "j", "accuracy_pct_micro": 55.5, "accuracy_pct_micro_judge": 50.0,
                      "by_class": {"chatbot": {"n": 3, "accuracy_pct": 40.0}, "ranking": {"n": 2, "accuracy_pct": 70.0}},
                      "by_task": {"personalized_recommendation": {"n": 2, "accuracy_pct": 70.0, "task_family": "ranking", "quality_flag": "insufficient_n"},
                                  "chatbot_personalized_response": {"n": 3, "accuracy_pct": 40.0, "task_family": "chatbot", "quality_flag": "ok"}},
                      "by_axis": {"mixed": {"n": 5, "accuracy_pct": 52.0}}, "n_queries": 5, "n_scored": 5, "n_unscored": 0, "n_failed": 0,
                      "n_personas": 1, "memory": {"status_counts": {"ok": 5}, "n_truncated": 0}, "e6_paired": {}})
        header = next(l for l in out.splitlines() if "ALL-JUDGE" in l)
        self.assertLess(header.index("ranking"), header.index("chatbot"))  # 官方类别顺序
        self.assertIn("insufficient_n", out)
        self.assertIn("混合 52.00(5)", out)

    def test_vitabench_breaks_scores_down_by_business_domain(self):
        from IMAct.benchmarks.vitabench.tables import render

        out = render({"model": "m", "results": [
            {"task_id": "A1", "trial": 0, "reward": 0.5, "subtasks": [{"index": "0", "reward": 1.0, "domain": "delivery"}, {"index": "1", "reward": 0.0, "domain": "ota"}]},
            {"task_id": "A2", "trial": 0, "reward": 1.0, "subtasks": [{"index": "0", "reward": 1.0, "domain": "delivery"}]}], "failed": [], "skipped": []})
        head = next(l for l in out.splitlines() if l.startswith("| m "))
        self.assertIn("75.00", head)  # 用户平均 (0.5 + 1.0) / 2
        self.assertIn("66.67", head)  # subtask 平均 2/3
        self.assertIn("外卖", out)
        self.assertIn("酒旅", out)
        self.assertNotIn("到店", out)  # 没出现的域不列


class ReportRebuildTests(unittest.TestCase):
    """``python -m IMAct.report``:从每题结果文件重新汇总,断点续跑过、旧口径的目录也能打出完整的表。"""

    def write(self, path: Path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding="utf-8")

    def test_tau2_rebuild_counts_every_result_file_not_just_the_last_session(self):
        from IMAct.harness.batch import result_path
        from IMAct.report import load

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for i, r in enumerate((1.0, 1.0, 0.0, 1.0)):
                self.write(result_path(out / "en-airline", str(i), 0), {"task_id": str(i), "reward_info": {"reward": r, "info": None}})
            # 旧的 summary 只记了最后一次会话跑的 1 个 task,另有一个失败后来又成功了
            self.write(out / "summary.json", {"benchmark": "text-v1", "agent_model": "m", "cells": {"en-airline": {"completed": 1, "avg_reward": 0.0}}})
            self.write(out / "en-airline" / "summary.json", {"failed": [{"task_id": "3", "trial": 0, "error": "x"}, {"task_id": "9", "trial": 0, "error": "y"}]})
            bench, s = load(out)
        self.assertEqual(bench, "tau2")
        self.assertEqual((s["completed"], s["failed"]), (4, 1))  # task 3 后来有了结果,不再算失败;task 9 仍是失败
        self.assertAlmostEqual(s["avg_reward"], 0.75)

    def test_mcp_universe_rebuild_reproduces_sr_ae_as(self):
        from IMAct.harness.batch import result_path
        from IMAct.report import load

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for i, (reward, score, iters) in enumerate(((1.0, 1.0, 4), (0.0, 0.5, 20))):
                self.write(result_path(out / "web_search", f"t{i}", 0), {"task_id": f"t{i}", "reward_info": {"reward": reward, "info": {
                    "evaluator_score": score, "iterations": iters, "llm_calls": iters, "answered": True}}})
            self.write(out / "summary.json", {"agent_model": "m", "SR": 0.5, "domains": {}})
            bench, s = load(out)
        self.assertEqual(bench, "mcp_universe")
        self.assertEqual((s["SR"], s["AE"], s["AS"]), (50.0, 75.0, 4.0))

    def test_the_report_command_prints_tables_and_reports_unknown_directories(self):
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO

        from IMAct.harness.batch import result_path
        from IMAct.report import main

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            self.write(result_path(out, "A1", 0), {"task_id": "A1", "reward_info": {"reward": 1.0, "info": {"subtask_rewards": {"subtask_0_reward": 1.0}}}})
            self.write(out / "summary.json", {"bench": "vitabench", "model": "m", "results": [], "failed": [], "skipped": []})
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main([str(out), str(Path(tmp) / "missing")])
        self.assertEqual(code, 2)
        self.assertIn("VitaBench", stdout.getvalue())
        self.assertIn("100.00", stdout.getvalue())
        self.assertIn("读不出", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
