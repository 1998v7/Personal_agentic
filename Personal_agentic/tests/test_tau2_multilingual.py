"""τ² 多语种(text-v1:6 语种 × 3 domain)的测试。全部离线。需要 ``data/tau2`` 数据集;缺数据时整个文件跳过。"""

import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from IMAct.benchmarks.tau2 import multilingual as ml
    from IMAct.benchmarks.tau2.multilingual import LANGUAGES, load_base_cell, load_cell, text_v1_cells

    HAS_DATA = bool(ml.frozen_canonical_ids("airline"))
except Exception:  # noqa: BLE001
    HAS_DATA = False

DOMAINS = ("airline", "retail", "telecom")


@unittest.skipUnless(HAS_DATA, "需要 data/tau2 数据集")
class ManifestTests(unittest.TestCase):
    def test_text_v1_is_six_languages_by_three_domains(self):
        cells = text_v1_cells()
        self.assertEqual(len(cells), 18)
        self.assertEqual({l for l, _ in cells}, set(LANGUAGES))
        self.assertEqual({d for _, d in cells}, set(DOMAINS))
        self.assertEqual(len(set(cells)), 18)

    def test_frozen_subsets_have_fifty_tasks_each(self):
        for domain in DOMAINS:
            ids = ml.frozen_canonical_ids(domain)
            self.assertEqual((len(ids), len(set(ids))), (50, 50), domain)

    def test_a_tampered_subset_file_is_rejected(self):
        ml.frozen_canonical_ids.cache_clear()
        real = ml.SUBSETS / "airline_50.json"
        with mock.patch.object(type(real), "read_bytes", return_value=b"{}"):
            with self.assertRaisesRegex(ValueError, "摘要不一致"):
                ml.frozen_canonical_ids("airline")
        ml.frozen_canonical_ids.cache_clear()

    def test_subset_digest_matches_the_manifest(self):
        for domain, ref in ml._manifest()["task_subsets"].items():
            self.assertEqual(hashlib.sha256((ml.SUBSETS / f"{ref['name']}.json").read_bytes()).hexdigest(), ref["sha256"])

    def test_unknown_language_or_domain_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "不支持的语言"):
            ml.load_pack("fr")
        with self.assertRaisesRegex(ValueError, "不支持的 domain"):
            load_cell("en", "hotel")
        with self.assertRaisesRegex(ValueError, "text-v1 里没有"):
            load_cell("fr", "airline")


@unittest.skipUnless(HAS_DATA, "需要 data/tau2 数据集")
class CellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = {(l, d): load_cell(l, d) for l, d in text_v1_cells()}

    def test_every_cell_has_the_frozen_fifty_tasks_in_the_same_order(self):
        for (language, domain), cell in self.cells.items():
            self.assertEqual(cell.canonical_ids, list(ml.frozen_canonical_ids(domain)))
            self.assertEqual(len(cell.tasks), 50, cell.name)
            self.assertEqual([cell.canonical_of(t) for t in cell.tasks], cell.canonical_ids, cell.name)

    def test_canonical_ids_are_shared_across_languages_but_local_ids_are_not(self):
        en, zh = self.cells[("en", "retail")], self.cells[("zh", "retail")]
        self.assertEqual(en.canonical_ids, zh.canonical_ids)
        self.assertEqual(en.tasks[0].id, en.canonical_ids[0])  # 英文本地 ID 就是 canonical ID
        self.assertEqual(zh.tasks[0].id, f"{zh.canonical_ids[0]}_zh_identity")
        self.assertNotEqual(en.tasks[0].user_scenario.instructions, zh.tasks[0].user_scenario.instructions)  # 场景本身是本地化的

    def test_each_language_has_exactly_one_persona_per_gender_so_assignment_is_seed_free(self):
        for language in LANGUAGES:
            genders = [p["gender"] for p in ml.load_pack(language)["personas"].values()]
            self.assertEqual(sorted(genders), ["female", "male"], language)
        cell = self.cells[("ko", "airline")]
        for task in cell.tasks:
            ids = {ml.assign_persona(cell.pack, cell.metadata[task.id], seed) for seed in (300, 1, 7, 12345)}
            self.assertEqual(len(ids), 1)

    def test_persona_matches_the_caller_gender_of_the_task(self):
        for cell in self.cells.values():
            for task in cell.tasks[:5]:
                self.assertEqual(cell.persona(task).gender, cell.metadata[task.id]["caller_gender"])

    def test_greeting_comes_from_the_language_pack_unless_the_task_overrides_it(self):
        for (language, _), cell in self.cells.items():
            self.assertEqual(cell.greeting(cell.tasks[0]), ml.load_pack(language)["agent_greeting"])
        task = self.cells[("en", "airline")].tasks[0].model_copy(update={"agent_opener": "Custom opener"})
        self.assertEqual(self.cells[("en", "airline")].greeting(task), "Custom opener")

    def test_english_cells_use_the_reviewed_replacement_tasks_not_the_originals(self):
        base = load_base_cell("retail")
        cell = self.cells[("en", "retail")]
        originals = {t.id: t for t in base.tasks}
        differing = [t.id for t in cell.tasks if str(t.user_scenario) != str(originals[t.id].user_scenario)]
        self.assertTrue(differing, "审阅过的英文 retail 任务应当与原任务有所不同")

    def test_base_cells_keep_the_pre_2_0_sizes(self):
        self.assertEqual([len(load_base_cell(d).tasks) for d in DOMAINS], [50, 114, 114])
        self.assertEqual(load_base_cell("retail").label, "base")

    def test_select_is_by_canonical_id_and_validates_its_input(self):
        cell = self.cells[("zh", "airline")]
        picked = cell.select(task_ids=["3", "1"])
        self.assertEqual([cell.canonical_of(t) for t in picked], [i for i in cell.canonical_ids if i in {"1", "3"}])
        self.assertEqual(len(cell.select(num_tasks=4)), 4)
        self.assertEqual(len(cell.select()), 50)
        with self.assertRaisesRegex(ValueError, "冻结子集里没有"):
            cell.select(task_ids=["no-such"])
        for bad in (0, 51):
            with self.assertRaisesRegex(ValueError, "必须在 1 到 50"):
                cell.select(num_tasks=bad)

    def test_regional_environments_carry_the_local_currency_and_a_local_policy(self):
        expected = {"es": "EUR", "pt": "BRL", "hi": "INR", "zh": "CNY", "ko": "KRW"}
        for language, currency in expected.items():
            cell = self.cells[(language, "airline")]
            env = cell.environment()
            self.assertIn(currency, " ".join(t.long_desc or "" for t in env.get_tools()), language)
            self.assertNotEqual(env.get_policy(), self.cells[("en", "airline")].environment().get_policy())

    def test_environments_are_independent_copies(self):
        cell = self.cells[("zh", "retail")]
        a, b = cell.environment(), cell.environment()
        self.assertIsNot(a.tools.db, b.tools.db)
        self.assertEqual(a.get_db_hash(), b.get_db_hash())

    def test_telecom_regional_cells_have_user_tools_and_a_separate_user_db(self):
        env = self.cells[("pt", "telecom")].environment()
        self.assertGreater(len(env.get_user_tools()), 20)
        self.assertIsNot(env.tools.db, env.user_tools.db)


@unittest.skipUnless(HAS_DATA, "需要 data/tau2 数据集")
class PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = {(l, d): load_cell(l, d) for l, d in text_v1_cells() if d in ("airline", "telecom")}

    def user_prompt(self, language, domain="airline"):
        from IMAct.benchmarks.tau2.prompts import user_system_prompt

        cell = self.cells[(language, domain)]
        task = cell.tasks[0]
        return user_system_prompt(cell, task, use_tools=domain == "telecom"), cell, task

    def agent_prompt(self, language, domain="airline"):
        from IMAct.benchmarks.tau2.prompts import agent_domain_policy, agent_system_prompt

        cell = self.cells[(language, domain)]
        task = cell.tasks[0]
        return agent_system_prompt(agent_domain_policy(cell.environment().get_policy(), cell, cell.persona(task))), cell

    def test_non_english_callers_are_told_to_write_in_their_language_and_reminded_at_the_end(self):
        for language, name in (("zh", "Mandarin"), ("es", "Spanish"), ("ko", "Korean"), ("hi", "Hindi"), ("pt", "Portuguese")):
            prompt, _, _ = self.user_prompt(language)
            self.assertIn("(MANDATORY)", prompt, language)
            self.assertIn(name, prompt, language)
            self.assertIn("FINAL REMINDER", prompt.rsplit("</scenario>", 1)[1], language)  # 提醒在场景之后

    def test_english_callers_get_no_language_directive_or_reminder(self):
        prompt, _, _ = self.user_prompt("en")
        self.assertNotIn("MANDATORY", prompt)
        self.assertNotIn("FINAL REMINDER", prompt)

    def test_localized_cells_state_the_currency_for_the_caller_english_cells_do_not(self):
        zh, _, _ = self.user_prompt("zh")
        en, _, _ = self.user_prompt("en")
        self.assertIn("Scenario monetary amounts are in CNY", zh)
        self.assertNotIn("monetary amounts", en)

    def test_the_user_prompt_contains_persona_scenario_and_the_right_scaffold_for_tools(self):
        plain, _, task = self.user_prompt("zh", "airline")
        tools, _, _ = self.user_prompt("zh", "telecom")
        self.assertIn(str(task.user_scenario), plain)
        self.assertNotIn("Make a tool call to perform an action", plain)
        self.assertIn("Make a tool call to perform an action", tools)
        self.assertNotIn("<PERSONA_GUIDELINES>", plain + tools)

    def test_agent_prompt_has_the_language_clause_and_the_regional_tool_convention(self):
        zh, cell = self.agent_prompt("zh")
        self.assertIn(cell.pack["agent_language_clause"], zh)
        self.assertIn("Use the native-script names, addresses and other localized values", zh)
        en, _ = self.agent_prompt("en")
        self.assertIn("Respond in English using Latin script", en)
        self.assertNotIn("native-script", en)
        self.assertNotIn("Tool names, argument keys", en)

    def test_the_transfer_line_in_the_policy_is_spoken_in_the_call_language(self):
        english = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
        for language in ("zh", "es", "ko", "hi", "pt"):
            prompt, _ = self.agent_prompt(language)
            self.assertNotIn(english, prompt, language)
        zh, _ = self.agent_prompt("zh")
        self.assertIn("现在为您转接人工客服", zh)
        en, _ = self.agent_prompt("en")
        self.assertIn(english, en)


@unittest.skipUnless(HAS_DATA, "需要 data/tau2 数据集")
class GoldenReplayInEveryLanguageTests(unittest.TestCase):
    """最有价值的功能测试:把每个 cell 里参考动作在**本语言的本地化环境**里重放,评分必须通过环境终态检查。

    证明每种语言的本地化数据库、policy、区域设置、工具与该语言的评分标准是一致的。
    """

    @staticmethod
    def golden_messages(cell, task):
        from IMAct.harness.messages import AssistantMessage, ToolCall, UserMessage

        env = cell.environment()
        state = task.initial_state
        env.set_state(initialization_data=state.initialization_data if state else None,
                      initialization_actions=state.initialization_actions if state else None,
                      message_history=list(state.message_history) if state and state.message_history else [])
        msgs = [AssistantMessage(role="assistant", content=cell.greeting(task), cost=0.0), UserMessage(role="user", content="hello", cost=0.0)]
        for i, action in enumerate(task.evaluation_criteria.actions or []):
            call = ToolCall(id=f"g{i}", name=action.name, arguments=action.arguments, requestor=action.requestor)
            msgs.append((AssistantMessage if action.requestor == "assistant" else UserMessage)(role=action.requestor, cost=0.0, tool_calls=[call]))
            tool_msg = env.get_response(call)
            from IMAct.harness.messages import ToolMessage

            msgs.append(ToolMessage(id=tool_msg.id, role="tool", content=tool_msg.content, requestor=tool_msg.requestor, error=tool_msg.error))
            env.sync_tools()
        return msgs + [UserMessage(role="user", content="thanks ###STOP###", cost=0.0)]

    def test_replaying_the_reference_actions_reaches_the_reference_end_state_in_all_18_cells(self):
        from IMAct.benchmarks.tau2.runner import Tau2Runner
        from IMAct.evaluation.tau2 import Tau2Evaluator

        for language, domain in text_v1_cells():
            cell = load_cell(language, domain)
            runner = Tau2Runner(cell, agent_llm=mock.Mock(), user_llm=mock.Mock(), score=False)
            task = next(t for t in cell.tasks if t.evaluation_criteria and t.evaluation_criteria.actions)
            info = Tau2Evaluator().evaluate_env(runner.eval_task(task), self.golden_messages(cell, task))
            self.assertEqual(info.reward, 1.0, f"{cell.name} task {task.id}: {info}")
            self.assertTrue(info.db_check.db_match, cell.name)

    def test_doing_nothing_fails_a_task_that_needs_a_change_in_every_language(self):
        from IMAct.benchmarks.tau2.runner import Tau2Runner
        from IMAct.evaluation.tau2 import Tau2Evaluator
        from IMAct.harness.messages import AssistantMessage, UserMessage

        for language in LANGUAGES:
            cell = load_cell(language, "retail")
            runner = Tau2Runner(cell, agent_llm=mock.Mock(), user_llm=mock.Mock(), score=False)
            task = next(t for t in cell.tasks if t.evaluation_criteria.actions and "DB" in [getattr(b, "value", b) for b in t.evaluation_criteria.reward_basis])
            idle = [AssistantMessage(role="assistant", content=cell.greeting(task), cost=0.0), UserMessage(role="user", content="bye ###STOP###", cost=0.0)]
            info = Tau2Evaluator().evaluate_env(runner.eval_task(task), idle)
            self.assertFalse(info.db_check.db_match, language)


@unittest.skipUnless(HAS_DATA, "需要 data/tau2 数据集")
class CrossLingualJudgeTests(unittest.TestCase):
    def judge_system_prompt(self, language):
        from IMAct.benchmarks.tau2.runner import Tau2Runner
        from IMAct.evaluation.tau2 import Tau2Evaluator
        from IMAct.harness.conversation import ConversationResult, TerminationReason
        from IMAct.harness.messages import AssistantMessage, UserMessage

        cell = load_cell(language, "airline")
        task = next(t for t in cell.tasks if t.evaluation_criteria.communicate_info)
        seen = []
        judge = SimpleNamespace(chat=lambda messages, **kw: (seen.append(messages[0].content), SimpleNamespace(content=json.dumps({"reasoning": "r", "met": True})))[1])
        runner = Tau2Runner(cell, agent_llm=mock.Mock(), user_llm=mock.Mock(), judge=judge)
        msgs = [AssistantMessage(role="assistant", content="hi", cost=0.0), UserMessage(role="user", content="bye ###STOP###", cost=0.0)]
        eval_task = runner.eval_task(task)
        Tau2Evaluator(judge).evaluate_communicate(eval_task, msgs)
        return seen[0], eval_task

    def test_the_judge_is_told_the_customer_language_only_for_cross_lingual_cells(self):
        zh, eval_zh = self.judge_system_prompt("zh")
        en, eval_en = self.judge_system_prompt("en")
        self.assertEqual((eval_zh.customer_language, eval_en.customer_language), ("zh", None))
        self.assertIn("ISO 639-1 code 'zh'", zh)
        self.assertNotIn("ISO 639-1", en)


class CliTests(unittest.TestCase):
    def parse(self, *argv):
        from IMAct.benchmarks.tau2.run import build_parser

        return build_parser().parse_args(["--models", "m.yaml", *argv])

    def test_defaults_are_english_text_v1(self):
        args = self.parse("--domain", "airline")
        self.assertEqual((args.language, args.benchmark, args.num_trials, args.max_steps, args.max_errors), (["en"], "text-v1", 1, 200, 10))

    def test_unknown_domain_language_or_benchmark_is_rejected(self):
        for bad in (["--domain", "hotel"], ["--domain", "airline", "--language", "fr"], ["--domain", "airline", "--benchmark", "voice"]):
            with self.assertRaises(SystemExit):
                self.parse(*bad)

    @unittest.skipUnless(HAS_DATA, "需要 data/tau2 数据集")
    def test_cell_selection_follows_the_manifest_order_and_the_filters(self):
        from IMAct.benchmarks.tau2.run import selected_cells

        names = [c.name for c in selected_cells(self.parse("--domain", "airline", "telecom", "--language", "zh", "en"))]
        self.assertEqual(names, ["en-airline", "en-telecom", "zh-airline", "zh-telecom"])
        self.assertEqual(len(selected_cells(self.parse("--domain", "all", "--language", "all"))), 18)

    @unittest.skipUnless(HAS_DATA, "需要 data/tau2 数据集")
    def test_invalid_combinations_are_refused(self):
        from IMAct.benchmarks.tau2.run import selected_cells

        with self.assertRaisesRegex(ValueError, "只支持英文"):
            selected_cells(self.parse("--domain", "airline", "--benchmark", "base", "--language", "zh"))
        with self.assertRaisesRegex(ValueError, "单个 --domain"):
            selected_cells(self.parse("--domain", "airline", "retail", "--task-ids", "3"))

    def test_summary_aggregates_cells_and_counts_evaluation_errors(self):
        from IMAct.benchmarks.tau2.run import summarize

        cells = {"en-airline": {"results": [{"reward": 1.0}, {"reward": 0.0}], "failed": [], "skipped": [], "avg_reward": 0.5, "num_evaluation_errors": 1},
                 "zh-airline": {"results": [{"reward": 1.0}], "failed": [{"task_id": "x", "error": "boom"}], "skipped": [{}], "avg_reward": 1.0}}
        s = summarize(cells, run_id="r", agent_model="m", benchmark="text-v1")
        self.assertEqual((s["completed"], s["failed"], s["skipped"], s["evaluation_errors"]), (3, 1, 1, 1))
        self.assertAlmostEqual(s["avg_reward"], 2 / 3)  # 按 task 平均,不是 cell 均值的均值
        self.assertEqual(s["cells"]["zh-airline"]["failed_tasks"][0]["task_id"], "x")


if __name__ == "__main__":
    unittest.main()
