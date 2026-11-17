"""MCP-Universe 接入的测试。全部离线:不连网关、不调模型。

``fixtures/mcp_universe_*`` 是上游 MCP-Universe(v1.1.3,mcp 1.13.1)在同样输入上的输出("对拍"基线):
ReAct 的每一轮 prompt / 历史 / 最终回答、729 条评估 × 多个回答在"假世界"(``mcp_universe_world.py``)上的判定、
212 个任务渲染后的问题与评估配置、清理参数的解析。这里用 IMAct 的实现重放,必须逐项一致。
"""

import asyncio
import gzip
import json
import os
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from IMAct.benchmarks.mcp_universe import checks  # noqa: F401
from IMAct.benchmarks.mcp_universe import cleanups as cleanups_mod
from IMAct.benchmarks.mcp_universe import runner as runner_mod
from IMAct.benchmarks.mcp_universe import tasks as T
from IMAct.benchmarks.mcp_universe.prompts import TOOL_RESPONSE_SUMMARIZER_PROMPT, question_with_format, render_react_prompt
from IMAct.evaluation.mcp_universe import COMPARISON_FUNCTIONS, Context, Evaluator, EvaluatorConfig, FunctionResult
from IMAct.harness.mcp import GatewayConfig, McpError, McpResult, McpTool
from IMAct.harness.react import ReActAgent

from . import mcp_universe_world as world

FIXTURES = Path(__file__).parent / "fixtures"
ENV = {"GITHUB_PERSONAL_ACCOUNT_NAME": "testuser", "NOTION_ROOT_PAGE": "Root-123", "SOME_OTHER": "v"}


def load_gz(name):
    with gzip.open(FIXTURES / f"{name}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def run_async(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------------------------- ReAct 对拍
class FakeLLM:
    def __init__(self, script):
        self.script, self.seen = list(script), []

    def chat(self, messages, **kw):
        self.seen.append(messages)
        return SimpleNamespace(content=self.script.pop(0), usage={})


class FakeToolset:
    def __init__(self, tools, servers, outs):
        self.tools = {s: [McpTool(t["name"], t["description"], t["schema"]) for t in tools[s]] for s in servers}
        self.outs, self.used = outs, {}

    def call(self, server, tool, arguments):
        key = f"{server}/{tool}"
        i = self.used.get(key, 0)
        self.used[key] = i + 1
        out = self.outs[key][i]
        kind = out["kind"]
        if kind == "raise":
            raise RuntimeError(out["msg"])
        blocks = {"text": [{"type": "text", "text": out.get("text")}], "error_text": [{"type": "text", "text": out.get("text")}],
                  "image": [{"type": "image", "data": "AAAA", "mimeType": "image/png"}], "empty": []}[kind]
        return McpResult({"content": blocks, "isError": kind == "error_text"})


class ReActGoldenTests(unittest.TestCase):
    """24 个用例:每一轮发给模型的 prompt、历史、最终回答都与上游 ReAct 逐字一致(含各种出错路径)。"""

    @classmethod
    def setUpClass(cls):
        data = load_gz("mcp_universe_react")
        cls.tools, cls.cases, cls.reference = data["tools"], data["cases"], data["reference"]

    def replay(self, case):
        servers = case.get("servers", list(self.tools))
        llm = FakeLLM(case["llm"])
        toolset = FakeToolset(self.tools, servers, case.get("tools_out", {}))
        instruction = case.get("instruction", "You are a test agent.")
        agent = ReActAgent(
            llm, toolset, build_prompt=lambda q, h: render_react_prompt(instruction, q, case["max_iterations"], h, toolset.tools),
            summary_prompt=lambda c, r: TOOL_RESPONSE_SUMMARIZER_PROMPT.format(context=c, tool_response=r),
            max_iterations=case["max_iterations"], summarize_tool_response=case.get("summarize", False))
        result = agent.run(question_with_format(case["question"], case.get("output_format")))
        return result, llm

    def test_every_case_matches_upstream(self):
        self.assertEqual(len(self.cases), 24)
        for case in self.cases:
            with self.subTest(case["name"]):
                result, llm = self.replay(case)
                want = self.reference[case["name"]]
                self.assertEqual(result.response, want["response"])
                self.assertEqual([m[0].content for m in llm.seen], want["prompts"])
                self.assertEqual("\n".join(result.history), want["history"])
                self.assertEqual(len(llm.seen), want["llm_calls"])

    def test_llm_calls_count_thoughts_and_tool_summaries(self):
        case = next(c for c in self.cases if c["name"] == "summarize_tool_response")
        result, llm = self.replay(case)
        self.assertEqual((result.iterations, result.usage["llm_calls"], len(llm.seen)), (2, 3, 3))  # 2 轮思考 + 1 次摘要

    def test_the_model_only_ever_sees_one_user_message_per_request(self):
        result, llm = self.replay(self.cases[0])
        self.assertTrue(all(len(m) == 1 and m[0].role == "user" for m in llm.seen))

    def test_executed_tool_calls_are_recorded_in_order_with_their_responses(self):
        case = next(c for c in self.cases if c["name"] == "two_servers_history_grows")
        result, _ = self.replay(case)
        self.assertEqual([(c["seq"], c["server"], c["tool_name"]) for c in result.tool_calls], [(0, "alpha", "search"), (1, "beta", "fetch")])
        self.assertEqual(result.tool_calls[1]["arguments"], {"url": "http://x", "mode": "b"})
        self.assertEqual(result.tool_calls[1]["response"]["content"][0]["text"], "page body\nline2")

    def test_calls_that_fail_validation_are_not_recorded_but_failed_executions_are(self):
        case = next(c for c in self.cases if c["name"] == "invalid_actions")
        result, _ = self.replay(case)
        self.assertEqual(result.tool_calls, [])  # 都在校验阶段被拒
        case = next(c for c in self.cases if c["name"] == "tool_raises")
        result, _ = self.replay(case)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertIn("boom", result.tool_calls[0]["error"])

    def test_partial_results_survive_a_model_failure(self):
        """模型请求中途失败抛出时,已执行的工具调用还在传入的 result 里,清理才找得到 agent 创建了什么。"""
        from IMAct.harness.react import ReActResult

        class Dying(FakeLLM):
            def chat(self, messages, **kw):
                if len(self.seen) == 1:
                    raise RuntimeError("model down")
                return super().chat(messages, **kw)

        toolset = FakeToolset(self.tools, ["alpha"], {"alpha/search": [{"kind": "text", "text": "r"}]})
        llm = Dying(['{"thought": "t", "action": {"server": "alpha", "tool": "search", "arguments": {"query": "q"}}}'])
        agent = ReActAgent(llm, toolset, build_prompt=lambda q, h: "\n".join(h) or q, max_iterations=3)
        partial = ReActResult(response="", answered=False, iterations=0)
        with self.assertRaisesRegex(RuntimeError, "model down"):
            agent.run("q", result=partial)
        self.assertEqual(len(partial.tool_calls), 1)

    def test_summarizing_needs_a_summary_prompt(self):
        with self.assertRaisesRegex(ValueError, "summary_prompt"):
            ReActAgent(FakeLLM([]), FakeToolset(self.tools, [], {}), build_prompt=lambda q, h: q, summarize_tool_response=True)


# ---------------------------------------------------------------------------------------------- 评分对拍
def patched_world():
    async def fake_call_once(gateway, server, tool, arguments, **kw):
        return McpResult(world.call(server, tool, arguments))

    async def fast_sleep(*a, **k):
        return None

    return [mock.patch("IMAct.benchmarks.mcp_universe.gateway.call_once", fake_call_once), mock.patch("asyncio.sleep", fast_sleep),
            mock.patch.dict(os.environ, {**ENV, "EVAL_REMOTE_URL": "", "MCP_GATEWAY_ADDRESS": "https://gw.invalid"})]


class EvaluationGoldenTests(unittest.TestCase):
    def test_all_evaluators_of_all_tasks_judge_like_upstream_on_a_fake_world(self):
        """729 条评估(排除要联网的 yfinance / playwright.check_ / weather / 裁判)× 6 种回答,共 3654 个判定,逐个与上游一致。"""
        want = load_gz("mcp_universe_evaluations")
        networked = ("yfinance.", "playwright.check_", "weather.", "google_search.")
        patches = patched_world()
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])

        async def go():
            got = {}
            for domain in T.domain_names():
                for path in T.load_domain(domain).task_paths:
                    raw = json.loads((T.TASKS / path).read_text(encoding="utf-8"))
                    fmt = raw.get("output_format") or {}
                    answers = ["plain text, not json", "{}", json.dumps(fmt), json.dumps(fmt).replace("[NUMBER]", "3"),
                               "```json\n" + json.dumps(fmt) + "\n```", json.dumps({"x": [1, 2, 3]})]
                    for ei, ev in enumerate(raw["evaluators"]):
                        if ev.get("op", "").startswith(networked):
                            continue
                        for ai, ans in enumerate(answers):
                            r = await Evaluator(json.loads(json.dumps(ev)), Context()).evaluate(ans)
                            got[f"{path}#{ei}#{ai}"] = [r.passed, r.reason, r.error.split("\n")[0] if r.error else ""]
            return got

        got = run_async(go())
        self.assertEqual(len(got), len(want))
        self.assertEqual(got, want)
        self.assertGreater(sum(v[0] for v in got.values()), 500)  # 确认确实走到了"通过"的分支,不是全部提前失败


class TaskGoldenTests(unittest.TestCase):
    @mock.patch.dict(os.environ, ENV)  # 上游渲染嵌套的 op_args 时只看进程环境变量
    def test_every_task_renders_like_upstream(self):
        want = load_gz("mcp_universe_tasks")
        got = {}
        for domain in T.domain_names():
            for path in T.load_domain(domain).task_paths:
                t = T.load_task(path, domain, ENV)
                got[f"{domain}/mcpuniverse/{path}"] = {
                    "question": t.question, "output_format": t.output_format, "servers": t.mcp_servers, "use_specified_server": t.use_specified_server,
                    "evaluators": [Evaluator(e, Context(ENV)).config.model_dump() for e in t.evaluators],
                    "cleanups": [{"server": c.get("server", ""), "tool": c.get("tool", ""), "cleanup_func": c.get("cleanup_func", ""),
                                  "cleanup_args": c.get("cleanup_args", {})} for c in t.cleanups]}
        self.assertEqual(len(got), 212)
        self.assertEqual(json.loads(json.dumps(got, sort_keys=True, default=str)), want)

    def test_domain_sizes_are_the_official_split(self):
        self.assertEqual({d: len(T.load_domain(d).task_paths) for d in T.domain_names()},
                         {"web_search": 55, "location_navigation": 45, "repository_management": 33, "financial_analysis": 40, "browser_automation": 39})

    def test_side_effect_tasks_are_exactly_the_ones_with_cleanups(self):
        counts = {d: sum(t.has_side_effects for t in T.load_tasks(d, ENV)) for d in T.domain_names()}
        self.assertEqual(counts, {"web_search": 5, "location_navigation": 0, "repository_management": 33, "financial_analysis": 0, "browser_automation": 5})

    def test_unknown_domain_and_task_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "不支持的领域"):
            T.load_domain("3d_design")
        tasks = T.load_tasks("financial_analysis", ENV)
        with self.assertRaisesRegex(ValueError, "没有这些 task"):
            T.select(tasks, task_ids=["nope"])
        self.assertEqual(len(T.select(tasks, num_tasks=3)), 3)
        self.assertEqual([t.id for t in T.select(tasks, task_ids=["yfinance_task_0002"])], ["yfinance_task_0002"])
        with self.assertRaisesRegex(ValueError, ">= 1"):
            T.select(tasks, num_tasks=0)


# ---------------------------------------------------------------------------------------------- 评分引擎
class EngineTests(unittest.TestCase):
    def evaluate(self, config, answer, context=None):
        return run_async(Evaluator(config, context or Context()).evaluate(answer))

    def test_chain_functions_and_comparisons(self):
        answer = json.dumps({"routes": [{"cities": [1, 2, 3, 4]}, {"cities": [1, 2, 3, 4]}], "n": 5})
        self.assertTrue(self.evaluate({"func": "json -> get(routes) -> len", "op": "=", "value": 2}, answer).passed)
        self.assertTrue(self.evaluate({"func": "json -> get(routes) -> foreach -> get(cities) -> len", "op": "=", "value": 4}, answer).passed)
        self.assertFalse(self.evaluate({"func": "json -> get(routes) -> foreach -> get(cities) -> len", "op": "=", "value": 3}, answer).passed)
        self.assertTrue(self.evaluate({"func": "json -> get(n)", "op": ">=", "value": 5}, answer).passed)
        self.assertTrue(self.evaluate({"func": "json -> get(n)", "op": "in", "value": [4, 5]}, answer).passed)
        self.assertTrue(self.evaluate({"func": "json -> get(routes) -> foreach -> get(cities)", "op": "contain", "value": 3}, answer).passed)
        self.assertTrue(self.evaluate({"func": "json"}, answer).passed)  # 没有 op:取值链不出错就算通过

    def test_one_failing_item_fails_the_whole_evaluator_with_its_reason(self):
        r = self.evaluate({"func": "json -> get(xs) -> foreach", "op": "<", "value": 3}, '{"xs": [1, 2, 9, 0]}')
        self.assertEqual((r.passed, r.reason), (False, "output is not less than ground-truth"))

    def test_bad_json_and_broken_chains_are_recorded_not_raised(self):
        r = self.evaluate({"func": "json -> get(a)", "op": "="}, "not json")
        self.assertEqual((r.passed, r.reason), (False, "JSON decoding error"))
        r = self.evaluate({"func": "json -> get(a)", "op": "="}, "[1]")
        self.assertEqual((r.passed, r.reason), (False, "Execution error"))
        self.assertIn("not a dict", r.error)

    def test_unknown_function_or_op_is_a_configuration_error(self):
        with self.assertRaisesRegex(AssertionError, "Unknown func"):
            Evaluator({"func": "nope"})
        with self.assertRaisesRegex(AssertionError, "Unknown comparison op"):
            Evaluator({"func": "json", "op": "no.such"})

    def test_environment_variables_are_rendered_into_value_and_op_args_and_unknown_ones_are_kept(self):
        ev = Evaluator({"func": "json", "op": "=", "value": "{{MY_VAR}}/x", "op_args": {"k": ["{{MY_VAR}}", "{{ UNSET_VAR_FOR_TEST }}"]}}, Context({"MY_VAR": "v1"}))
        self.assertEqual(ev.config.value, "v1/x")
        self.assertEqual(ev.config.op_args, {"k": ["v1", "{{ UNSET_VAR_FOR_TEST }}"]})

    def test_context_env_wins_over_the_process_environment(self):
        with mock.patch.dict(os.environ, {"CTX_TEST": "process"}):
            self.assertEqual(Context({"CTX_TEST": "ctx"}).get_env("CTX_TEST"), "ctx")
            self.assertEqual(Context().get_env("CTX_TEST"), "process")
            self.assertEqual(Context().get_env("CTX_TEST_MISSING", "dflt"), "dflt")

    def test_every_op_used_by_the_tasks_is_registered(self):
        for domain in T.domain_names():
            for t in T.load_tasks(domain, ENV):
                for e in t.evaluators:
                    self.assertTrue(e.get("op", "") == "" or e["op"] in COMPARISON_FUNCTIONS, (t.id, e.get("op")))

    def test_remote_ops_are_sent_to_the_remote_server_and_others_stay_local(self):
        sent = []

        class FakeClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                sent.append((url, json, headers))
                return SimpleNamespace(status_code=200, json=lambda: {"passed": True, "reason": "remote says ok"}, text="")

        ctx = Context({"EVAL_REMOTE_URL": "https://eval.example.com/", "EVAL_REMOTE_PREFIXES": "yfinance., playwright.check_", "MCP_GATEWAY_TOKEN": "tok"})
        with mock.patch("IMAct.evaluation.mcp_universe.httpx.AsyncClient", FakeClient):
            r = self.evaluate({"func": "json", "op": "yfinance.check_portfolio_task_output", "op_args": {"tickers": ["MSFT"]}}, '{"total value": 1}', ctx)
            self.assertTrue(r.passed)
            self.assertEqual(sent[0][0], "https://eval.example.com/eval/compare")
            self.assertEqual(sent[0][1]["op"], "yfinance.check_portfolio_task_output")
            self.assertEqual(sent[0][1]["a"], {"total value": 1})
            self.assertEqual(sent[0][2], {"Authorization": "Bearer tok"})
            self.assertFalse(self.evaluate({"func": "json -> get(a)", "op": "=", "value": 2}, '{"a": 1}', ctx).passed)
            self.assertEqual(len(sent), 1)  # "=" 不在白名单里,本地比较

    def test_a_remote_server_error_is_an_execution_error(self):
        class Bad:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                return SimpleNamespace(status_code=502, text="bad gateway")

        ctx = Context({"EVAL_REMOTE_URL": "https://e", "EVAL_REMOTE_PREFIXES": "yfinance."})
        with mock.patch("IMAct.evaluation.mcp_universe.httpx.AsyncClient", Bad):
            r = self.evaluate({"func": "json", "op": "yfinance.check_total_revenue", "op_args": {}}, "{}", ctx)
        self.assertEqual((r.passed, r.reason), (False, "Execution error"))
        self.assertIn("502", r.error)

    def test_function_result_is_unwrapped_by_comparisons(self):
        self.assertEqual(run_async(COMPARISON_FUNCTIONS["="](FunctionResult(result=3), FunctionResult(result=3))), (True, ""))


# ---------------------------------------------------------------------------------------------- 经网关查询的 shim
class GatewayShimTests(unittest.TestCase):
    def test_results_expose_sdk_style_attributes_and_missing_fields_raise_attribute_error(self):
        from IMAct.benchmarks.mcp_universe.gateway import ToolOutput

        raw = next(r for r in (world.call("github", "get_file_contents", {"owner": "o", "repo": "r", "path": f"p{i}", "ref": "main"}) for i in range(30))
                   if not r["isError"])
        out = ToolOutput(McpResult(raw))
        self.assertFalse(out.isError)
        self.assertEqual(out.content[0].type, "text")
        self.assertTrue(out.content[1].resource.text)
        with self.assertRaises(AttributeError):
            out.content[0].resource

    def test_github_file_lookups_name_the_branch_for_both_server_generations(self):
        """网关上的 github-mcp-server 只认 branch,上游固定的 0.5.0 只认 ref;只传 ref 时新版会悄悄读默认分支(真机误判过)。"""
        from IMAct.benchmarks.mcp_universe.checks import github

        sent = []

        async def capture(gateway, server, tool, arguments, **kw):
            sent.append(arguments)
            return McpResult({"content": [{"type": "text", "text": "ok"}, {"type": "resource", "resource": {"uri": "x", "text": "body"}}], "isError": False})

        with mock.patch("IMAct.benchmarks.mcp_universe.gateway.call_once", capture):
            run_async(github.github__get_file_contents("me", "r", "app.js", "feature-itinerary", context=Context(gateway=GatewayConfig("https://gw"))))
            run_async(github.github__get_file_contents("me", "r", "app.js", None, context=Context(gateway=GatewayConfig("https://gw"))))
        self.assertEqual(sent[0], {"owner": "me", "repo": "r", "path": "app.js", "ref": "feature-itinerary", "branch": "feature-itinerary"})
        self.assertEqual(sent[1], {"owner": "me", "repo": "r", "path": "app.js"})  # 没指定分支:两个都不传

    def test_notion_search_sends_a_plain_string_query_and_a_separate_filter(self):
        """网关上较新的 Notion server 拒绝"整个请求体序列化后塞进 query"的上游写法(真机 400),所以这里必须是纯字符串。"""
        from IMAct.benchmarks.mcp_universe.checks import notion

        sent = []

        async def capture(gateway, server, tool, arguments, **kw):
            sent.append((server, tool, arguments))
            return McpResult({"content": [{"type": "text", "text": json.dumps({"results": []})}], "isError": False})

        with mock.patch("IMAct.benchmarks.mcp_universe.gateway.call_once", capture):
            run_async(notion.notion__search_page("Performance of My Favorite Player", context=Context(gateway=GatewayConfig("https://gw"))))
        self.assertEqual(sent, [("notion", "API-post-search", {"query": "Performance of My Favorite Player", "filter": {"property": "object", "value": "page"}})])
        self.assertIsInstance(sent[0][2]["query"], str)
        self.assertFalse(sent[0][2]["query"].startswith("{"))

    def test_infrastructure_failures_are_recorded_on_the_context_and_still_raised(self):
        from IMAct.benchmarks.mcp_universe.gateway import MCPManager

        async def boom(*a, **k):
            raise McpError("连接 MCP server 失败")

        ctx = Context(gateway=GatewayConfig("https://gw"))
        with mock.patch("IMAct.benchmarks.mcp_universe.gateway.call_once", boom):
            with self.assertRaises(McpError):
                run_async(MCPManager(ctx).execute("github", "list_branches", {}))
        self.assertEqual(len(ctx.infra_errors), 1)
        self.assertIn("github.list_branches", ctx.infra_errors[0])

    def test_a_tool_level_error_result_is_a_normal_result_not_an_infrastructure_error(self):
        from IMAct.benchmarks.mcp_universe.gateway import MCPManager

        async def tool_error(*a, **k):
            return McpResult(world.error("not found"))

        ctx = Context(gateway=GatewayConfig("https://gw"))
        with mock.patch("IMAct.benchmarks.mcp_universe.gateway.call_once", tool_error):
            out = run_async(MCPManager(ctx).execute("github", "list_branches", {}))
        self.assertTrue(out.isError)
        self.assertEqual(ctx.infra_errors, [])

    def test_an_evaluator_that_swallows_a_gateway_failure_still_leaves_a_trace(self):
        """上游的 github 评估函数把网关异常吞成"没找到";这里 task 仍然判为没通过,但 infra_errors 能让调用方看出这不是 agent 的错。"""
        async def boom(*a, **k):
            raise McpError("down")

        ctx = Context(gateway=GatewayConfig("https://gw"))
        with mock.patch("IMAct.benchmarks.mcp_universe.gateway.call_once", boom):
            r = run_async(Evaluator({"func": "raw", "op": "github.check_repository", "op_args": "testuser/x"}, ctx).evaluate("{}"))
        self.assertFalse(r.passed)
        self.assertEqual(len(ctx.infra_errors), 1)

    def test_the_judge_comes_from_the_context_and_its_failure_is_recorded(self):
        ctx = Context()
        r = run_async(Evaluator({"func": "raw", "op": "google_search.llm_as_a_judge", "op_args": {"question": "q", "correct_answer": "a"}}, ctx).evaluate("a"))
        self.assertFalse(r.passed)  # 没有裁判:不能算通过
        calls = []

        class Judge:
            def chat(self, messages, **kw):
                calls.append(messages)
                return SimpleNamespace(content="extracted_final_answer: a\nreasoning: same\ncorrect: yes")

        ctx = Context(judge=Judge())
        r = run_async(Evaluator({"func": "raw", "op": "google_search.llm_as_a_judge", "op_args": {"question": "Who?", "correct_answer": "a"}}, ctx).evaluate("a"))
        self.assertTrue(r.passed)
        self.assertEqual([m.role for m in calls[0]], ["system", "user"])
        self.assertIn("[question]: Who?", calls[0][1].content)

        class Wrong(Judge):
            def chat(self, messages, **kw):
                return SimpleNamespace(content="correct: no")

        r = run_async(Evaluator({"func": "raw", "op": "google_search.llm_as_a_judge", "op_args": {"question": "Who?", "correct_answer": "a"}}, Context(judge=Wrong())).evaluate("b"))
        self.assertEqual((r.passed, r.reason), (False, "output is not equal to ground-truth"))

        class Down(Judge):
            def chat(self, messages, **kw):
                raise RuntimeError("judge unreachable")

        down = Context(judge=Down())
        r = run_async(Evaluator({"func": "raw", "op": "google_search.llm_as_a_judge", "op_args": {"question": "q", "correct_answer": "a"}}, down).evaluate("a"))
        self.assertFalse(r.passed)
        self.assertTrue(any("judge unreachable" in e for e in down.infra_errors))


# ---------------------------------------------------------------------------------------------- 清理
class CleanupTests(unittest.TestCase):
    def test_argument_parsing_matches_upstream(self):
        want = json.loads((FIXTURES / "mcp_universe_cleanup.json").read_text(encoding="utf-8"))
        calls = [
            {"arguments": {"name": "travel-planner-app", "private": False}, "response": {"content": [{"type": "text", "text": json.dumps({"id": "abc-123", "url": "u"})}], "isError": False}},
            {"arguments": {"repo": "TravelPlanner", "owner": "OSU"}, "response": {"content": [{"type": "text", "text": "{not json"}]}},
            {"arguments": {"name": "x"}, "response": ""},
        ]
        args = [{"repo": "$name"}, {"repo": "$repo"}, {"page": "$return -> get(content) -> array(0) -> get(text) -> json -> get(id)"},
                {"a": ["$name", 3, "plain"], "b": {"c": "$name"}}, "$name", "no dollar", {"x": "pre$name"}, {"x": "$missing"},
                {"x": "$return -> bogus(1)"}, {"x": "$return -> get(content) -> array(5)"}, {"x": "$name -> get(k)"}, {"x": " $name "}]
        got = {}
        for i, call in enumerate(calls + [None]):
            for j, a in enumerate(args):
                try:
                    got[f"{i}/{j}"] = ["ok", cleanups_mod.parse_cleanup_args(a, call)]
                except Exception as e:  # noqa: BLE001
                    got[f"{i}/{j}"] = ["err", type(e).__name__, str(e)[:80]]
        self.assertEqual(json.loads(json.dumps(got, sort_keys=True, default=str)), want)

    def run_cleanups(self, cleanups, calls, **funcs):
        executed = []

        def make(name):
            async def fn(**kw):
                executed.append((name, {k: v for k, v in kw.items() if k != "context"}))
                if name in funcs.get("fail", ()):
                    raise RuntimeError("nope")
                return "deleted"

            return fn

        registry = {("github", "delete_repository"): make("github.delete_repository"), ("notion", "delete_page"): make("notion.delete_page")}
        with mock.patch.dict(cleanups_mod.CLEANUP_FUNCTIONS, registry, clear=True):
            outcomes = run_async(cleanups_mod.run_cleanups(cleanups, calls, Context()))
        return executed, outcomes

    def test_each_created_resource_is_cleaned_latest_first(self):
        cleanups = [{"server": "github", "tool": "create_repository", "cleanup_func": "delete_repository", "cleanup_args": {"repo": "$name"}}]
        calls = [{"seq": 0, "server": "github", "tool_name": "create_repository", "arguments": {"name": "a"}, "response": "", "error": ""},
                 {"seq": 1, "server": "github", "tool_name": "get_me", "arguments": {}, "response": "", "error": ""},
                 {"seq": 2, "server": "github", "tool_name": "create_repository", "arguments": {"name": "b"}, "response": "", "error": ""}]
        executed, outcomes = self.run_cleanups(cleanups, calls)
        self.assertEqual([e[1]["repo"] for e in executed], ["b", "a"])
        self.assertEqual(len(outcomes), 2)

    def test_calls_that_created_nothing_are_skipped_not_reported_as_cleanup_failures(self):
        cleanups = [{"server": "notion", "tool": "API-post-page", "cleanup_func": "delete_page",
                     "cleanup_args": {"page": "$return -> get(content) -> array(0) -> get(text) -> json -> get(id)"}}]
        notion_400 = {"content": [{"type": "text", "text": json.dumps({"status": 400, "object": "error", "code": "validation_error", "message": "bad parent"})}], "isError": False}
        created = {"content": [{"type": "text", "text": json.dumps({"object": "page", "id": "page-2"})}], "isError": False}
        flagged = {"content": [{"type": "text", "text": "boom"}], "isError": True}
        calls = [{"seq": 0, "server": "notion", "tool_name": "API-post-page", "arguments": {}, "response": notion_400, "error": ""},
                 {"seq": 1, "server": "notion", "tool_name": "API-post-page", "arguments": {}, "response": flagged, "error": ""},
                 {"seq": 2, "server": "notion", "tool_name": "API-post-page", "arguments": {}, "response": "", "error": "Connection lost"},
                 {"seq": 3, "server": "notion", "tool_name": "API-post-page", "arguments": {}, "response": created, "error": ""}]
        executed, outcomes = self.run_cleanups(cleanups, calls)
        self.assertEqual(executed, [("notion.delete_page", {"page": "page-2"})])
        self.assertEqual(sum("SKIPPED" in o for o in outcomes), 3)
        self.assertFalse(any("FAILED" in o for o in outcomes))

    def test_a_response_that_looks_unusual_but_may_hold_a_resource_is_still_cleaned_or_reported(self):
        """判断"没建出资源"只认明确的失败信号;读不懂的返回仍然尝试清理,读不出 id 就如实报 FAILED,不会悄悄放过。"""
        cleanups = [{"server": "notion", "tool": "API-post-page", "cleanup_func": "delete_page",
                     "cleanup_args": {"page": "$return -> get(content) -> array(0) -> get(text) -> json -> get(id)"}}]
        odd = {"content": [{"type": "text", "text": json.dumps({"object": "page", "no_id_here": 1})}], "isError": False}
        executed, outcomes = self.run_cleanups(cleanups, [{"seq": 0, "server": "notion", "tool_name": "API-post-page", "arguments": {}, "response": odd, "error": ""}])
        self.assertEqual(executed, [])
        self.assertTrue(outcomes[0].startswith("notion.delete_page: FAILED"))

    def test_a_failing_cleanup_is_reported_and_does_not_stop_the_others(self):
        cleanups = [{"server": "github", "tool": "create_repository", "cleanup_func": "delete_repository", "cleanup_args": {"repo": "$name"}}]
        calls = [{"seq": i, "server": "github", "tool_name": "create_repository", "arguments": {"name": n}, "response": "", "error": ""} for i, n in enumerate("ab")]
        executed, outcomes = self.run_cleanups(cleanups, calls, fail=("github.delete_repository",))
        self.assertEqual(len(executed), 2)
        self.assertTrue(all("FAILED nope" in o for o in outcomes))

    def test_nothing_is_cleaned_when_the_tool_was_never_called(self):
        cleanups = [{"server": "github", "tool": "create_repository", "cleanup_func": "delete_repository", "cleanup_args": {"repo": "$name"}}]
        executed, outcomes = self.run_cleanups(cleanups, [])
        self.assertEqual((executed, outcomes), ([], []))

    def test_notion_pages_are_cleaned_by_the_id_in_the_create_response(self):
        cleanups = [{"server": "notion", "tool": "API-post-page", "cleanup_func": "delete_page",
                     "cleanup_args": {"page": "$return -> get(content) -> array(0) -> get(text) -> json -> get(id)"}}]
        response = {"content": [{"type": "text", "text": json.dumps({"id": "page-1"})}], "isError": False}
        executed, _ = self.run_cleanups(cleanups, [{"seq": 0, "server": "notion", "tool_name": "API-post-page", "arguments": {}, "response": response, "error": ""}])
        self.assertEqual(executed, [("notion.delete_page", {"page": "page-1"})])

    def test_github_deletion_is_confined_to_the_configured_account_and_to_plain_repo_names(self):
        ctx = Context({"GITHUB_PERSONAL_ACCOUNT_NAME": "", "GITHUB_PERSONAL_ACCESS_TOKEN": "t"})
        with mock.patch.dict(os.environ, {"GITHUB_PERSONAL_ACCOUNT_NAME": ""}):
            with self.assertRaisesRegex(ValueError, "owner is empty"):
                run_async(cleanups_mod.github_delete_repository("r", context=ctx))
        ctx = Context({"GITHUB_PERSONAL_ACCOUNT_NAME": "me", "GITHUB_PERSONAL_ACCESS_TOKEN": "t"})
        with self.assertRaisesRegex(ValueError, "unexpected name"):
            run_async(cleanups_mod.github_delete_repository("someone-else/their-repo", context=ctx))

    def test_github_deletion_calls_the_rest_api_for_the_configured_account(self):
        seen = []

        class FakeClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def delete(self, url, headers=None):
                seen.append((url, headers["Authorization"]))
                return SimpleNamespace(status_code=204)

        ctx = Context({"GITHUB_PERSONAL_ACCOUNT_NAME": "me", "GITHUB_PERSONAL_ACCESS_TOKEN": "tok"})
        with mock.patch("IMAct.benchmarks.mcp_universe.cleanups.httpx.AsyncClient", FakeClient):
            msg = run_async(cleanups_mod.github_delete_repository("travel-planner-app", context=ctx))
        self.assertEqual(seen, [("https://api.github.com/repos/me/travel-planner-app", "Bearer tok")])
        self.assertIn("successfully deleted", msg)


class CleanupWithoutLocalKeysTests(unittest.TestCase):
    """密钥只在网关那一侧时的清理行为。"""

    def test_github_without_a_local_token_fails_loudly_instead_of_sending_an_unauthenticated_request(self):
        ctx = Context({"GITHUB_PERSONAL_ACCOUNT_NAME": "me", "GITHUB_PERSONAL_ACCESS_TOKEN": ""})
        with mock.patch.dict(os.environ, {"GITHUB_PERSONAL_ACCESS_TOKEN": ""}):
            with mock.patch("IMAct.benchmarks.mcp_universe.cleanups.httpx.AsyncClient") as client:
                with self.assertRaisesRegex(RuntimeError, "me/travel-planner-app 没有被删除"):
                    run_async(cleanups_mod.github_delete_repository("travel-planner-app", context=ctx))
                client.assert_not_called()

    def test_the_failure_is_visible_in_the_run_outcomes(self):
        cleanups = [{"server": "github", "tool": "create_repository", "cleanup_func": "delete_repository", "cleanup_args": {"repo": "$name"}}]
        calls = [{"seq": 0, "server": "github", "tool_name": "create_repository", "arguments": {"name": "x"}, "response": "", "error": ""}]
        ctx = Context({"GITHUB_PERSONAL_ACCOUNT_NAME": "me"})
        with mock.patch.dict(os.environ, {"GITHUB_PERSONAL_ACCESS_TOKEN": ""}):
            outcomes = run_async(cleanups_mod.run_cleanups(cleanups, calls, ctx))
        self.assertEqual(len(outcomes), 1)
        self.assertIn("FAILED", outcomes[0])
        self.assertIn("手动删除", outcomes[0])

    def test_notion_without_a_local_key_archives_the_page_through_the_gateway(self):
        seen = []

        async def fake_call_once(gateway, server, tool, arguments, **kw):
            seen.append((server, tool, arguments))
            return McpResult({"content": [{"type": "text", "text": "{}"}], "isError": False})

        ctx = Context({"NOTION_API_KEY": ""}, gateway=GatewayConfig("https://gw"))
        with mock.patch.dict(os.environ, {"NOTION_API_KEY": ""}):
            with mock.patch.object(cleanups_mod, "call_once", fake_call_once):
                msg = run_async(cleanups_mod.notion_delete_page("page-1", context=ctx))
        self.assertEqual(seen, [("notion", "API-patch-page", {"page_id": "page-1", "in_trash": True})])
        self.assertIn("via gateway", msg)

    def test_a_gateway_notion_error_is_reported_not_swallowed(self):
        async def fake_call_once(*a, **k):
            return McpResult({"content": [{"type": "text", "text": "object_not_found"}], "isError": True})

        ctx = Context({"NOTION_API_KEY": ""}, gateway=GatewayConfig("https://gw"))
        with mock.patch.dict(os.environ, {"NOTION_API_KEY": ""}):
            with mock.patch.object(cleanups_mod, "call_once", fake_call_once):
                with self.assertRaisesRegex(RuntimeError, "object_not_found"):
                    run_async(cleanups_mod.notion_delete_page("page-1", context=ctx))


# ---------------------------------------------------------------------------------------------- runner
class FakeLoopToolset:
    """替身 McpToolset:不连网关。``calls`` 记录所有实例,方便检查是否关闭。"""

    instances = []

    def __init__(self, gateway, servers, bridge, **options):
        self.tools = {s["name"]: [McpTool("search", "d", {"type": "object", "properties": {}})] for s in servers}
        self.closed = False
        self.opened = False
        type(self).instances.append(self)

    def open(self):
        if type(self).fail_open:
            raise McpError("gateway down")
        self.opened = True
        return self

    def call(self, server, tool, arguments):
        return McpResult({"content": [{"type": "text", "text": json.dumps({"id": "page-9"})}], "isError": False})

    def close(self):
        self.closed = True

    fail_open = False


class ImmediateBridge:
    def run(self, coro, timeout=None):
        return asyncio.run(coro)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        FakeLoopToolset.instances, FakeLoopToolset.fail_open = [], False
        patch = mock.patch.object(runner_mod, "McpToolset", FakeLoopToolset)
        patch.start()
        self.addCleanup(patch.stop)
        self.domain = T.load_domain("financial_analysis")

    def runner(self, llm_script, **kw):
        llm = SimpleNamespace(chat=lambda messages, **k: SimpleNamespace(content=llm_script.pop(0), usage={"prompt_tokens": 10, "completion_tokens": 5}))
        return runner_mod.McpUniverseRunner(self.domain, llm, ImmediateBridge(), GatewayConfig("https://gw"), **kw)

    def task(self, evaluators, **kw):
        base = dict(id="t1", path="x/t1.json", domain="financial_analysis", category="general", question="Q?", output_format={"v": "[NUMBER]"},
                    mcp_servers=[{"name": "yfinance"}], evaluators=evaluators, cleanups=[], use_specified_server=True)
        base.update(kw)
        return T.McpTask(**base)

    def test_a_correct_answer_scores_one_and_is_fully_recorded(self):
        run = self.runner(['{"thought": "t", "answer": {"v": 3}}']).run_task(self.task([{"func": "json -> get(v)", "op": "=", "value": 3}]), run_id="r")
        self.assertEqual(run.reward_info.reward, 1.0)
        info = run.reward_info.info
        self.assertEqual((info["success"], info["evaluator_score"], info["evaluations_passed"], info["iterations"]), (True, 1.0, 1, 1))
        self.assertEqual(run.response, '{"v": 3}')
        self.assertEqual(run.usage, {"prompt_tokens": 10, "completion_tokens": 5, "llm_calls": 1})
        self.assertEqual(info["llm_calls"], 1)
        self.assertTrue(FakeLoopToolset.instances[0].closed)
        json.dumps(run.model_dump(mode="json"))  # 可以落盘

    def test_the_score_counts_a_task_successful_only_when_every_evaluator_passes_and_ae_is_the_fraction(self):
        run = self.runner(['{"thought": "t", "answer": {"v": 3}}']).run_task(
            self.task([{"func": "json -> get(v)", "op": "=", "value": 3}, {"func": "json -> get(v)", "op": "=", "value": 4}]), run_id="r")
        self.assertEqual(run.reward_info.reward, 0.0)
        self.assertEqual(run.reward_info.info["evaluator_score"], 0.5)

    def test_giving_up_without_an_answer_fails_the_evaluators_but_is_not_an_error(self):
        run = self.runner(['{"thought": "t", "result": "hmm"}'] * 20).run_task(self.task([{"func": "json -> get(v)", "op": "=", "value": 3}]), run_id="r")
        self.assertEqual((run.reward_info.reward, run.answered, run.iterations), (0.0, False, 20))
        self.assertEqual(run.response, "I'm sorry, but I couldn't find a satisfactory answer within the allowed number of iterations.")  # 评估器看到的是这句

    def test_a_gateway_failure_while_connecting_propagates_so_the_batch_retries_the_task(self):
        FakeLoopToolset.fail_open = True
        with self.assertRaisesRegex(McpError, "gateway down"):
            self.runner([]).run_task(self.task([]), run_id="r")
        self.assertTrue(FakeLoopToolset.instances[0].closed)

    def test_infrastructure_errors_during_evaluation_flag_an_evaluation_error(self):
        async def boom(*a, **k):
            raise McpError("down")

        task = self.task([{"func": "raw", "op": "github.check_repository", "op_args": "me/x"}])
        with mock.patch("IMAct.benchmarks.mcp_universe.gateway.call_once", boom):
            run = self.runner(['{"thought": "t", "answer": "ok"}']).run_task(task, run_id="r")
        self.assertEqual(run.reward_info.reward, 0.0)
        self.assertTrue(run.reward_info.info["evaluation_error"])
        self.assertIn("github.search_repositories", run.infra_errors[0])

    def test_side_effect_tasks_can_be_forbidden_for_a_run(self):
        cleanups = [{"server": "github", "tool": "create_repository", "cleanup_func": "delete_repository", "cleanup_args": {"repo": "$name"}}]
        task = self.task([], cleanups=cleanups)
        with self.assertRaisesRegex(RuntimeError, "--skip-side-effects"):
            self.runner([], side_effects=False).run_task(task, run_id="r")
        self.assertEqual(FakeLoopToolset.instances, [])  # 连服务都没连

    def test_cleanup_runs_with_what_the_agent_created_and_even_when_the_agent_crashes(self):
        cleanups = [{"server": "yfinance", "tool": "search", "cleanup_func": "delete_page",
                     "cleanup_args": {"page": "$return -> get(content) -> array(0) -> get(text) -> json -> get(id)"}}]
        task = self.task([{"func": "json", "op": ""}], cleanups=cleanups)
        deleted = []

        async def fake_delete(page, **kw):
            deleted.append(page)
            return "ok"

        action = '{"thought": "t", "action": {"server": "yfinance", "tool": "search", "arguments": {}}}'
        crashing = iter([action])

        def chat(messages, **kw):
            try:
                return SimpleNamespace(content=next(crashing), usage={})
            except StopIteration:
                raise RuntimeError("model down") from None

        llm = SimpleNamespace(chat=chat)
        runner = runner_mod.McpUniverseRunner(self.domain, llm, ImmediateBridge(), GatewayConfig("https://gw"))
        with mock.patch.dict(cleanups_mod.CLEANUP_FUNCTIONS, {("yfinance", "delete_page"): fake_delete}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "model down"):
                runner.run_task(task, run_id="r")
        self.assertEqual(deleted, ["page-9"])  # 重试前先把已创建的资源清掉
        self.assertTrue(FakeLoopToolset.instances[0].closed)

    def test_side_effect_tasks_never_overlap_in_time(self):
        cleanups = [{"server": "yfinance", "tool": "nothing", "cleanup_func": "x", "cleanup_args": {}}]
        active, peak, lock = [0], [0], threading.Lock()

        def chat(messages, **kw):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return SimpleNamespace(content='{"thought": "t", "answer": "a"}', usage={})

        runner = runner_mod.McpUniverseRunner(self.domain, SimpleNamespace(chat=chat), ImmediateBridge(), GatewayConfig("https://gw"), side_effects=True)
        threads = [threading.Thread(target=runner.run_task, args=(self.task([], id=f"t{i}", cleanups=cleanups),), kwargs={"run_id": "r"}) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(peak[0], 1)

    def test_a_task_without_a_specified_server_is_refused(self):
        with self.assertRaisesRegex(ValueError, "没有指定 MCP server"):
            self.runner([]).run_task(self.task([], use_specified_server=False), run_id="r")

    def test_a_hung_evaluator_times_out_and_is_flagged(self):
        async def hang(*a, **k):
            await asyncio.sleep(30)

        task = self.task([{"func": "raw", "op": "github.check_repository", "op_args": "me/x"}])
        runner = self.runner(['{"thought": "t", "answer": "ok"}'], eval_timeout=0.2)
        with mock.patch("IMAct.benchmarks.mcp_universe.gateway.call_once", hang):
            run = runner.run_task(task, run_id="r")
        self.assertEqual(run.reward_info.reward, 0.0)
        self.assertTrue(run.reward_info.info["evaluation_error"])


# ---------------------------------------------------------------------------------------------- CLI
class CliTests(unittest.TestCase):
    def parse(self, *argv):
        from IMAct.benchmarks.mcp_universe.run import build_parser

        return build_parser().parse_args(["--models", "m.yaml", *argv])

    def test_defaults(self):
        args = self.parse("--domain", "web_search")
        self.assertEqual((args.num_trials, args.max_concurrency, args.skip_side_effects, args.call_timeout), (1, 1, False, 60.0))

    def test_unknown_domain_is_rejected(self):
        for bad in (["--domain", "3d_design"], ["--domain", "hotel"]):
            with self.assertRaises(SystemExit):
                self.parse(*bad)

    def test_domain_all_means_the_five_domains(self):
        from IMAct.benchmarks.mcp_universe.run import selected_domains

        self.assertEqual(selected_domains(self.parse("--domain", "all")), T.domain_names())
        self.assertEqual(selected_domains(self.parse("--domain", "web_search", "web_search")), ["web_search"])

    def test_task_ids_need_a_single_domain(self):
        from IMAct.benchmarks.mcp_universe.run import select_tasks

        with self.assertRaisesRegex(ValueError, "单个 --domain"):
            select_tasks(self.parse("--domain", "web_search", "financial_analysis", "--task-ids", "x"), ["web_search", "financial_analysis"], ENV)

    def test_the_check_probes_playwright_with_a_real_call_and_reports_a_broken_browser(self):
        from IMAct.benchmarks.mcp_universe.run import probe_server

        class Toolset:
            def __init__(self, reply):
                self.reply, self.calls = reply, []

            def call(self, server, tool, arguments):
                self.calls.append(tool)
                return McpResult({"content": [{"type": "text", "text": self.reply}], "isError": False})

        broken = Toolset('### Error\nError: Browser "chrome-for-testing" is not installed')
        self.assertIn("not installed", probe_server("playwright", broken))
        self.assertEqual(broken.calls, ["browser_navigate"])
        healthy = Toolset("### Page\n- Page URL: about:blank")
        self.assertIsNone(probe_server("playwright", healthy))
        self.assertEqual(healthy.calls, ["browser_navigate", "browser_close"])
        self.assertIsNone(probe_server("yfinance", Toolset("anything")))  # 没有探测的 server 不调用

    def test_unresolved_variables_are_reported(self):
        from IMAct.benchmarks.mcp_universe.run import unresolved_variables

        tasks = [t for t in T.load_tasks("repository_management", {}) if "{{" in json.dumps(t.evaluators)]
        self.assertTrue(tasks)
        with mock.patch.dict(os.environ, {"GITHUB_PERSONAL_ACCOUNT_NAME": ""}):
            os.environ.pop("GITHUB_PERSONAL_ACCOUNT_NAME")
            self.assertIn("GITHUB_PERSONAL_ACCOUNT_NAME", unresolved_variables(tasks[0], {}))
        self.assertEqual([unresolved_variables(t, ENV) for t in tasks if unresolved_variables(t, ENV)], [])

    def test_local_network_evaluations_are_reported_unless_remote_evaluation_covers_them(self):
        from IMAct.benchmarks.mcp_universe.run import local_evaluation_needs

        tasks = T.load_tasks("financial_analysis", ENV) + T.load_tasks("location_navigation", ENV)
        self.assertGreater(local_evaluation_needs(tasks, {}).get("yfinance", 0), 30)
        remote = {"EVAL_REMOTE_URL": "https://e", "EVAL_REMOTE_PREFIXES": "yfinance.,playwright.check_"}
        self.assertNotIn("yfinance", local_evaluation_needs(tasks, remote))
        self.assertIn("weather", local_evaluation_needs(tasks, remote))  # 天气评估不在远程白名单里

    def test_summary_means_are_per_task_and_count_failures_separately(self):
        from IMAct.benchmarks.mcp_universe.run import summarize

        domains = {
            "web_search": {"results": [{"reward": 1.0, "evaluator_score": 1.0, "iterations": 4, "llm_calls": 7}, {"reward": 0.0, "evaluator_score": 0.5, "iterations": 6, "llm_calls": 11}], "failed": [],
                           "skipped": [], "avg_reward": 0.5, "num_evaluation_errors": 1},
            "financial_analysis": {"results": [{"reward": 0.0, "evaluator_score": 0.0, "iterations": 8, "llm_calls": 8}], "failed": [{"task_id": "x", "error": "boom"}], "skipped": [{}],
                                   "avg_reward": 0.0}}
        s = summarize(domains, run_id="r", agent_model="m", skipped_side_effects=["a", "b"])
        self.assertEqual((s["completed"], s["failed"], s["skipped"], s["evaluation_errors"]), (3, 1, 1, 1))
        self.assertAlmostEqual(s["SR"], 100 / 3)  # 百分数,与论文同口径
        self.assertAlmostEqual(s["AE"], 50.0)
        self.assertAlmostEqual(s["avg_iterations"], 6.0)
        # AS 只算成功的 task:只有 web_search 的第一个成功
        self.assertEqual((s["num_successful"], s["AS"], s["AS_llm_calls"]), (1, 4.0, 7.0))
        self.assertEqual((s["domains"]["web_search"]["AS"], s["domains"]["web_search"]["num_successful"]), (4.0, 1))
        self.assertIsNone(s["domains"]["financial_analysis"]["AS"])  # 没有成功的 task:没有 AS,而不是 0
        self.assertEqual(s["skipped_side_effects"], ["a", "b"])
        self.assertEqual(s["domains"]["financial_analysis"]["failed_tasks"][0]["task_id"], "x")

    def test_row_fields_carry_the_task_level_metrics_into_the_batch_summary(self):
        import tempfile

        from IMAct.benchmarks.mcp_universe.run import row_fields
        from IMAct.evaluation.base import RewardInfo
        from IMAct.harness.batch import run_jobs

        ri = RewardInfo(reward=1.0, info={"evaluator_score": 0.75, "iterations": 3, "answered": True, "evaluations_passed": 3, "evaluations_total": 4})
        sim = SimpleNamespace(model_dump=lambda mode=None: {"ok": 1, "reward_info": ri.model_dump(mode="json")}, reward_info=ri)
        with tempfile.TemporaryDirectory() as d:
            summary = run_jobs([SimpleNamespace(id="a")], lambda task, run_id, seed: sim, out=Path(d), run_id="r", row_fields=row_fields)
        self.assertEqual(summary["results"][0]["evaluator_score"], 0.75)
        self.assertEqual(summary["results"][0]["iterations"], 3)


class IndependenceTests(unittest.TestCase):
    def test_nothing_imports_the_upstream_package(self):
        root = Path(__file__).resolve().parent.parent / "IMAct"
        offenders = []
        for path in list(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "import mcpuniverse" in text or "from mcpuniverse" in text:
                offenders.append(str(path.relative_to(root)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
