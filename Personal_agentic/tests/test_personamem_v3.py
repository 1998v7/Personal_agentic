"""PersonaMem-v3 接入(llm_memory 模式 + 外部 Memory 服务)的测试。全部离线:不连真实网关、不调真实模型。

* 对拍:``fixtures/personamem_v3_upstream.json.gz`` 是上游 PersonaMem-v3 代码在 3 个 persona(25 / 36 / 115,共 490 道题,
  30 个题型)上、用 ``personamem_v3_fakes`` 的确定性假模型 / 假裁判 / 假记忆跑出的结果:每道题发给模型的 prompt、
  裁判 prompt(只存 sha256)、metrics、status、回答。这里用 IMAct 的移植版重放,必须逐项一致。需要 ``data/personamem_v3``
  里有这 3 个 persona(且与 fixture 生成时的 test.json 相同),否则跳过。
* 其余测试覆盖 Memory 接口的入参、失败处理、记忆截断、汇总口径、命令行端到端(本地假网关 + 假模型服务)。
"""

import gzip
import hashlib
import http.server
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from IMAct.benchmarks.personamem_v3 import data as D
from IMAct.benchmarks.personamem_v3 import runner as R
from IMAct.benchmarks.personamem_v3.scoring import summarize

from . import personamem_v3_fakes as F

FIXTURE = Path(__file__).parent / "fixtures" / "personamem_v3_upstream.json.gz"


def load_fixture():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
        return json.load(f)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def data_matches_fixture() -> tuple[bool, str]:
    fx = load_fixture()
    for uid, want in fx["test_json_sha256"].items():
        path = D.backend_dir() / uid / "test.json"
        if not path.is_file() or D.missing_files(uid):
            return False, f"需要 data/personamem_v3 里的 persona {uid}(python -m IMAct.benchmarks.personamem_v3.download --users 25 36 115)"
        if hashlib.sha256(path.read_bytes()).hexdigest() != want:
            return False, f"persona {uid} 的 test.json 与生成 fixture 时不同(数据集更新过),对拍基线需要重新生成"
    return True, ""


HAS_DATA, WHY = data_matches_fixture()


class FakeAdapterFactory:
    """替换 ``runner.QueryLLMAdapter``:每道题一个假模型 + 一个假裁判,记录 prompt。"""

    def __init__(self):
        self.current = {}

    def __call__(self, llm, stage):
        client = self.current["answer" if stage == "agent" else "judge"]
        return SimpleNamespace(query_llm=lambda prompt, *a, **k: client.query_llm(prompt), usage=lambda: {"calls": len(client.prompts)})


def truncate(resp):  # 上游 results.csv 的 4 KB 截断(fixture 里的回答是截断后的)
    if resp is None:
        return ""
    s = resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False)
    if len(s.encode("utf-8")) <= 4096:
        return s
    return s.encode("utf-8")[:4096].decode("utf-8", errors="replace") + "…[truncated]"


def strip_volatile(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if not any(s in k for s in ("duration", "latency", "elapsed", "_ms", "wall"))}


def fake_memory_query(**kw):
    return {"status": "ok", "context": F.fake_memory(kw["user_id"]), "error": None}


@unittest.skipUnless(HAS_DATA, WHY)
class UpstreamGoldenTests(unittest.TestCase):
    """最有价值的测试:IMAct 的移植版与上游 PersonaMem-v3 在同样输入下逐题一致。"""

    @classmethod
    def setUpClass(cls):
        from IMAct.benchmarks.personamem_v3.pm3.evaluation import inference_utils as IU
        from IMAct.benchmarks.personamem_v3.pm3.evaluation.tasks import over_personalization as op

        op._C1C_CLUSTER_CACHE.clear()  # 上游的 cluster 级缓存是模块级的;别的测试跑过的 cluster 不能带进对拍
        fx = load_fixture()
        cls.want = fx["rows"]
        factory = FakeAdapterFactory()
        cls.got = {}
        # fixture 是在没装 tiktoken 的环境(token 数按字符数 / 4)里生成的;装了 tiktoken(如 uni-agent 的 venv)记忆截断与
        # token 计数都会变,所以这里固定走字符数 / 4 的路径。
        with mock.patch.object(R, "QueryLLMAdapter", factory), mock.patch.object(IU, "_TIKTOKEN_ENC", False):
            for uid in fx["users"]:
                runner = R.PersonaMemRunner(backend_dir=D.backend_dir(), answer_llm=object(), judge_llm=object(), memory_query=fake_memory_query)
                for seq, item in enumerate(D.load_test(uid)):
                    factory.current = {"answer": F.FakeClient("answer"), "judge": F.FakeClient("judge")}
                    run = runner.run_item(uid, seq, item)
                    resp = truncate(run.agent_response)
                    cls.got[item["query_id"]] = {
                        "task_type": item["task_type"], "status": run.status, "metrics": strip_volatile(run.metrics),
                        "agent_response": sha(resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False, sort_keys=True)),
                        "answer_prompts": [sha(p) for p in factory.current["answer"].prompts],
                        "judge_prompts": sorted(sha(p) for p in factory.current["judge"].prompts),  # 上游对重复疲劳题并发评审,顺序不定
                    }

    def test_every_query_of_every_task_type_matches_upstream(self):
        self.assertEqual(set(self.got), set(self.want))
        self.assertEqual(len({v["task_type"] for v in self.want.values()}), 30)
        for qid, want in self.want.items():
            got = self.got[qid]
            for field in ("status", "metrics", "agent_response", "answer_prompts", "judge_prompts"):
                with self.subTest(qid=qid, field=field):
                    self.assertEqual(got[field], want[field])

    def test_the_comparison_really_exercises_the_scorers(self):
        n_answer = sum(len(v["answer_prompts"]) for v in self.want.values())
        n_judge = sum(len(v["judge_prompts"]) for v in self.want.values())
        self.assertGreater(n_answer, 500)
        self.assertGreater(n_judge, 500)
        self.assertTrue(all(v["status"] == "ok" for v in self.want.values()))
        self.assertGreater(sum(1 for v in self.want.values() if v["metrics"]), 400)


@unittest.skipUnless(HAS_DATA, WHY)
class MemoryProtocolTests(unittest.TestCase):
    """统一调用接口:query({bench, user_id, task:{query_id, user_query, timestamp}}),每道题一次。"""

    def run_one(self, item, *, memory_query, placeholder="empty", cap=R.DEFAULT_MEMORY_TOKEN_CAP, uid="25"):
        factory = FakeAdapterFactory()
        factory.current = {"answer": F.FakeClient("answer"), "judge": F.FakeClient("judge")}
        runner = R.PersonaMemRunner(backend_dir=D.backend_dir(), answer_llm=object(), judge_llm=object(), memory_query=memory_query,
                                    query_placeholder=placeholder, memory_token_cap=cap)
        with mock.patch.object(R, "QueryLLMAdapter", factory):
            run = runner.run_item(uid, 0, item)
        return run, factory.current["answer"].prompts

    @classmethod
    def setUpClass(cls):
        cls.items = D.load_test("25")
        cls.with_text = next(i for i in cls.items if i["task_type"] == "local_recommendation_geo_shift")
        cls.placeholder = next(i for i in cls.items if (i.get("user_query") or "").startswith("[system prompt]"))

    def test_query_carries_the_raw_query_id_text_and_unix_seconds(self):
        calls = []
        run, _ = self.run_one(self.with_text, memory_query=lambda **kw: (calls.append(kw), fake_memory_query(**kw))[1])
        self.assertEqual(calls, [{"user_id": "25", "query_id": self.with_text["query_id"], "user_query": self.with_text["user_query"],
                                  "timestamp": self.with_text["ts"]}])
        self.assertIsInstance(calls[0]["timestamp"], int)
        self.assertEqual(run.memory["status"], "ok")

    def test_placeholder_queries_send_an_empty_string_unless_verbatim_is_requested(self):
        calls = []
        self.run_one(self.placeholder, memory_query=lambda **kw: (calls.append(kw), fake_memory_query(**kw))[1])
        self.run_one(self.placeholder, memory_query=lambda **kw: (calls.append(kw), fake_memory_query(**kw))[1], placeholder="verbatim")
        self.assertEqual(calls[0]["user_query"], "")
        self.assertEqual(calls[1]["user_query"], self.placeholder["user_query"])

    def test_the_memory_context_and_the_calendar_reach_the_model_prompt(self):
        marker = "MEMORY-MARKER-7f3a"
        run, prompts = self.run_one(self.with_text, memory_query=lambda **kw: {"status": "ok", "context": f"- {marker}", "error": None})
        self.assertTrue(prompts)
        self.assertIn(marker, prompts[0])
        self.assertIn("# Calendar — current entries as of now", prompts[0])  # 官方 llm_memory 的日历块

    def test_empty_memory_is_answered_without_memory_not_treated_as_a_failure(self):
        run, prompts = self.run_one(self.with_text, memory_query=lambda **kw: {"status": "empty", "context": "", "error": None})
        self.assertEqual(run.memory["status"], "empty")
        self.assertEqual(run.status, "ok")
        self.assertTrue(prompts)

    def test_a_memory_error_stops_the_query_instead_of_answering_without_memory(self):
        with self.assertRaisesRegex(R.MemoryUnavailable, "快照尚未构建"):
            self.run_one(self.with_text, memory_query=lambda **kw: {"status": "error", "context": "", "error": "当前任务对应的记忆快照尚未构建完成"})

    def test_memory_is_capped_like_the_official_llm_memory_budget(self):
        big = "\n".join(f"- fact {i} " + "word " * 30 for i in range(400))
        run, prompts = self.run_one(self.with_text, memory_query=lambda **kw: {"status": "ok", "context": big, "error": None}, cap=500)
        self.assertTrue(run.memory["truncated"])
        self.assertLessEqual(run.memory["tokens"], 500)
        self.assertIn("- fact 0 ", prompts[0])
        self.assertNotIn("- fact 399 ", prompts[0])

    def test_the_memory_is_requested_once_per_query(self):
        calls = []
        self.run_one(self.with_text, memory_query=lambda **kw: (calls.append(kw), fake_memory_query(**kw))[1])
        self.assertEqual(len(calls), 1)


@unittest.skipUnless(HAS_DATA, WHY)
class FailureHandlingTests(unittest.TestCase):
    def runner(self, memory_query=fake_memory_query):
        return R.PersonaMemRunner(backend_dir=D.backend_dir(), answer_llm=object(), judge_llm=object(), memory_query=memory_query)

    def test_a_model_request_that_keeps_failing_propagates_so_the_batch_retries(self):
        from IMAct.retry import CallError

        def boom(llm, stage):
            def q(prompt, *a, **k):
                raise CallError("rate_limit", "429 after retries")
            return SimpleNamespace(query_llm=q, usage=lambda: {})

        item = next(i for i in D.load_test("25") if i["task_type"] == "local_recommendation_geo_shift")
        with mock.patch.object(R, "QueryLLMAdapter", boom):
            with self.assertRaises(Exception) as ctx:
                self.runner().run_item("25", 0, item)
        self.assertTrue(R._is_infrastructure_error(ctx.exception))

    def test_a_bug_inside_task_code_is_recorded_like_upstream_status_error_scored_zero(self):
        item = next(i for i in D.load_test("25") if i["task_type"] == "local_recommendation_geo_shift")
        factory = FakeAdapterFactory()
        factory.current = {"answer": F.FakeClient("answer"), "judge": F.FakeClient("judge")}
        with mock.patch.object(R, "QueryLLMAdapter", factory), mock.patch.object(R, "dispatch_single", side_effect=KeyError("held_out_idx")):
            run = self.runner().run_item("25", 0, item)
        self.assertEqual(run.status, "error")
        self.assertIn("KeyError", run.error)
        self.assertEqual(run.accuracy_pct, 0.0)  # 官方 _accuracy_value:status=error → 0

    def test_repetition_clusters_run_once_even_when_their_rows_run_concurrently(self):
        from IMAct.benchmarks.personamem_v3.pm3.evaluation.tasks import over_personalization as op

        items = [i for i in D.load_test("25") if i["task_type"] == "over_personalization_repetition_recsys"]
        cid = items[0]["instance_full"]["cluster_id"]
        rows = [i for i in items if i["instance_full"]["cluster_id"] == cid]
        self.assertGreater(len(rows), 1)
        op._C1C_CLUSTER_CACHE.pop(cid, None)
        answer_calls = []
        lock = threading.Lock()

        def factory(llm, stage):
            client = F.FakeClient("answer" if stage == "agent" else "judge")

            def q(prompt, *a, **k):
                if stage == "agent":
                    with lock:
                        answer_calls.append(prompt)
                return client.query_llm(prompt)
            return SimpleNamespace(query_llm=q, usage=lambda: {})

        runner = self.runner()
        with mock.patch.object(R, "QueryLLMAdapter", factory):
            threads = [threading.Thread(target=runner.run_item, args=("25", n, row)) for n, row in enumerate(rows)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        n_queries = len(rows[0]["instance_full"]["queries"])
        self.assertEqual(len(answer_calls), n_queries)  # 整个 cluster 只跑了一遍

    def test_retired_task_types_are_refused(self):
        item = dict(D.load_test("25")[0], task_type="daily_personalized_briefing")
        with self.assertRaisesRegex(ValueError, "弃用"):
            self.runner().run_item("25", 0, item)


class CapTokensTests(unittest.TestCase):
    def test_short_text_is_untouched_and_long_text_is_cut_at_a_line_boundary(self):
        self.assertEqual(R.cap_tokens("a\nb", 100), ("a\nb", False))
        text = "\n".join("line " + str(i) * 40 for i in range(50))
        cut, truncated = R.cap_tokens(text, 200)
        self.assertTrue(truncated)
        self.assertTrue(text.startswith(cut))
        self.assertTrue(cut.endswith(cut.split("\n")[-1]))
        self.assertLessEqual(R.count_tokens(cut), 200)

    def test_a_single_overlong_line_is_cut_by_characters(self):
        cut, truncated = R.cap_tokens("x" * 10000, 100)
        self.assertTrue(truncated)
        self.assertLessEqual(R.count_tokens(cut), 100)
        self.assertTrue(cut)


class ScoringTests(unittest.TestCase):
    def row(self, task_type, metrics, status="ok", acc=None, uid="1", qid=None, memory=None):
        return {"task_id": qid or f"{uid}:{task_type}:{len(metrics)}", "user_id": uid, "task_type": task_type, "status": status,
                "metrics": metrics, "accuracy_pct": acc, "duration_ms": 1, "memory": memory or {"status": "ok"}}

    def test_micro_is_row_weighted_over_scored_rows_and_retired_types_are_dropped(self):
        runs = [
            self.row("personal_qa_hallucination", {"abstention_quality_0_10": 8}, acc=80.0, qid="1:a"),
            self.row("personal_qa_hallucination", {"abstention_quality_0_10": 4}, acc=40.0, qid="1:b"),
            self.row("hidden_persona_implicit_qa", {"deep_motivation_alignment": 3}, acc=100.0, qid="1:c"),
            self.row("hidden_persona_implicit_qa", {}, acc=None, qid="1:d"),  # 主指标缺失:不计入
            self.row("daily_personalized_briefing", {"x": 1}, acc=0.0, qid="1:e"),  # 已弃用
        ]
        s = summarize(runs, failed=2)
        self.assertAlmostEqual(s["accuracy_pct_micro"], (80 + 40 + 100) / 3, places=1)
        self.assertEqual((s["n_queries"], s["n_scored"], s["n_unscored"], s["n_failed"]), (4, 3, 1, 2))
        self.assertIn("personal_qa_hallucination", s["by_task"])
        self.assertNotIn("daily_personalized_briefing", s["by_task"])

    def test_error_rows_count_as_zero_like_upstream(self):
        s = summarize([self.row("personal_qa_hallucination", {}, status="error", acc=0.0, qid="1:a"),
                       self.row("personal_qa_hallucination", {"abstention_quality_0_10": 10}, acc=100.0, qid="1:b")])
        self.assertAlmostEqual(s["accuracy_pct_micro"], 50.0)

    def test_memory_statuses_are_counted(self):
        s = summarize([self.row("personal_qa_hallucination", {"abstention_quality_0_10": 10}, acc=100.0, qid="1:a",
                                memory={"status": "empty"}),
                       self.row("personal_qa_hallucination", {"abstention_quality_0_10": 10}, acc=100.0, qid="1:b",
                                memory={"status": "ok", "truncated": True})])
        self.assertEqual(s["memory"], {"status_counts": {"empty": 1, "ok": 1}, "n_truncated": 1})


class LLMCallerIncompleteTests(unittest.TestCase):
    """``allow_incomplete``:空正文照常重试,重试用尽后返回而不是抛错(PersonaMem 按"空答案"评分)。"""

    def caller(self, replies, **kw):
        from IMAct.exec_layer.bridge import AsyncBridge
        from IMAct.harness.llm import LLMCaller
        from IMAct.models.base import ChatResponse
        from IMAct.models.config import ModelConfig
        from IMAct.retry import RetryPolicy

        bridge = AsyncBridge("t")
        self.addCleanup(bridge.close)
        llm = LLMCaller(ModelConfig(model="m", provider="openai_compatible", base_url="https://x"), bridge,
                        policy=RetryPolicy(max_retries=2, backoff_s=0.0))
        seq = iter(replies)

        async def chat(request):
            content, finish = next(seq)
            return ChatResponse(content=content, finish_reason=finish, model="m", usage={}, tool_calls=[], raw={})

        llm.provider = SimpleNamespace(chat=chat, aclose=None)
        return llm

    def test_default_still_raises_and_allow_incomplete_returns_the_last_reply(self):
        from IMAct.harness.messages import UserMessage
        from IMAct.retry import CallError

        msg = [UserMessage(role="user", content="hi")]
        with self.assertRaises(CallError):
            self.caller([("", "stop")] * 3).chat(msg)
        self.assertEqual(self.caller([("", "stop")] * 3).chat(msg, allow_incomplete=True).content, "")
        self.assertEqual(self.caller([("", "stop"), ("cut off", "length"), ("partial", "length")]).chat(msg, allow_incomplete=True).content, "partial")
        self.assertEqual(self.caller([("", "stop"), ("fine", "stop")]).chat(msg, allow_incomplete=True).content, "fine")  # 重试成功就用成功的


# ---------------------------------------------------------------------------------------------- 端到端
class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, dict(self.headers), body))
        if self.path.endswith("/chat/completions"):
            prompt = body["messages"][-1]["content"]
            content = F.fake_judge(prompt) if body["model"] == "fake-judge" else F.fake_answer(prompt)
            reply = {"id": "x", "model": body["model"], "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(content) // 4}}
        else:  # MPS 网关
            feats = body["tensorFeatures"]
            task = json.loads(feats["task"]["stringValues"][0])
            uid = feats["user_id"]["stringValues"][0]
            reply = {"success": True, "resultMap": {"context": f"## Who they are\n- e2e memory for {uid} at {task['timestamp']}", "status": "ok", "error": ""}}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@unittest.skipUnless(HAS_DATA, WHY)
class EndToEndTests(unittest.TestCase):
    """命令行全流程:本地假 MPS 网关 + 假 OpenAI 兼容模型服务。验证 Memory 的真实请求格式、结果落盘、汇总、断点续跑。"""

    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.server.requests = []
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def run_cli(self, out, *extra):
        from IMAct.benchmarks.personamem_v3.run import main

        models = Path(out).parent / "models.yaml"
        models.write_text(json.dumps({
            "roles": {"action": "fake-answer", "user": "fake-answer", "judge": "fake-judge"},
            "models": {name: {"provider": "openai_compatible", "model": name, "base_url": f"{self.base}/v1/chat/completions",
                              "params": {"stream": False}} for name in ("fake-answer", "fake-judge")},
        }), encoding="utf-8")
        with mock.patch.dict(os.environ, {"IMACT_MEMORY_URL": f"{self.base}/memory"}):
            return main(["--models", str(models), "--users", "25", "--num-queries", "8", "--max-concurrency", "3",
                         "--model-retries", "0", "--out", str(out), *extra])

    def test_a_full_run_writes_results_and_a_summary_with_the_official_headline(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            self.server.requests.clear()
            self.assertEqual(self.run_cli(out), 0)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual((summary["n_queries"], summary["n_failed"], summary["n_personas"]), (8, 0, 1))
            self.assertIsNotNone(summary["accuracy_pct_micro"])
            self.assertEqual(summary["memory"]["status_counts"], {"ok": 8})
            rows = [json.loads(p.read_text(encoding="utf-8")) for p in (out / "25").glob("*.trial0.json")]
            self.assertEqual(len(rows), 8)
            self.assertTrue(all("e2e memory for 25" in r["memory"]["context"] for r in rows))

            mem = [b for p, h, b in self.server.requests if p == "/memory"]
            self.assertEqual(len(mem), 8)
            feats = mem[0]["tensorFeatures"]
            self.assertEqual(feats["action"]["stringValues"], ["query"])
            self.assertEqual(feats["bench"]["stringValues"], ["personamem_v3"])
            self.assertEqual(feats["user_id"]["stringValues"], ["25"])
            task = json.loads(feats["task"]["stringValues"][0])
            self.assertEqual(set(task), {"query_id", "user_query", "timestamp"})
            first = D.load_test("25")[0]
            sent = {json.loads(b["tensorFeatures"]["task"]["stringValues"][0])["query_id"]: json.loads(b["tensorFeatures"]["task"]["stringValues"][0]) for b in mem}
            self.assertEqual(sent[first["query_id"]]["timestamp"], first["ts"])
            mps_headers = next(h for p, h, b in self.server.requests if p == "/memory")
            self.assertEqual(mps_headers.get("MPS-app-name"), "imact")
            self.assertTrue(mps_headers.get("MPS-trace-id"))

            # 断点续跑:同一个 --out 再跑一次,不再发任何请求,汇总不变
            self.server.requests.clear()
            self.assertEqual(self.run_cli(out), 0)
            self.assertEqual(self.server.requests, [])
            again = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(again["accuracy_pct_micro"], summary["accuracy_pct_micro"])

            # python -m IMAct.report:只读结果文件重算,与运行结束时的汇总一致
            from IMAct.report import load

            bench, rebuilt = load(out)
            self.assertEqual(bench, "personamem_v3")
            for key in ("accuracy_pct_micro", "accuracy_pct_micro_judge", "n_queries", "n_scored", "by_class", "by_task"):
                self.assertEqual(rebuilt[key], summary[key], key)

    def test_check_reports_an_unreachable_memory_gateway(self):
        from IMAct.benchmarks.personamem_v3.run import main

        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models.yaml"
            models.write_text(json.dumps({"roles": {"action": "a", "user": "a", "judge": "a"},
                                          "models": {"a": {"provider": "openai_compatible", "model": "a", "base_url": "http://127.0.0.1:9/x", "params": {"stream": False}}}}),
                              encoding="utf-8")
            code = main(["--models", str(models), "--users", "25", "--check", "--memory-url", "http://127.0.0.1:9/memory",
                         "--memory-timeout", "2", "--memory-retries", "0"])
        self.assertEqual(code, 2)


class CliTests(unittest.TestCase):
    def parse(self, *argv):
        from IMAct.benchmarks.personamem_v3.run import build_parser

        return build_parser().parse_args(["--models", "m.yaml", *argv])

    def test_defaults(self):
        a = self.parse()
        self.assertEqual((a.memory_timeout, a.memory_token_cap, a.memory_query_placeholder, a.max_concurrency, a.no_judge), (20.0, 4096, "empty", 4, False))

    @unittest.skipUnless(HAS_DATA, WHY)
    def test_task_type_filter_and_limits(self):
        from IMAct.benchmarks.personamem_v3.run import select_jobs, select_users

        jobs = select_jobs("25", self.parse("--task-types", "personalized_recommendation"))
        self.assertTrue(jobs and all(j.item["task_type"] == "personalized_recommendation" for j in jobs))
        self.assertEqual(len(select_jobs("25", self.parse("--num-queries", "3"))), 3)
        with self.assertRaisesRegex(ValueError, "未知题型"):
            select_jobs("25", self.parse("--task-types", "hotel_booking"))
        with self.assertRaisesRegex(ValueError, "没有数据"):
            select_users(self.parse("--users", "999999"))


if __name__ == "__main__":
    unittest.main()
