import json
import threading
import unittest

import httpx

from IMAct.retry import CallError, RetryPolicy
from IMAct.exec_layer.bridge import AsyncBridge
from IMAct.exec_layer.config import OuterConfig
from IMAct.exec_layer.memory_mps import MpsMemoryClient, MpsMemoryConfig, build_body, parse_response
from IMAct.exec_layer.outer import ExecutionError, MemoryRequest
from IMAct.models.base import ChatResponse
from tests.exec_helpers import RECEIPTS, DemoProvider, make_runtime, no_sleep, run_demo_episode, start_event

from imact.errors import HarnessError  # noqa: E402  (exec_helpers 已保证 imact 可导入)


def mem_ok(req: MemoryRequest) -> dict:
    return {"status": "ok", "context": "演示偏好:辛香,优先30分钟内配送。"}


class OuterFlowTests(unittest.TestCase):
    def test_full_episode_through_our_provider(self):
        provider = DemoProvider()
        rt = make_runtime(provider)
        rt.outer.memory_router = _counting_router(mem_ok)
        try:
            actions = run_demo_episode(rt, "ep1")
        finally:
            rt.close()
        self.assertEqual([a.type for a in actions], ["call_tools", "reply_user", "call_tools", "call_tools", "call_tools", "reply_user"])
        # query_memory 在外层内部消化:对外看不到,但第一个动作是经过了一次 memory 往返得到的
        self.assertEqual(actions[0].memory_hops, 1)
        self.assertEqual(actions[0].calls[0]["name"], "delivery_store_search_recommand")
        self.assertEqual(actions[-1].mode, "final")
        self.assertEqual(actions[-1].outcome, "completed")
        # 同一个 advance 内的 usage 被汇总(start 事件触发 2 次模型调用:query_memory 一次、memory_result 一次)
        self.assertEqual(actions[0].usage, {"prompt_tokens": 20, "completion_tokens": 10, "model_calls": 2})
        # rev 单调递增,由执行层提交
        revs = [a.rev for a in actions]
        self.assertEqual(revs, sorted(revs))
        self.assertEqual(provider.requests[0].messages[-1]["role"], "user")

    def test_request_uses_executor_context_not_our_own(self):
        provider = DemoProvider()
        rt = make_runtime(provider)
        rt.outer.memory_router = _counting_router(mem_ok)
        try:
            run_demo_episode(rt, "ep2")
        finally:
            rt.close()
        ctx = json.loads(provider.requests[0].messages[-1]["content"])
        self.assertEqual(ctx["latest_event"]["type"], "start")
        self.assertIn("available_tools", ctx)
        self.assertEqual(provider.requests[0].messages[0]["role"], "system")  # 执行层自己的 prompt
        self.assertEqual(provider.requests[0].messages[1]["role"], "system")  # task_policy = start.system_prompt

    def test_replay_is_idempotent_and_does_not_requery_memory_or_model(self):
        provider = DemoProvider()
        rt = make_runtime(provider)
        router = _counting_router(mem_ok)
        rt.outer.memory_router = router
        try:
            first = rt.bridge.run(rt.outer.drive("ep3", 0, start_event()))
            calls_after_first = (len(provider.requests), router.calls)
            again = rt.bridge.run(rt.outer.drive("ep3", 0, start_event()))  # benchmark 用同一状态重放
        finally:
            rt.close()
        self.assertEqual((first.calls, first.rev), (again.calls, again.rev))
        self.assertEqual((len(provider.requests), router.calls), calls_after_first)
        self.assertEqual(router.calls, 1)

    def test_changed_memory_on_replay_does_not_conflict(self):
        # 回归点:第二次网关返回不同内容,若不缓存就会让 memory_result 与已提交事件冲突
        answers = iter(["第一次的记忆", "第二次不同的记忆"])

        async def flaky(req):
            return {"status": "ok", "context": next(answers)}

        rt = make_runtime(DemoProvider())
        rt.outer.memory_router = flaky
        try:
            a = rt.bridge.run(rt.outer.drive("ep4", 0, start_event()))
            b = rt.bridge.run(rt.outer.drive("ep4", 0, start_event()))
        finally:
            rt.close()
        self.assertEqual(a.calls, b.calls)

    def test_memory_error_is_passed_through_as_error(self):
        async def down(req):
            return {"status": "error", "error": "memory 网关unavailable错误"}

        provider = DemoProvider()
        rt = make_runtime(provider)
        rt.outer.memory_router = down
        try:
            act = rt.bridge.run(rt.outer.drive("ep5", 0, start_event()))
        finally:
            rt.close()
        self.assertEqual(act.type, "call_tools")  # 记忆故障不阻断任务
        ctx = json.loads(provider.requests[1].messages[-1]["content"])
        self.assertEqual(ctx["latest_event"]["status"], "error")

    def test_router_crash_becomes_memory_error_not_task_failure(self):
        async def boom(req):
            raise RuntimeError("bug in router")

        rt = make_runtime(DemoProvider())
        rt.outer.memory_router = boom
        try:
            act = rt.bridge.run(rt.outer.drive("ep6", 0, start_event()))
        finally:
            rt.close()
        self.assertEqual(act.type, "call_tools")

    def test_no_router_reports_memory_unavailable(self):
        provider = DemoProvider()
        rt = make_runtime(provider)  # 没有配置 memory
        try:
            rt.bridge.run(rt.outer.drive("ep7", 0, start_event()))
        finally:
            rt.close()
        ctx = json.loads(provider.requests[1].messages[-1]["content"])
        self.assertEqual(ctx["latest_event"]["status"], "error")

    def test_endless_memory_queries_end_in_a_blocked_reply_not_a_crash(self):
        class LoopModel(DemoProvider):
            async def chat(self, request):
                self.requests.append(request)
                d = {"task_changes": {}, "working_notes": [], "next_action": {"type": "query_memory", "question": "再问一次?"}}
                return ChatResponse(content=json.dumps(d), finish_reason="stop", usage={})

        provider = LoopModel()
        rt = make_runtime(provider, max_memory_hops=2)
        rt.outer.memory_router = _counting_router(mem_ok)
        try:
            act = rt.bridge.run(rt.outer.drive("ep8", 0, start_event()))
        except (ExecutionError, HarnessError):
            act = None  # 外层的 hops 上限先触发,同样是有界的
        finally:
            rt.close()
        self.assertTrue(act is None or (act.type == "reply_user" and act.outcome == "blocked"))
        self.assertLess(len(provider.requests), 30)  # 有界,没有空转


class RetryTests(unittest.TestCase):
    def test_transient_model_failure_is_retried_with_identical_request(self):
        provider = DemoProvider(faults=[CallError("server_error", "x", status=503), CallError("timeout", "slow")])
        rt = make_runtime(provider)
        rt.outer.memory_router = _counting_router(mem_ok)
        try:
            act = rt.bridge.run(rt.outer.drive("r1", 0, start_event()))
            attempts = rt.outer.harness.store.trace("r1")
        finally:
            rt.close()
        self.assertEqual(act.type, "call_tools")
        text = json.dumps(attempts, ensure_ascii=False)
        self.assertIn("model_http_error", text)
        self.assertIn("model_transport_error", text)

    def test_auth_error_is_not_retried(self):
        provider = DemoProvider(faults=[CallError("auth", "bad key", status=401)] * 5)
        rt = make_runtime(provider)
        try:
            with self.assertRaises(HarnessError) as cm:
                rt.bridge.run(rt.outer.drive("r2", 0, start_event()))
        finally:
            rt.close()
        self.assertEqual(cm.exception.code, "model_auth_error")
        self.assertEqual(len(provider.requests), 1)  # 配置错误立刻暴露,不烧重试额度

    def test_exhausted_retries_return_blocked_reply_instead_of_looping(self):
        provider = DemoProvider(faults=[CallError("server_error", "down", status=500)] * 50)
        rt = make_runtime(provider, retry={"model_call": {"max_retries": 2, "backoff_s": 0.0}})
        try:
            act = rt.bridge.run(rt.outer.drive("r3", 0, start_event()))
        finally:
            rt.close()
        self.assertEqual((act.type, act.outcome), ("reply_user", "blocked"))
        self.assertEqual(len(provider.requests), 3)  # 首次 + 2 次重试 = max_decision_attempts

    def test_retry_budget_is_customizable(self):
        for retries in (0, 1, 4):
            provider = DemoProvider(faults=[CallError("server_error", "down", status=500)] * 50)
            rt = make_runtime(provider, retry={"model_call": {"max_retries": retries, "backoff_s": 0.0}})
            try:
                rt.bridge.run(rt.outer.drive(f"r4-{retries}", 0, start_event()))
            finally:
                rt.close()
            self.assertEqual(len(provider.requests), retries + 1, msg=f"retries={retries}")

    def test_bad_model_output_is_retried(self):
        native_tool_call = ChatResponse(
            content="", tool_calls=[{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}], finish_reason="tool_calls"
        )
        truncated = ChatResponse(content='{"task_changes":', finish_reason="length")
        not_json = ChatResponse(content="好的,我来帮您", finish_reason="stop")
        provider = DemoProvider(faults=[native_tool_call, truncated, not_json])
        rt = make_runtime(provider, retry={"model_call": {"max_retries": 4, "backoff_s": 0.0}})
        rt.outer.memory_router = _counting_router(mem_ok)
        try:
            act = rt.bridge.run(rt.outer.drive("r5", 0, start_event()))
            text = json.dumps(rt.outer.harness.store.trace("r5"), ensure_ascii=False)
        finally:
            rt.close()
        self.assertEqual(act.type, "call_tools")
        for code in ("model_format", "model_incomplete", "invalid_decision"):
            self.assertIn(code, text)

    def test_repair_feedback_reaches_the_model_on_retry(self):
        provider = DemoProvider(faults=[ChatResponse(content="不是 JSON", finish_reason="stop")])
        rt = make_runtime(provider)
        rt.outer.memory_router = _counting_router(mem_ok)
        try:
            rt.bridge.run(rt.outer.drive("r6", 0, start_event()))
        finally:
            rt.close()
        second = json.loads(provider.requests[1].messages[-1]["content"])
        self.assertTrue(second["repair_feedback"])


class RealRunRegressionTests(unittest.TestCase):
    """真实 GPT-5.5 冒烟运行暴露的两个问题。"""

    def _ungrounded_order(self):
        d = {"task_changes": {}, "working_notes": [], "next_action": {"type": "call_tools", "calls": [
            {"name": "create_delivery_order", "arguments": {"user_id": "demo_user", "store_id": "NOT_A_REAL_STORE", "product_ids": ["P1"],
                                                            "product_cnts": [1], "address": "x", "dispatch_time": "2027-01-01 12:00:00", "note": ""}}]}}
        return ChatResponse(content=json.dumps(d, ensure_ascii=False), finish_reason="stop", usage={})

    def test_ungrounded_id_error_code_is_retryable_not_fatal(self):
        from IMAct.exec_layer.outer import NON_RETRYABLE_CODES

        for code in ("ungrounded_id", "invalid_decision", "ungrounded_note", "store_product_mismatch", "unverified_completion",
                     "model_format", "model_incomplete", "decision_timeout", "model_http_error", "model_transport_error"):
            self.assertNotIn(code, NON_RETRYABLE_CODES, msg=code)
        for code in ("event_payload_conflict", "revision_conflict", "event_rejected", "model_auth_error", "context_overflow"):
            self.assertIn(code, NON_RETRYABLE_CODES, msg=code)

    def test_ungrounded_decision_is_retried_end_to_end(self):
        """让执行层真的校验失败一次:模型编造 store_id,重投后给出合法决策,任务继续。"""
        provider = DemoProvider()
        rt = make_runtime(provider)
        rt.outer.memory_router = _counting_router(mem_ok)
        try:
            first = rt.bridge.run(rt.outer.drive("rr2", 0, start_event()))
            c = first.calls[0]
            provider.faults = [self._ungrounded_order()]
            act = rt.bridge.run(rt.outer.drive("rr2", first.rev, {"type": "tool_results", "results": [
                {"call_id": c["call_id"], "name": c["name"], "delivery": "received", "content": RECEIPTS[c["name"]]}]}))
            trace = json.dumps(rt.outer.harness.store.trace("rr2"), ensure_ascii=False)
        finally:
            rt.close()
        self.assertIn("ungrounded_id", trace)  # 确实发生过一次证据校验失败
        self.assertIn(act.type, ("call_tools", "reply_user"))  # 重试后任务继续,而不是抛异常
        self.assertGreaterEqual(len(provider.requests), 4)

    def test_terminate_after_failed_event_does_not_conflict(self):
        # 不可重试的错误(如鉴权失败)会留下"失败未提交"的事件 e{n};随后的 terminate 以前复用同一个 id,触发 event_payload_conflict
        provider = DemoProvider()
        rt = make_runtime(provider)
        rt.outer.memory_router = _counting_router(mem_ok)
        try:
            first = rt.bridge.run(rt.outer.drive("rr3", 0, start_event()))
            c = first.calls[0]
            provider.faults = [CallError("auth", "bad key", status=401)]
            result_event = {"type": "tool_results", "results": [{"call_id": c["call_id"], "name": c["name"], "delivery": "received", "content": RECEIPTS[c["name"]]}]}
            with self.assertRaises(HarnessError) as cm:
                rt.bridge.run(rt.outer.drive("rr3", first.rev, result_event))
            self.assertEqual(cm.exception.code, "model_auth_error")
            out = rt.bridge.run(rt.outer.advance("rr3", first.rev, {"type": "terminate", "reason": "infrastructure_error"}))
        finally:
            rt.close()
        self.assertEqual(out.action.model_dump()["type"], "reply_user")

    def test_event_ids(self):
        from IMAct.exec_layer.outer import event_id_for

        self.assertEqual(event_id_for(2), "e3")
        self.assertEqual(event_id_for(2, "terminate"), "term3")
        self.assertNotEqual(event_id_for(2, "tool_results"), event_id_for(2, "terminate"))


class MpsMemoryTests(unittest.IsolatedAsyncioTestCase):
    CFG = MpsMemoryConfig(url="https://gw.example/inference/x/V0.1", app_name="imact-test")

    def client(self, handler, **kw):
        return MpsMemoryClient(
            self.CFG, retry=kw.pop("retry", RetryPolicy(max_retries=1, backoff_s=0.0)), client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            sleep=no_sleep, **kw,
        )

    async def test_request_shape_matches_the_documented_gateway_contract(self):
        seen = {}

        def handler(req):
            seen["headers"], seen["body"], seen["url"] = req.headers, json.loads(req.content), str(req.url)
            return httpx.Response(200, json={"context": "近期偏好喜来登"})

        r = await self.client(handler).query(user_id="A891207", query_id="sub_A891207_1", user_query="帮我点外卖", timestamp="2026-03-02")
        self.assertEqual(r, {"status": "ok", "context": "近期偏好喜来登", "error": None})
        h, b = seen["headers"], seen["body"]
        self.assertEqual((h["mps-app-name"], h["mps-http-version"]), ("imact-test", "1.0"))
        self.assertTrue(h["mps-trace-id"])
        self.assertEqual(b["features"], {})
        tf = b["tensorFeatures"]
        self.assertEqual(tf["action"], {"shapes": [1], "stringValues": ["query"]})
        self.assertEqual(tf["bench"]["stringValues"], ["vitabench"])
        self.assertEqual(tf["user_id"]["stringValues"], ["A891207"])
        task = json.loads(tf["task"]["stringValues"][0])  # task 是 JSON 字符串
        self.assertEqual(task, {"query_id": "sub_A891207_1", "user_query": "帮我点外卖", "timestamp": "2026-03-02"})

    async def test_response_variants(self):
        wrapped = [
            {"context": "x"},
            {"result": {"context": "x"}},
            {"outputs": {"tensorFeatures": {"r": {"stringValues": [json.dumps({"context": "x"})]}}}},
            {"data": json.dumps({"context": "x"})},
        ]
        for payload in wrapped:
            self.assertEqual(parse_response(payload)["context"], "x", msg=str(payload))
        self.assertEqual(parse_response({"context": ""})["status"], "empty")
        self.assertEqual(parse_response({"context": "   "})["status"], "empty")
        self.assertEqual(parse_response({"error": "快照未构建"}), {"status": "error", "context": "", "error": "快照未构建"})
        self.assertEqual(parse_response({"error": "", "context": "x"})["status"], "ok")
        self.assertEqual(parse_response({"foo": 1})["status"], "error")  # 看不懂的响应不能当 empty
        self.assertEqual(parse_response("纯文本")["status"], "error")

    async def test_real_gateway_envelope_and_timeout_retry(self):
        ok = {"costTime": 0, "errorMessage": "", "resultMap": {"context": "辛香、肉食、快送", "error": "", "status": "ok"}, "success": True}
        timeout = {"errorMessage": "ERROR_REQUEST_TIMEOUT;totalCost(ms):8000", "resultCode": "ERROR_REQUEST_TIMEOUT", "resultMap": {}, "success": False}
        replies = [timeout, ok]  # 第一次网关超时,重试后成功:与真实网关的表现一致
        r = await self.client(lambda req: httpx.Response(200, json=replies.pop(0))).query(user_id="u", query_id="q", user_query="x", timestamp=None)
        self.assertEqual((r["status"], r["context"]), ("ok", "辛香、肉食、快送"))
        self.assertEqual(replies, [])

        always = await self.client(lambda req: httpx.Response(200, json=timeout)).query(user_id="u", query_id="q", user_query="x", timestamp=None)
        self.assertEqual((always["status"], always["context"]), ("error", ""))
        self.assertIn("TIMEOUT", always["error"])

    async def test_logs_curl_and_returned_context(self):
        ok = {"resultMap": {"context": "第一行\n第二行", "status": "ok", "error": ""}, "success": True}
        client = MpsMemoryClient(
            MpsMemoryConfig(url="https://gw.example/x", api_key_env="IMACT_TEST_MEM_KEY"), api_key="SECRET",
            client=httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=ok))), sleep=no_sleep,
        )
        with self.assertLogs("imact.memory", level="INFO") as logs:
            await client.query(user_id="A1", query_id="sub_A1_1", user_query="点外卖", timestamp="2026-03-02")
        text = "\n".join(logs.output)
        self.assertIn("curl -X POST", text)
        self.assertIn("https://gw.example/x", text)
        self.assertIn("sub_A1_1", text)
        self.assertIn("点外卖", text)
        self.assertNotIn("SECRET", text)  # 鉴权头必须打码
        self.assertIn("第一行\n第二行", text)  # context 按原样(真实换行)记录

    async def test_failures_are_errors_never_empty(self):
        cases = [httpx.Response(500, text="boom"), httpx.Response(401), httpx.Response(404), httpx.Response(200, text="<html>")]
        for response in cases:
            r = await self.client(lambda req, resp=response: resp).query(user_id="u", query_id="q", user_query="x", timestamp=None)
            self.assertEqual(r["status"], "error", msg=str(response.status_code))
            self.assertEqual(r["context"], "")

        def refuse(req):
            raise httpx.ConnectError("refused")

        self.assertEqual((await self.client(refuse).query(user_id="u", query_id="q", user_query="x", timestamp=None))["status"], "error")

    async def test_5xx_is_retried_but_401_is_not(self):
        calls = []

        def handler(req):
            calls.append(1)
            return httpx.Response(503 if len(calls) == 1 else 200, json={"context": "ok"})

        self.assertEqual((await self.client(handler).query(user_id="u", query_id="q", user_query="x", timestamp=None))["status"], "ok")
        self.assertEqual(len(calls), 2)
        calls.clear()
        await self.client(lambda req: (calls.append(1), httpx.Response(401))[1]).query(user_id="u", query_id="q", user_query="x", timestamp=None)
        self.assertEqual(len(calls), 1)

    async def test_timestamp_precision_is_preserved(self):
        # 文档:时间保留原始精度,只有日期时不补造时分秒;PersonaMem-v2 为 null
        for ts in ("2026-03-02", 1775525523, None):
            body = build_body("query", "vitabench", "u", task={"query_id": "q", "user_query": "x", "timestamp": ts})
            self.assertEqual(json.loads(body["tensorFeatures"]["task"]["stringValues"][0])["timestamp"], ts)

    async def test_api_key_header_only_when_configured(self):
        seen = {}
        cfg = MpsMemoryConfig(url="https://gw/x", api_key_env="NOPE_KEY_ENV")
        with self.assertRaises(ValueError):
            MpsMemoryClient(cfg)
        c = MpsMemoryClient(cfg, api_key="SECRET", client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: (seen.update(h=r.headers), httpx.Response(200, json={"context": "x"}))[1])))
        await c.query(user_id="u", query_id="q", user_query="x", timestamp=None)
        self.assertEqual(seen["h"]["authorization"], "Bearer SECRET")

    async def test_build_requires_complete(self):
        ok = await self.client(lambda r: httpx.Response(200, json={"status": "complete"})).build(user_id="u", data={"subtasks": []})
        self.assertEqual(ok, {"status": "complete"})
        bad = await self.client(lambda r: httpx.Response(200, json={"error": "历史不全"})).build(user_id="u", data={})
        self.assertEqual(bad["status"], "error")


class BridgeTests(unittest.TestCase):
    def test_concurrent_callers_share_one_loop(self):
        b = AsyncBridge()
        out, threads = [], []

        async def work(i):
            import asyncio

            await asyncio.sleep(0.01)
            return i * 2

        def caller(i):
            out.append(b.run(work(i)))

        for i in range(8):
            t = threading.Thread(target=caller, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        b.close()
        self.assertEqual(sorted(out), [i * 2 for i in range(8)])

    def test_reentrant_call_from_loop_thread_is_rejected(self):
        b = AsyncBridge()

        async def bad():
            coro = asyncio.sleep(0)
            try:
                b.run(coro)
            finally:
                coro.close()

        import asyncio

        with self.assertRaises(RuntimeError):
            b.run(bad())
        b.close()

    def test_exceptions_propagate(self):
        b = AsyncBridge()

        async def boom():
            raise ValueError("x")

        with self.assertRaises(ValueError):
            b.run(boom())
        b.close()


class NameCollisionTests(unittest.TestCase):
    def test_imported_imact_is_the_executor_not_our_package(self):
        from pathlib import Path

        import imact

        from IMAct.exec_layer.loader import DEFAULT_SRC, _check_not_ours

        _check_not_ours(imact, DEFAULT_SRC)  # 正常情况下不抛
        with self.assertRaises(ImportError):
            import IMAct

            _check_not_ours(IMAct, DEFAULT_SRC)  # 如果解析到的是我们自己的包,必须报错


class ConfigTests(unittest.TestCase):
    def test_strict_keys(self):
        with self.assertRaises(Exception):
            OuterConfig.model_validate({"bogus": 1})
        with self.assertRaises(Exception):
            OuterConfig.model_validate({"memory": {"gateway": {"urll": "x"}}})

    def test_decision_timeout_must_exceed_model_timeout(self):
        from IMAct.exec_layer import build_runtime

        cfg = OuterConfig.model_validate({"model": {"timeout_s": 100}, "exec": {"decision_timeout_s": 50}})
        with self.assertRaises(ValueError):
            build_runtime(cfg, provider=DemoProvider())


class _counting_router:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    async def __call__(self, req):
        self.calls += 1
        return self.fn(req)


if __name__ == "__main__":
    unittest.main()
