"""本地适配的线程、取消、生命周期与执行层协议回归;不请求模型。"""
import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from IMAct.exec_layer.config import OuterConfig, apply_memory_overrides, load_outer_config
from IMAct.exec_layer.memory_local import LocalMemoryClient, LocalMemoryConfig, normalize_local_response
from IMAct.exec_layer.memory_mps import MpsMemoryClient
from IMAct.trace import TraceRecorder, trace_scope
from tests.exec_helpers import DemoProvider, make_cfg
from IMAct.exec_layer.factory import build_runtime, _memory_router
from IMAct.exec_layer.outer import MemoryRequest


class SyncClient:
    def __init__(self, cfg):
        self.threads = [threading.get_ident()]
        self.requests = []
        self.closed = 0

    def query(self, **kwargs):
        self.threads.append(threading.get_ident())
        self.requests.append(kwargs)
        return {"status": "ok", "context": "用户不吃花生", "error": ""}

    def close(self):
        self.threads.append(threading.get_ident())
        self.closed += 1


REQUEST = dict(user_id="A891207", query_id="sub_A891207_1", user_query="午餐有什么忌口?", timestamp="2026-03-02 12:01:02")


class LocalMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_reuse_mapping_threads_and_trace(self):
        made = []

        def factory(cfg):
            client = SyncClient(cfg)
            made.append(client)
            return client

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            client = LocalMemoryClient(LocalMemoryConfig(), client_factory=factory, trace=TraceRecorder(path))
            try:
                await client.start()
                with trace_scope(task_id="t1", phase="prefetch"):
                    results = await asyncio.gather(*(client.query(**REQUEST) for _ in range(3)))
            finally:
                await client.aclose()
            await client.aclose()
            self.assertEqual(len(made), 1)
            sync = made[0]
            self.assertEqual(sync.closed, 1)
            self.assertEqual(len(set(sync.threads)), 1)
            self.assertNotIn(threading.get_ident(), sync.threads)
            self.assertTrue(all(r["status"] == "ok" for r in results))
            self.assertTrue(all(r["timestamp"] == REQUEST["timestamp"] and r["query"] == REQUEST["user_query"] for r in sync.requests))
            self.assertEqual(len({r["trace_id"] for r in sync.requests}), 3)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r["correlation"]["task_id"] == "t1" and r["data"]["transport"] == "local" for r in rows))
            self.assertEqual((await client.query(**REQUEST))["status"], "error")
            self.assertEqual(len(sync.requests), 3)

    async def test_cancelled_query_does_not_overlap_next_query_or_close(self):
        entered, release = threading.Event(), threading.Event()
        events = []

        class BlockingClient(SyncClient):
            def query(self, **kwargs):
                events.append("query_start")
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("test did not release worker")
                result = super().query(**kwargs)
                events.append("query_end")
                return result

            def close(self):
                events.append("close")
                super().close()

        client = LocalMemoryClient(LocalMemoryConfig(), client_factory=BlockingClient)
        first = asyncio.create_task(client.query(**REQUEST))
        try:
            # 同步 worker 阻塞时,主事件循环仍能完成这个 await。
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(client.query(**REQUEST))
            await asyncio.sleep(0)
            close = asyncio.create_task(client.aclose())
            await asyncio.sleep(0)
            self.assertFalse(close.done())
            self.assertEqual(events, ["query_start"])
            release.set()
            self.assertEqual((await second)["status"], "ok")
            await close
            self.assertEqual(events, ["query_start", "query_end", "query_start", "query_end", "close"])
        finally:
            release.set()
            await client.aclose()

    async def test_exception_is_error_and_not_retried(self):
        class FailedClient(SyncClient):
            def query(self, **kwargs):
                self.requests.append(kwargs)
                raise ValueError("snapshot mismatch")

        client = LocalMemoryClient(LocalMemoryConfig(), client_factory=FailedClient)
        try:
            result = await client.query(**REQUEST)
            self.assertEqual(result["status"], "error")
            self.assertIn("snapshot mismatch", result["error"])
            self.assertEqual(len(client._client.requests), 1)
        finally:
            await client.aclose()

    async def test_active_query_router_preserves_identity(self):
        client = LocalMemoryClient(LocalMemoryConfig(), client_factory=SyncClient)
        cfg = OuterConfig(memory={"backend": "local"})
        req = MemoryRequest("task1", "req1", "忌口?", {"user_id": "A891207", "query_id": "sub_A891207_1", "timestamp": "2026-03-02", "instruction": "帮我点午餐"})
        try:
            self.assertEqual((await _memory_router(client, cfg)(req))["status"], "ok")
            self.assertEqual(client._client.requests[-1]["query"], "忌口?")
            cfg.memory.active_query_text = "instruction"
            await _memory_router(client, cfg)(req)
            self.assertEqual(client._client.requests[-1]["query"], "帮我点午餐")
            self.assertEqual(client._client.requests[-1]["query_id"], "sub_A891207_1")
        finally:
            await client.aclose()


class LocalConfigTests(unittest.TestCase):
    def test_response_states(self):
        for payload in [None, {"status": "building", "context": "", "error": ""},
                        {"status": "error", "context": "", "error": ""},
                        {"status": "unknown", "context": "text"},
                        {"status": "ok", "context": "text", "error": "failure"},
                        {"status": "ok", "context": {}}, {"status": "empty", "context": "text"}]:
            with self.subTest(payload=payload):
                self.assertEqual(normalize_local_response(payload)["status"], "error")
        self.assertEqual(normalize_local_response({"status": "ok", "context": " "})["status"], "empty")

    def test_paths_relative_to_yaml_and_mode_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.yaml"
            path.write_text("memory:\n  backend: local\n  require_prefetch: true\n  local:\n    client_path: lib/query.py\n    config_path: lib/harness.json\n")
            cfg = load_outer_config(path)
            self.assertEqual(cfg.memory.local.config_path, str((Path(tmp) / "lib/harness.json").resolve()))
            apply_memory_overrides(cfg, no_memory=True)
            self.assertEqual(cfg.memory.backend, "none")
            self.assertFalse(cfg.memory.require_prefetch)
            apply_memory_overrides(cfg, memory_url="https://memory.example")
            self.assertEqual(cfg.memory.backend, "mps")
            apply_memory_overrides(cfg, local_memory=path)
            self.assertEqual(cfg.memory.backend, "local")
            self.assertEqual(cfg.memory.local.config_path, str(path.resolve()))

    def test_factory_backend_selection_and_close(self):
        for backend, expected in [("local", LocalMemoryClient), ("mps", MpsMemoryClient), ("none", None)]:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                cfg = make_cfg(tmp, memory={"backend": backend, "gateway": {"url": "https://memory.invalid"}})
                rt = build_runtime(cfg, provider=DemoProvider())
                try:
                    if expected:
                        self.assertIsInstance(rt.memory, expected)
                    else:
                        self.assertIsNone(rt.memory)
                finally:
                    rt.close()
                    rt.close()  # 重复关闭不创建无人等待的协程

    def test_none_backend_disables_prefetch_even_if_strict_flag_is_set(self):
        from tests.test_vitabench_bridge import make_agent, spec_for

        agent, rt = make_agent(memory=False)
        rt.cfg.memory.backend = "none"
        rt.cfg.memory.require_prefetch = True
        try:
            self.assertEqual(agent.begin_episode(spec_for()).memory, {"status": "not_requested"})
        finally:
            rt.close()

    def test_agent_prefetch_keeps_original_time_and_injects_memory(self):
        from tests.test_vitabench_bridge import make_agent, spec_for, drive

        agent, rt = make_agent(memory=False)
        local = LocalMemoryClient(LocalMemoryConfig(), client_factory=SyncClient)
        rt.memory = local
        rt.cfg.memory.backend = "local"
        rt.cfg.memory.timestamp_mode = "date"  # 本地模式仍必须保留原始值
        try:
            with mock.patch("IMAct.benchmarks.vitabench.personalization.current_time_map", return_value={"sub_A891207_1": REQUEST["timestamp"]}):
                ctx = agent.begin_episode(spec_for(memory_cfg=rt.cfg.memory))
            self.assertEqual(ctx.spec.timestamp, REQUEST["timestamp"])
            self.assertEqual(ctx.memory["context"], "用户不吃花生")
            drive(agent)
            inputs = [json.loads(r.messages[-1]["content"]) for r in rt.provider.requests]
            self.assertIn("用户不吃花生", json.dumps(inputs, ensure_ascii=False))
            with mock.patch("IMAct.benchmarks.vitabench.personalization.current_time_map", return_value={}):
                with self.assertRaisesRegex(ValueError, "current_time"):
                    spec_for(memory_cfg=rt.cfg.memory)
        finally:
            rt.close()
