"""通用 MCP 客户端(harness/mcp.py)的测试:连本地起的真实 MCP server(SSE),不联网。"""

import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

from IMAct.exec_layer.bridge import AsyncBridge
from IMAct.harness.mcp import GatewayConfig, McpClient, McpConnection, McpError, call_once

SERVER = Path(__file__).with_name("mcp_test_server.py")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LocalServerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.proc = subprocess.Popen([sys.executable, "-X", "utf8", str(SERVER), str(cls.port)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", cls.port), timeout=0.5).close()
                break
            except OSError:
                if cls.proc.poll() is not None:
                    raise RuntimeError(f"测试 MCP server 启动失败: {cls.proc.stderr.read().decode(errors='replace')}")
                time.sleep(0.2)
        else:
            raise RuntimeError("测试 MCP server 启动超时")
        cls.url = f"http://127.0.0.1:{cls.port}/sse"
        cls.bridge = AsyncBridge("mcp-test")

    @classmethod
    def tearDownClass(cls):
        cls.bridge.close()
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.proc.stderr.close()


class McpClientTests(LocalServerCase):
    def test_lists_tools_with_name_description_and_schema(self):
        with McpClient("t", self.url, self.bridge) as client:
            tools = {t.name: t for t in client.list_tools()}
        self.assertEqual(set(tools), {"echo", "add", "boom", "sleepy"})
        self.assertEqual(tools["echo"].description, "Echo the text back.")
        self.assertEqual(tools["add"].input_schema["required"], ["a", "b"])

    def test_calls_a_tool_and_returns_the_text_block(self):
        with McpClient("t", self.url, self.bridge) as client:
            result = client.call_tool("echo", {"text": "你好"})
        self.assertFalse(result.is_error)
        self.assertEqual(result.first_text, "echo:你好")
        self.assertEqual(result.model_dump()["content"][0]["type"], "text")

    def test_a_tool_level_error_is_a_result_not_an_exception(self):
        with McpClient("t", self.url, self.bridge) as client:
            result = client.call_tool("boom", {})
        self.assertTrue(result.is_error)
        self.assertIn("kaboom", result.first_text)

    def test_the_connection_survives_many_calls_from_many_threads(self):
        from concurrent.futures import ThreadPoolExecutor

        with McpClient("t", self.url, self.bridge) as client:
            with ThreadPoolExecutor(8) as pool:
                outputs = list(pool.map(lambda i: client.call_tool("add", {"a": i, "b": 1}).first_text, range(24)))
        self.assertEqual(outputs, [str(i + 1) for i in range(24)])

    def test_a_slow_call_times_out_without_retrying(self):
        with McpClient("t", self.url, self.bridge, call_timeout=1.0) as client:
            started = time.time()
            with self.assertRaisesRegex(McpError, "Connection lost"):
                client.call_tool("sleepy", {"seconds": 5})
            self.assertLess(time.time() - started, 4)

    def test_close_is_idempotent_and_calls_after_close_fail_cleanly(self):
        client = McpClient("t", self.url, self.bridge).connect()
        client.close()
        client.close()
        with self.assertRaises(Exception):
            client.call_tool("echo", {"text": "x"})

    def test_connecting_to_a_dead_port_raises_mcp_error_after_retries(self):
        client = McpClient("t", f"http://127.0.0.1:{free_port()}/sse", self.bridge, connect_retries=2, connect_retry_delay=0.05, connect_timeout=2)
        with self.assertRaisesRegex(McpError, "连接 MCP server 失败"):
            client.connect()


class OneShotTests(LocalServerCase):
    def test_call_once_connects_calls_and_closes(self):
        import asyncio

        async def go():
            async with McpConnection(self.url) as conn:
                return (await conn.call_tool("add", {"a": 2, "b": 3})).first_text

        self.assertEqual(asyncio.run(go()), "5")


class GatewayTests(unittest.TestCase):
    def test_url_and_auth_header(self):
        gw = GatewayConfig.from_env({"MCP_GATEWAY_ADDRESS": "https://gw.example.com/", "MCP_GATEWAY_TOKEN": "tok"})
        self.assertEqual(gw.url("github"), "https://gw.example.com/github/sse")
        self.assertEqual(gw.headers, {"Authorization": "Bearer tok"})
        self.assertIsNone(GatewayConfig.from_env({"MCP_GATEWAY_ADDRESS": "http://x"}).headers)

    def test_missing_address_is_an_actionable_error(self):
        with self.assertRaisesRegex(McpError, "MCP_GATEWAY_ADDRESS"):
            GatewayConfig.from_env({})


if __name__ == "__main__":
    unittest.main()
