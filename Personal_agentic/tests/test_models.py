import json
import unittest

import httpx

from IMAct.retry import CallError, RetryPolicy, run_with_retry
from IMAct.models.anthropic_messages import AnthropicMessagesProvider
from IMAct.models.azure_openai import AzureOpenAIProvider
from IMAct.models.base import ChatRequest, ChatResponse
from IMAct.models.config import ConfigError, ModelConfig, create_provider
from IMAct.models.mock import ScriptedProvider, text_response
from IMAct.models.openai_compat import OpenAICompatibleProvider, normalize_messages
from IMAct.models.openai_responses import OpenAIResponsesProvider
from IMAct.models.openrouter import OpenRouterProvider


def client_for(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


OK_BODY = {
    "id": "x",
    "model": "m1",
    "choices": [
        {
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning_content": "想一想",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}],
            },
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
}


class OpenAICompatTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_and_response(self):
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["url"] = str(req.url)
            seen["auth"] = req.headers.get("authorization")
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json=OK_BODY)

        p = OpenAICompatibleProvider(base_url="http://x/v1/", model="m1", api_key="K", extra_body={"top_k": 5}, client=client_for(handler))
        resp = await p.chat(
            ChatRequest(
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": {"a": 1}}}]},
                    {"role": "tool", "tool_call_id": "c1", "name": "f", "content": "ok"},
                ],
                tools=[{"type": "function", "function": {"name": "f"}}],
                params={"temperature": 0.2},
            )
        )
        self.assertEqual(seen["url"], "http://x/v1/chat/completions")
        self.assertEqual(seen["auth"], "Bearer K")
        b = seen["body"]
        self.assertEqual((b["model"], b["temperature"], b["top_k"]), ("m1", 0.2, 5))
        self.assertEqual(b["tools"][0]["function"]["name"], "f")
        # arguments 对象被编码成 JSON 字符串;content 为 None 的字段被省略
        self.assertEqual(b["messages"][1]["tool_calls"][0]["function"]["arguments"], "{\"a\": 1}")
        self.assertNotIn("content", b["messages"][1])
        self.assertEqual(resp.tool_calls[0]["function"]["name"], "f")
        self.assertEqual((resp.usage["prompt_tokens"], resp.reasoning, resp.finish_reason, resp.model), (11, "想一想", "tool_calls", "m1"))

    async def test_content_parts_and_no_key(self):
        def handler(req):
            self.assertNotIn("authorization", req.headers)
            return httpx.Response(200, json={"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]})

        p = OpenAICompatibleProvider(base_url="http://x", model="m", client=client_for(handler))
        self.assertEqual((await p.chat(ChatRequest(messages=[{"role": "user", "content": "x"}]))).content, "ab")

    async def test_error_mapping(self):
        cases = [
            (httpx.Response(429, headers={"retry-after": "3"}, text="slow"), "rate_limit"),
            (httpx.Response(500, text="boom"), "server_error"),
            (httpx.Response(503, text="boom"), "server_error"),
            (httpx.Response(401, text="no"), "auth"),
            (httpx.Response(403, text="no"), "auth"),
            (httpx.Response(400, text="bad"), "client"),
            (httpx.Response(200, text="not json"), "bad_response"),
            (httpx.Response(200, json={"error": {"code": 502, "message": "upstream"}}), "server_error"),
            (httpx.Response(200, json={"error": {"code": 429, "message": "slow"}}), "rate_limit"),
            (httpx.Response(200, json={"foo": 1}), "bad_response"),
        ]
        for response, category in cases:
            p = OpenAICompatibleProvider(base_url="http://x", model="m", client=client_for(lambda req, r=response: r))
            with self.assertRaises(CallError, msg=str(response.status_code)) as cm:
                await p.chat(ChatRequest(messages=[{"role": "user", "content": "x"}]))
            self.assertEqual(cm.exception.category, category, msg=f"{response.status_code} {response.text[:20]}")

    async def test_retry_after_is_carried(self):
        p = OpenAICompatibleProvider(base_url="http://x", model="m", client=client_for(lambda r: httpx.Response(429, headers={"retry-after": "3"})))
        with self.assertRaises(CallError) as cm:
            await p.chat(ChatRequest(messages=[{"role": "user", "content": "x"}]))
        self.assertEqual(cm.exception.retry_after, 3.0)

    async def test_transport_and_timeout(self):
        def boom(req):
            raise httpx.ConnectError("refused")

        def slow(req):
            raise httpx.ReadTimeout("slow")

        for h, cat in ((boom, "transport"), (slow, "timeout")):
            p = OpenAICompatibleProvider(base_url="http://x", model="m", client=client_for(h))
            with self.assertRaises(CallError) as cm:
                await p.chat(ChatRequest(messages=[{"role": "user", "content": "x"}]))
            self.assertEqual(cm.exception.category, cat)


class OpenRouterAzureTests(unittest.IsolatedAsyncioTestCase):
    async def test_openrouter_headers_and_body(self):
        seen = {}

        def handler(req):
            seen["url"], seen["h"], seen["b"] = str(req.url), req.headers, json.loads(req.content)
            return httpx.Response(200, json=OK_BODY)

        p = OpenRouterProvider(
            model="anthropic/claude-x", api_key="K", http_referer="https://app", app_title="IMAct",
            provider_routing={"order": ["Anthropic"]}, reasoning={"effort": "high"}, fallback_models=["openai/gpt-x"],
            client=client_for(handler),
        )
        await p.chat(ChatRequest(messages=[{"role": "user", "content": "x"}]))
        self.assertEqual(seen["url"], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(seen["h"]["authorization"], "Bearer K")
        self.assertEqual((seen["h"]["http-referer"], seen["h"]["x-title"]), ("https://app", "IMAct"))
        b = seen["b"]
        self.assertEqual(b["model"], "anthropic/claude-x")
        self.assertEqual(b["provider"], {"order": ["Anthropic"]})
        self.assertEqual(b["reasoning"], {"effort": "high"})
        self.assertEqual(b["models"], ["anthropic/claude-x", "openai/gpt-x"])

    async def test_azure_url_header_and_body(self):
        seen = {}

        def handler(req):
            seen["url"], seen["h"], seen["b"] = str(req.url), req.headers, json.loads(req.content)
            return httpx.Response(200, json=OK_BODY)

        p = AzureOpenAIProvider(endpoint="https://res.openai.azure.com", deployment="gpt4o", api_key="K", client=client_for(handler))
        await p.chat(ChatRequest(messages=[{"role": "user", "content": "x"}]))
        self.assertEqual(seen["url"], "https://res.openai.azure.com/openai/deployments/gpt4o/chat/completions?api-version=2024-10-21")
        self.assertEqual(seen["h"]["api-key"], "K")
        self.assertNotIn("authorization", seen["h"])
        self.assertNotIn("model", seen["b"])


class MatrixLLMTests(unittest.IsolatedAsyncioTestCase):
    CLAUDE_SSE = (
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_9","model":"claude-opus-5","usage":{"input_tokens":13,"output_tokens":1}}}\n\n'
        'event: ping\ndata: {"type":"ping"}\n\n'
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"第一点。"}}\n\n'
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"第二点。"}}\n\n'
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":8}}\n\n'
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )

    async def test_anthropic_messages_streams_by_default(self):
        seen = {}

        def handler(req):
            seen["body"] = json.loads(req.content)
            seen["headers"] = req.headers
            return httpx.Response(200, text=self.CLAUDE_SSE, headers={"content-type": "text/event-stream"})

        provider = AnthropicMessagesProvider(
            base_url="https://matrixllm.example/anthropic/v1", model="claude-opus-5",
            api_key="test-key", x_ly_token="claude-token", client=client_for(handler),
        )
        response = await provider.chat(ChatRequest(messages=[{"role": "user", "content": "给建议"}], params={"max_tokens": 512}))
        self.assertIs(seen["body"]["stream"], True)
        self.assertEqual(seen["headers"]["x-ly-token"], "claude-token")
        self.assertEqual(response.content, "第一点。第二点。")
        self.assertEqual((response.finish_reason, response.model, response.raw["id"]), ("stop", "claude-opus-5", "msg_9"))
        self.assertEqual(response.usage, {"prompt_tokens": 13, "completion_tokens": 8})

    async def test_anthropic_messages_stream_errors(self):
        provider = lambda handler: AnthropicMessagesProvider(
            base_url="https://matrixllm.example/anthropic/v1", model="m", api_key="k", client=client_for(handler))
        request = ChatRequest(messages=[{"role": "user", "content": "x"}])

        truncated = self.CLAUDE_SSE.split("event: message_stop")[0]
        r = await provider(lambda req: httpx.Response(200, text=truncated)).chat(request)
        self.assertEqual(r.finish_reason, "incomplete")

        overloaded = 'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"busy"}}\n\n'
        with self.assertRaises(CallError) as cm:
            await provider(lambda req: httpx.Response(200, text=overloaded)).chat(request)
        self.assertEqual(cm.exception.category, "server_error")

        with self.assertRaises(CallError) as cm:
            await provider(lambda req: httpx.Response(403, text="denied")).chat(request)
        self.assertEqual(cm.exception.category, "auth")

    async def test_anthropic_messages_request_and_response(self):
        seen = {}

        def handler(req):
            seen["url"] = str(req.url)
            seen["headers"] = req.headers
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json={
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "第一点。"}, {"type": "text", "text": "第二点。"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 13, "output_tokens": 8},
            })

        provider = AnthropicMessagesProvider(
            base_url="https://matrixllm.example/anthropic/v1", model="claude-opus-4-6",
            api_key="test-key", x_ly_token="claude-token", client=client_for(handler),
        )
        response = await provider.chat(ChatRequest(
            messages=[{"role": "system", "content": "系统规则"}, {"role": "user", "content": "给建议"}],
            params={"max_tokens": 512, "stream": False},
        ))
        self.assertEqual(seen["url"], "https://matrixllm.example/anthropic/v1/messages")
        self.assertEqual(seen["headers"]["authorization"], "Bearer test-key")
        self.assertEqual(seen["headers"]["x-ly-token"], "claude-token")
        self.assertEqual(seen["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(seen["body"]["system"], "系统规则")
        self.assertEqual(seen["body"]["messages"], [{"role": "user", "content": "给建议"}])
        self.assertEqual(response.content, "第一点。第二点。")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage, {"prompt_tokens": 13, "completion_tokens": 8})

    async def test_openai_responses_stream_request_and_response(self):
        seen = {}

        def handler(req):
            seen["url"] = str(req.url)
            seen["headers"] = req.headers
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, text=(
                'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"{\\"ok\\":"}\n\n'
                'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"true}"}\n\n'
                'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","model":"gpt-5.6-sol","usage":{"input_tokens":9,"output_tokens":4}}}\n\n'
            ), headers={"content-type": "text/event-stream"})

        provider = OpenAIResponsesProvider(
            base_url="https://matrixllm.example/v1", model="gpt-5.6-sol",
            api_key="test-key", x_ly_token="gpt-token", client=client_for(handler),
        )
        response = await provider.chat(ChatRequest(
            messages=[{"role": "user", "content": "总结故障复盘"}],
            params={"stream": True, "max_output_tokens": 512},
        ))
        self.assertEqual(seen["url"], "https://matrixllm.example/v1/responses")
        self.assertEqual(seen["headers"]["authorization"], "Bearer test-key")
        self.assertEqual(seen["headers"]["x-ly-token"], "gpt-token")
        self.assertEqual(seen["body"]["input"], [{"role": "user", "content": "总结故障复盘"}])
        self.assertEqual(seen["body"]["max_output_tokens"], 512)
        self.assertEqual(response.content, '{"ok":true}')
        self.assertEqual(response.usage, {"prompt_tokens": 9, "completion_tokens": 4})

    async def test_openai_compatible_supports_x_ly_token_env(self):
        import os

        seen = {}

        def handler(req):
            seen["url"] = str(req.url)
            seen["headers"] = req.headers
            return httpx.Response(200, json=OK_BODY)

        os.environ["IMACT_TEST_GEMINI_TOKEN"] = "gemini-token"
        try:
            provider = create_provider(ModelConfig(
                provider="openai_compatible", model="gemini-3.5-flash",
                base_url="https://matrixllm.example/v1", api_key="test-key",
                options={"x_ly_token_env": "IMACT_TEST_GEMINI_TOKEN"},
            ), client=client_for(handler))
            await provider.chat(ChatRequest(messages=[{"role": "user", "content": "三点建议"}]))
        finally:
            del os.environ["IMACT_TEST_GEMINI_TOKEN"]
        self.assertEqual(seen["url"], "https://matrixllm.example/v1/chat/completions")
        self.assertEqual(seen["headers"]["authorization"], "Bearer test-key")
        self.assertEqual(seen["headers"]["x-ly-token"], "gemini-token")


class RegistryTests(unittest.TestCase):
    def test_create_each_provider(self):
        self.assertIsInstance(create_provider(ModelConfig(provider="openai_compatible", model="m", base_url="http://x")), OpenAICompatibleProvider)
        self.assertIsInstance(create_provider(ModelConfig(provider="openai", model="m", api_key="K")), OpenAICompatibleProvider)
        r = create_provider(ModelConfig(provider="openrouter", model="m", api_key="K", options={"app_title": "t"}))
        self.assertIsInstance(r, OpenRouterProvider)
        self.assertIsInstance(
            create_provider(ModelConfig(provider="azure_openai", model="dep", api_key="K", base_url="https://r.openai.azure.com", options={"api_version": "v"})),
            AzureOpenAIProvider,
        )
        self.assertIsInstance(create_provider(ModelConfig(provider="anthropic_messages", model="m", api_key="K", base_url="https://matrixllm.example/anthropic/v1")), AnthropicMessagesProvider)
        self.assertIsInstance(create_provider(ModelConfig(provider="openai_responses", model="m", api_key="K", base_url="https://matrixllm.example/v1")), OpenAIResponsesProvider)

    def test_api_key_from_env(self):
        import os

        os.environ["IMACT_TEST_KEY"] = "secret"
        try:
            p = create_provider(ModelConfig(provider="openai", model="m", api_key_env="IMACT_TEST_KEY"))
            self.assertEqual(p._api_key, "secret")
        finally:
            del os.environ["IMACT_TEST_KEY"]

    def test_config_errors(self):
        with self.assertRaises(ConfigError):
            create_provider(ModelConfig(provider="nope", model="m"))
        with self.assertRaises(ConfigError):
            create_provider(ModelConfig(provider="openai", model="m", api_key_env="IMACT_DEFINITELY_UNSET"))
        with self.assertRaises(ConfigError):
            create_provider(ModelConfig(provider="openai_compatible", model="m"))  # 缺 base_url
        with self.assertRaises(ConfigError):
            create_provider(ModelConfig(
                provider="openai_compatible", model="m", base_url="http://x",
                api_key_env="IMACT_DEFINITELY_UNSET",
            ))
        with self.assertRaises(ConfigError):
            create_provider(ModelConfig(provider="openrouter", model="m", api_key="K", options={"bogus": 1}))

    def test_normalize_messages_drops_unknown_fields(self):
        out = normalize_messages([{"role": "user", "content": "x", "internal": 1}])
        self.assertEqual(out, [{"role": "user", "content": "x"}])


class RetryTests(unittest.IsolatedAsyncioTestCase):
    def test_policy_decisions(self):
        p = RetryPolicy(max_retries=3, backoff_s=2.0, backoff_factor=2.0, max_backoff_s=5.0)
        e = CallError("server_error", "x")
        self.assertEqual([p.should_retry(e, i) for i in range(4)], [2.0, 4.0, 5.0, None])
        self.assertIsNone(p.should_retry(CallError("auth", "x"), 0))
        self.assertEqual(p.should_retry(CallError("rate_limit", "x", retry_after=4.5), 0), 4.5)

    async def test_run_with_retry_sequence(self):
        delays, seen = [], []
        results = [CallError("timeout", "t"), CallError("rate_limit", "r"), "ok"]

        async def fn():
            r = results.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        async def fake_sleep(d):
            delays.append(d)

        out = await run_with_retry(fn, RetryPolicy(max_retries=3), sleep=fake_sleep, on_retry=lambda *a: seen.append(a[:2]))
        self.assertEqual((out, delays, seen), ("ok", [2.0, 4.0], [("model_call", 1), ("model_call", 2)]))

    async def test_gives_up_after_max_retries(self):
        calls = []

        async def fn():
            calls.append(1)
            raise CallError("server_error", "x")

        async def fake_sleep(d):
            pass

        with self.assertRaises(CallError):
            await run_with_retry(fn, RetryPolicy(max_retries=3), sleep=fake_sleep)
        self.assertEqual(len(calls), 4)  # 首次 + 3 次重试


if __name__ == "__main__":
    unittest.main()
