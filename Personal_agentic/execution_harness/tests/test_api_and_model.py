import json

import httpx2 as httpx
import pytest
from conftest import search, start
from fastapi.testclient import TestClient

from imact.adapters import to_chat_message
from imact.api import create_app
from imact.errors import HarnessError
from imact.model import ChatModel

pytestmark = pytest.mark.anyio


def test_http_boundary_auth_schema_and_duplicate(make_harness):
    harness, model, store = make_harness(search())
    with TestClient(create_app(harness, token="test-secret")) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/v1/execution/advance", json=start()).status_code == 401
        headers = {"Authorization": "Bearer test-secret"}
        r = client.post("/v1/execution/advance", json=start(), headers=headers)
        assert r.status_code == 200, r.text
        assert (
            client.post("/v1/execution/advance", json=start(), headers=headers).json() == r.json()
        )
        message = to_chat_message(r.json())
        assert message["tool_calls"][0]["id"] == r.json()["action"]["calls"][0]["call_id"]
        assert json.loads(message["tool_calls"][0]["function"]["arguments"])["keywords"] == ["辣味"]
        assert client.get("/v1/execution/tasks/test", headers=headers).json()["rev"] == 1
        assert client.get("/v1/execution/tasks/test/trace", headers=headers).status_code == 200
        assert client.get("/openapi.json").json()["paths"]["/v1/execution/advance"]
        bad = start()
        bad["event"]["unexpected"] = True
        assert client.post("/v1/execution/advance", json=bad, headers=headers).status_code == 422
        assert len(model.contexts) == 1


async def test_chat_provider_sends_one_joint_request_and_complete_schemas():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(search())},
                    }
                ]
            },
        )

    model = ChatModel("https://model.example/v1", "test", transport=httpx.MockTransport(handler))
    context = {
        "system": "joint instructions",
        "task_policy": "task policy",
        "input": {"available_tools": start()["event"]["tools"], "latest_event": {"type": "start"}},
    }
    response = await model.decide(context)
    assert json.loads(response.content) == search()
    assert len(requests) == 1
    assert "tools" not in requests[0]
    assert len(json.loads(requests[0]["messages"][-1]["content"])["available_tools"]) == 21


async def test_chat_provider_does_not_retry_or_leak_upstream_body():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, text="sensitive upstream debug")

    model = ChatModel("https://model.example/v1", "test", transport=httpx.MockTransport(handler))
    with pytest.raises(HarnessError) as caught:
        await model.decide({"system": "x", "task_policy": "", "input": {}})
    assert caught.value.code == "model_http_error"
    assert "sensitive" not in caught.value.message
    assert len(calls) == 1


async def test_truncated_model_response_saved_in_attempt(make_harness):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "real-response-name",
                "id": "req-truncated",
                "usage": {"total_tokens": 77},
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": '{"task_changes":'},
                    }
                ],
            },
        )

    h, _, store = make_harness()
    h.model = ChatModel("https://model.example/v1", "test", transport=httpx.MockTransport(handler))
    with pytest.raises(HarnessError) as caught:
        await h.advance(start())
    assert caught.value.code == "model_incomplete"
    attempt = store.trace("test")["attempts"][0]
    assert attempt["decision"] == '{"task_changes":'
    assert attempt["metadata"]["finish_reason"] == "length"
    assert attempt["metadata"]["usage"]["total_tokens"] == 77
    assert attempt["metadata"]["prompt_sha256"]


def test_invalid_memory_type_returns_validation_error(make_harness):
    h, _, _ = make_harness(search())
    with TestClient(create_app(h, token="")) as client:
        request = start()
        request["event"]["memory"] = {"context": 123}
        assert client.post("/v1/execution/advance", json=request).status_code == 422
