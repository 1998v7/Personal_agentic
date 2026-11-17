import json

import httpx2 as httpx
import pytest
from conftest import reply, search, start
from fastapi.testclient import TestClient

from imact.api import create_app
from imact.errors import HarnessError
from imact.model import ChatModel
from imact.settings import Settings


def test_missing_model_configuration_never_uses_demo(tmp_path, monkeypatch):
    monkeypatch.delenv("IMACT_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("IMACT_MODEL_NAME", raising=False)
    path = tmp_path / "model.toml"
    path.write_text('[model]\nbase_url=""\nname=""\n')
    with pytest.raises(HarnessError) as caught:
        Settings.load(path).model()
    assert caught.value.code == "model_not_configured"


def test_configured_api_returns_actions_and_waits_for_outer_receipts(tmp_path, monkeypatch):
    config = tmp_path / "model.toml"
    config.write_text("""[model]
base_url = "https://model.example/v1"
name = "configured-model"
api_key_env = "TEST_ACTION_MODEL_KEY"
[runtime]
output_dir = "state"
""")
    monkeypatch.setenv("TEST_ACTION_MODEL_KEY", "test-secret-never-in-trace")
    requests = []

    def handler(request):
        assert str(request.url) == "https://model.example/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-secret-never-in-trace"
        body = json.loads(request.content)
        assert body["model"] == "configured-model"
        requests.append(body)
        value = search() if len(requests) == 1 else reply("已读取上层回传的结果", mode="inform")
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]
            },
        )

    original = Settings.model
    monkeypatch.setattr(
        Settings, "model", lambda self: original(self, transport=httpx.MockTransport(handler))
    )
    app = create_app(config=str(config), token="")
    assert isinstance(app.state.harness.model, ChatModel)
    with TestClient(app) as client:
        first = client.post("/v1/execution/advance", json=start())
        assert first.status_code == 200
        response = first.json()
        call = response["action"]["calls"][0]
        assert len(requests) == 1
        workspace = client.get("/v1/execution/tasks/test").json()["workspace"]
        assert list(workspace["pending_calls"]) == [call["call_id"]]
        assert not workspace["facts"] and not workspace["orders"]
        second = client.post(
            "/v1/execution/advance",
            json={
                "task_id": "test",
                "event_id": "e2",
                "base_rev": response["rev"],
                "event": {
                    "type": "tool_results",
                    "results": [
                        {
                            "call_id": call["call_id"],
                            "name": call["name"],
                            "delivery": "received",
                            "content": "Store(name=上层查到的门店, store_id=OUTER_STORE, score=4.8)",
                        }
                    ],
                },
            },
        )
        assert second.status_code == 200 and len(requests) == 2
        workspace = client.get("/v1/execution/tasks/test").json()["workspace"]
        assert workspace["facts"]["OUTER_STORE"]["name"] == "上层查到的门店"
        assert not workspace["pending_calls"]
        assert "test-secret-never-in-trace" not in client.get("/v1/execution/tasks/test/trace").text


@pytest.mark.parametrize("source", ["file", "environment"])
@pytest.mark.parametrize(
    "model_timeout,decision_timeout,expected",
    [
        (300, None, 330),
        (300, 350, 350),
        (300, 120, None),
        (-1, None, None),
    ],
)
def test_service_timeout_relationship(
    tmp_path, monkeypatch, source, model_timeout, decision_timeout, expected
):
    import os

    for name in list(os.environ):
        if name.startswith("IMACT_"):
            monkeypatch.delenv(name)
    if source == "file":
        config = tmp_path / "model.toml"
        text = (
            f'[model]\nbase_url="https://model.example/v1"\nname="test"\n'
            f'timeout_seconds={model_timeout}\n[runtime]\noutput_dir="state"\n'
        )
        if decision_timeout is not None:
            text += f"decision_timeout_seconds={decision_timeout}\n"
        config.write_text(text)
        kwargs = {"config": str(config)}
    else:
        monkeypatch.setenv("IMACT_MODEL_BASE_URL", "https://model.example/v1")
        monkeypatch.setenv("IMACT_MODEL_NAME", "test")
        monkeypatch.setenv("IMACT_MODEL_TIMEOUT", str(model_timeout))
        monkeypatch.setenv("IMACT_DB_PATH", str(tmp_path / "state.sqlite3"))
        if decision_timeout is not None:
            monkeypatch.setenv("IMACT_DECISION_TIMEOUT", str(decision_timeout))
        kwargs = {}
    if expected is None:
        with pytest.raises(HarnessError) as caught:
            create_app(**kwargs)
        assert caught.value.code == "invalid_config"
    else:
        app = create_app(**kwargs)
        assert app.state.harness.model.timeout == model_timeout
        assert app.state.harness.decision_timeout == expected
        with TestClient(app) as client:
            health = client.get("/health").json()
        assert health["decision_timeout_seconds"] == expected
        assert health["recommended_client_timeout_seconds"] == expected + 10
