import importlib.util
import json
import os
from pathlib import Path

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.anyio
async def test_inprocess_smoke_does_not_use_caller_configuration(tmp_path, monkeypatch):
    missing = str(tmp_path / "caller-model.toml")
    monkeypatch.setenv("IMACT_CONFIG", missing)
    monkeypatch.setenv("IMACT_MODEL_MODE", "chat")
    monkeypatch.setenv("IMACT_MODEL_TIMEOUT", "invalid-caller-value")
    monkeypatch.setenv("IMACT_DECISION_TIMEOUT", "invalid-caller-value")
    await load_script("verify_inprocess").main(output=tmp_path / "smoke")
    result = json.loads((tmp_path / "smoke/asgi-smoke.json").read_text())
    assert result["passed"] and result["model_attempts"] == 7
    assert os.environ["IMACT_CONFIG"] == missing
    assert os.environ["IMACT_MODEL_MODE"] == "chat"


def test_http_smoke_child_environment_is_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("IMACT_CONFIG", "caller-model.toml")
    monkeypatch.setenv("IMACT_MODEL_API_KEY", "caller-key-not-for-demo")
    monkeypatch.setenv("IMACT_DECISION_TIMEOUT", "invalid-caller-value")
    child = load_script("verify_http").smoke_environment(tmp_path)
    assert "IMACT_CONFIG" not in child and "IMACT_MODEL_API_KEY" not in child
    assert "IMACT_DECISION_TIMEOUT" not in child
    assert child["IMACT_MODEL_MODE"] == "demo"
    assert child["IMACT_DB_PATH"] == str(tmp_path / "http-workspace.sqlite3")
    assert os.environ["IMACT_CONFIG"] == "caller-model.toml"
