"""受限环境中验证完整 ASGI 协议，不监听端口，不替代真实 socket 冒烟。"""

import asyncio
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

import httpx2 as httpx

from imact.api import create_app


async def main(output: Path | None = None):
    root = Path(__file__).resolve().parents[1]
    output = output or root / "data/verification"
    output.mkdir(parents=True, exist_ok=True)
    smoke_env = {key: value for key, value in os.environ.items() if not key.startswith("IMACT_")}
    smoke_env.update(
        {
            "IMACT_MODEL_MODE": "demo",
            "IMACT_SERVICE_TOKEN": "test-token",
            "IMACT_DB_PATH": str(output / "asgi.sqlite3"),
        }
    )
    with patch.dict(os.environ, smoke_env, clear=True):
        app = create_app()
    transport = httpx.ASGITransport(app=app)
    spec = importlib.util.spec_from_file_location(
        "outer_example", root / "examples/outer_harness_example.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = await module.run_roundtrip("http://test", "test-token", transport=transport)
    report["mode"] = "inprocess_asgi_demo_model_no_socket"
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get(
            f"http://test/v1/execution/tasks/{report['task_id']}/trace",
            headers={"Authorization": "Bearer test-token"},
        )
        response.raise_for_status()
    report["model_attempts"] = len(response.json()["attempts"])
    report["passed"] = (
        report["committed_events"] == report["model_attempts"] == 7
        and report["http_advance_requests"] == 14
    )
    assert report["passed"]
    (output / "asgi-smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "outputs"}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
