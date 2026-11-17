"""启动临时 localhost 服务验证七轮 HTTP 闭环，结束后关闭进程；不调用外部模型。"""

import asyncio
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx2 as httpx


def smoke_environment(output: Path) -> dict:
    """仅保留运行 Python 所需环境；不继承调用者的模型/服务配置。"""
    env = {key: value for key, value in os.environ.items() if not key.startswith("IMACT_")}
    env.update(
        IMACT_MODEL_MODE="demo",
        IMACT_DB_PATH=str(output / "http-workspace.sqlite3"),
        IMACT_SERVICE_TOKEN="local-smoke-token",
    )
    return env


def main():
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    output = root / "data/verification"
    output.mkdir(parents=True, exist_ok=True)
    env = smoke_environment(output)
    with (output / "http-server.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "imact.cli", "serve", "--port", str(port)],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("服务提前退出，请检查 data/verification/http-server.log")
                try:
                    if httpx.get(base + "/health", timeout=1, trust_env=False).status_code == 200:
                        break
                except httpx.RequestError:
                    pass
                time.sleep(0.05)
            else:
                raise RuntimeError("临时服务启动超时")
            spec = importlib.util.spec_from_file_location(
                "outer_example", root / "examples/outer_harness_example.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            report = asyncio.run(module.run_roundtrip(base, "local-smoke-token"))
            response = httpx.get(
                base + f"/v1/execution/tasks/{report['task_id']}/trace",
                headers={"Authorization": "Bearer local-smoke-token"},
                trust_env=False,
            )
            response.raise_for_status()
            report["model_attempts"] = len(response.json()["attempts"])
            report["passed"] = (
                report["committed_events"] == report["model_attempts"] == 7
                and report["http_advance_requests"] == 14
            )
            if not report["passed"]:
                raise RuntimeError("HTTP 闭环或事件去重计数不符合预期")
            (output / "http-smoke.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {k: v for k, v in report.items() if k != "outputs"},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            print("临时 HTTP 服务已停止。")


if __name__ == "__main__":
    main()
