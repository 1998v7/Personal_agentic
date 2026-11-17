"""从原始任务取时间和指令,通过异步适配器查询真实本地 Memory;不重建历史。"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from IMAct.exec_layer.config import load_outer_config
from IMAct.exec_layer.memory_local import LocalMemoryClient
from IMAct.trace import TraceRecorder, trace_scope


async def probe(args):
    cfg = load_outer_config(args.config)
    if cfg.memory.backend != "local":
        raise ValueError("smoke_local_memory requires memory.backend=local")
    harness_path = Path(cfg.memory.local.config_path)
    harness = json.loads(harness_path.read_text(encoding="utf-8"))
    tasks_path = (harness_path.parent / harness["benchmarks"]["vitabench"]["tasks_path"]).resolve()
    users = json.loads(tasks_path.read_text(encoding="utf-8"))
    user, task = next((u, t) for u in users for t in u["subtasks"] if t["subtask_id"] == args.subtask_id)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    client = LocalMemoryClient(cfg.memory.local, trace=TraceRecorder(out / "trace.jsonl"))
    started = time.perf_counter()
    try:
        await client.start()
        request = dict(user_id=user["user_id"], query_id=task["subtask_id"],
                       user_query=task["instruction"], timestamp=task["current_time"])
        if args.check:
            result = {"status": "configuration_loaded", "model_calls": 0}
        else:
            with trace_scope(query_id=task["subtask_id"], phase="prefetch"):
                result = await client.query(**request)
        report = {"request": request, "result": result, "duration_s": round(time.perf_counter() - started, 2),
                  "transport": "local", "client_path": cfg.memory.local.client_path,
                  "config_path": str(harness_path), "tasks_path": str(tasks_path)}
        (out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if args.check or result["status"] == "ok" else 1
    finally:
        await client.aclose()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.vitabench.yaml")
    p.add_argument("--subtask-id", default="sub_A891207_1")
    p.add_argument("--out", default=f"results/local-memory-{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--check", action="store_true", help="仅检查配置加载,不调用模型")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    return asyncio.run(probe(args))


if __name__ == "__main__":
    raise SystemExit(main())
