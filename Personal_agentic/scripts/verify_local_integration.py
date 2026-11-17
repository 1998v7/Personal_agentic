"""离线联调:真实 MemoryClient/快照/Navigator/Reader + 固定模型和工具回执。

只替换模型传输为 httpx.MockTransport,不发外部请求、不构建历史、不产生评测分数。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from IMAct.exec_layer import build_runtime, load_outer_config
from IMAct.exec_layer.memory_local import LocalMemoryClient, _create_client
from IMAct.trace import TraceRecorder
from tests.exec_helpers import DemoProvider
from IMAct.harness import HarnessAgent
from tests.test_vitabench_bridge import drive, spec_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.vitabench.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_outer_config(args.config)
    cfg.memory.backend = "local"
    cfg.memory.require_prefetch = True
    cfg.exec.db_path = str(out / "exec.sqlite3")
    trace = TraceRecorder(out / "trace.jsonl")
    model_calls, clients = [], []

    def transport(request):
        body = json.loads(request.content)
        messages = body["messages"]
        if body.get("tools"):
            # Navigator 初始观察来自真实快照中的 profile;只引用已展示的证据。
            observed = json.loads(messages[-1]["content"])["memory"]["blocks"]
            refs = [b["evidence_ref"] for b in observed]
            assert refs, "published snapshot must expose evidence"
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"nav-{len(model_calls)}", "type": "function", "function": {
                    "name": "finish", "arguments": json.dumps({"evidence_refs": refs, "status": "sufficient", "note": "offline protocol fixture"})}}]}
            role, finish = "navigator", "tool_calls"
        else:
            material = json.loads(messages[-1]["content"])["memory_material"]
            blocks = material["blocks"]
            assert blocks and all(b.get("text") for b in blocks)
            # 原样转交已检索文本,不宣称固定响应具备语义总结能力。
            message = {"role": "assistant", "content": "离线联调固定 Reader 输出:\n" + "\n".join(b["text"] for b in blocks)}
            role, finish = "reader", "stop"
        model_calls.append({"role": role, "messages": messages, "response": message})
        return httpx.Response(200, json={"id": f"offline-{len(model_calls)}", "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})

    def client_factory(local_cfg):
        client = _create_client(local_cfg)
        client._service.backend.runtime.transport = httpx.MockTransport(transport)
        clients.append(client)
        return client

    local = LocalMemoryClient(cfg.memory.local, trace=trace, client_factory=client_factory)
    # 预取之后再让 Action 主动查询一次,验证两条入口共享客户端并继续执行。
    provider = DemoProvider()
    rt = build_runtime(cfg, provider=provider, memory_client=local, trace=trace)
    try:
        agent = HarnessAgent(runtime=rt, run_id="offline-local")
        with patch("IMAct.exec_layer.memory_mps.MpsMemoryClient._post", side_effect=AssertionError("unexpected MPS call")):
            ctx = agent.begin_episode(spec_for(memory_cfg=cfg.memory))
            replies, _ = drive(agent)
        assert ctx.memory["status"] == "ok"
        assert len(clients) == 1 and len(model_calls) == 4
        assert replies[-1].raw_data["imact"]["mode"] == "final"
        assert replies[-1].raw_data["imact"]["outcome"] == "completed"
        inputs = [json.loads(r.messages[-1]["content"]) for r in provider.requests]
        assert any(json.dumps(ctx.memory["context"], ensure_ascii=False) in json.dumps(c, ensure_ascii=False) for c in inputs)
    finally:
        rt.close()
    import sqlite3
    with sqlite3.connect(cfg.exec.db_path) as db:
        integrity = db.execute("pragma integrity_check").fetchone()[0]
        event_types = [json.loads(r[0])["event"]["type"] for r in db.execute("select input from events")]
    report = {"status": "ok", "mode": "offline: scripted model transport and synthetic business receipts",
        "external_model_calls": 0, "reward": None, "real_memory_client_instances": len(clients),
        "native_model_fixture_calls": len(model_calls), "memory_context_chars": len(ctx.memory["context"]),
        "prefetch_status": ctx.memory["status"], "timestamp": ctx.spec.timestamp,
        "action_decisions": len(provider.requests), "last_action": replies[-1].raw_data["imact"],
        "sqlite_integrity_check": integrity, "event_types": event_types,
        "memory_closed": clients[0]._closed}
    (out / "model-fixtures.json").write_text(json.dumps(model_calls, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
