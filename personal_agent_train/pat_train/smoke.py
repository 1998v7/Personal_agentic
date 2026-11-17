"""不启动训练器,只用真实的 Gateway + vLLM 跑几条样本,检查交给训练器的轨迹长什么样。

    # 先起 vLLM:scripts/wsl/serve_vllm.sh
    python -m pat_train.smoke --data ~/pat/data/train.parquet --rows 0 1 --out ~/pat/smoke

Gateway 的配置与训练时 ``uni_agent.framework.entry.build_gateway_manager`` 一致(真实 tokenizer 与 chat template、
qwen3_coder 工具解析、enable_thinking=False、同样的长度预算)。每条样本走 ``pat_train.runner.run_task``,
即训练时的同一入口。输出:

* ``<out>/<session>/trajectories.jsonl``:Gateway 交给训练器的每条轨迹(token、mask、logprob、reward);
* ``<out>/<session>/readable.txt``:按 mask 分段解码后的文本(``[MODEL]`` 段是训练的 token,``[ENV]`` 段不训练);
* 屏幕上打印每个 session 的摘要:轨迹数、轮数、可训练 token、rollback 丢掉的 token、reward。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from examples.gateway.debug_launcher import OpenAICompletionsBackend, TemplateResultTokenIdsWrapper, trajectory_to_record
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor

from .runner import run_task

ROOT = Path(__file__).resolve().parents[1]


def segments(trajectory, tokenizer) -> list[tuple[str, str]]:
    """把 response 按 mask 切成连续段,解码成文本。"""
    out: list[tuple[str, str]] = []
    ids, mask = trajectory.response_ids, trajectory.response_mask
    start = 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or mask[i] != mask[start]:
            out.append(("MODEL" if mask[start] else "ENV", tokenizer.decode(ids[start:i], skip_special_tokens=False)))
            start = i
    return out


def summarize(trajectories) -> dict[str, Any]:
    return {
        "trajectories": len(trajectories),
        "turns": [t.num_turns for t in trajectories],
        "prompt_tokens": [len(t.prompt_ids) for t in trajectories],
        "response_tokens": [len(t.response_ids) for t in trajectories],
        "model_tokens": [sum(t.response_mask) for t in trajectories],
        "rollback_dropped": sum(int((t.extra_fields or {}).get("rollback_dropped_trainable_tokens", 0) or 0) for t in trajectories),
        "materialization": sorted({str((t.extra_fields or {}).get("materialization_reason")) for t in trajectories}),
    }


async def run_one(actor: _GatewayActor, tokenizer, sample: dict, args) -> dict[str, Any]:
    sid = f"smoke-{sample['extra_info']['index']}-{uuid.uuid4().hex[:6]}"
    handle = await actor.create_session(sid, sampling_params={"temperature": args.temperature, "top_p": 1.0, "top_k": -1})
    started = time.time()
    error = None
    result = None
    try:
        result = await run_task(
            session=handle, tools_kwargs=sample["extra_info"]["tools_kwargs"], raw_prompt=sample["prompt"],
            sample_index=sample["extra_info"]["index"], task_config_path=args.task_config, model_name=args.model_name,
        )
    except Exception as exc:  # noqa: BLE001  训练时这条样本会被丢弃;冒烟里记下原因
        error = f"{type(exc).__name__}: {exc}"
    trajectories = await actor.finalize_session(sid)
    session_dir = Path(args.out).expanduser() / sid
    session_dir.mkdir(parents=True, exist_ok=True)
    with (session_dir / "trajectories.jsonl").open("w", encoding="utf-8") as f:
        for i, t in enumerate(trajectories):
            f.write(json.dumps(trajectory_to_record(session_id=sid, trajectory_index=i, trajectory=t), ensure_ascii=False) + "\n")
    with (session_dir / "readable.txt").open("w", encoding="utf-8") as f:
        for i, t in enumerate(trajectories):
            f.write(f"######## trajectory {i}  turns={t.num_turns}  extra={json.dumps(t.extra_fields, ensure_ascii=False, default=str)}\n")
            f.write("======== [PROMPT]\n" + tokenizer.decode(t.prompt_ids, skip_special_tokens=False) + "\n")
            for kind, text in segments(t, tokenizer):
                f.write(f"======== [{kind}]\n{text}\n")
    summary = {
        "session": sid, "task": sample["extra_info"]["tools_kwargs"]["task"], "seconds": round(time.time() - started, 1),
        "reward": None if result is None else result.reward, "accuracy": None if result is None else result.accuracy,
        "extra_info": None if result is None else result.extra_info, "error": error, **summarize(trajectories),
    }
    (session_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


async def main_async(args) -> int:
    import pandas as pd
    from transformers import AutoConfig, AutoTokenizer

    rows = pd.read_parquet(Path(args.data).expanduser()).to_dict("records")
    samples = [_plain(rows[i]) for i in args.rows]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    hf_type = getattr(AutoConfig.from_pretrained(args.model_path, trust_remote_code=True), "model_type", None)
    actor = _GatewayActor(
        GatewayActorConfig(
            tokenizer=TemplateResultTokenIdsWrapper(tokenizer), tool_parser_name=args.tool_parser, rollout_backend="vllm",
            hf_model_type=hf_type, apply_chat_template_kwargs={"enable_thinking": False},
            allowed_request_sampling_param_keys={"temperature", "top_p", "top_k"},
            prompt_length=args.prompt_length, response_length=args.response_length,
        ),
        OpenAICompletionsBackend(backend_base_url=args.vllm_url, backend_model=args.model_name, timeout=args.timeout),
    )
    await actor.start()
    try:
        results = await asyncio.gather(*(run_one(actor, tokenizer, s, args) for s in samples))
    finally:
        await actor.shutdown()
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    return 0 if all(r["error"] is None for r in results) else 1


def _plain(value):
    """parquet 读出来的 numpy 数组 / 嵌套结构 → 普通 list / dict。"""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--rows", type=int, nargs="+", required=True, help="parquet 的行号,同时并发跑")
    p.add_argument("--out", required=True)
    p.add_argument("--model-path", default=str(Path.home() / "pat" / "models" / "Qwen3.5-2B"))
    p.add_argument("--model-name", default="Qwen3.5-2B")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8100/v1")
    p.add_argument("--task-config", default=str(ROOT / "configs" / "task_config.yaml"))
    p.add_argument("--tool-parser", default="qwen3_coder")
    p.add_argument("--prompt-length", type=int, default=12288)
    p.add_argument("--response-length", type=int, default=12288)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=600.0)
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
