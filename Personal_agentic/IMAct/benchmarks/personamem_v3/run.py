"""用 IMAct harness 跑 PersonaMem-v3(官方 ``llm_memory`` 模式,记忆来自外部 Memory 服务)的入口。

    python -X utf8 -m IMAct.benchmarks.personamem_v3.run --models models.yaml --users 25 --check
    python -X utf8 -m IMAct.benchmarks.personamem_v3.run --models models.yaml --users 25 --num-queries 10 --out results/pm3-smoke

每道题:向 Memory 服务 ``query``(统一调用接口:bench=personamem_v3, user_id=persona 目录名, task={query_id, user_query, timestamp=ts})
拿回 ``context`` → 接上截至该时刻的日历 → 按官方 llm_memory 的 prompt 让 action 模型作答 → 按官方规则评分(裁判 = judge 角色)。
用户历史由 Memory 服务那边预先 build 好,这里不调用 build。

结果:``<out>/<persona>/<query>.trial0.json`` 每题一份,``<out>/summary.json`` 是总汇总(官方口径的 micro 分 + 各题型 / 类别)。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from ...exec_layer.bridge import AsyncBridge
from ...exec_layer.memory_mps import MpsMemoryClient, MpsMemoryConfig
from ...harness import LLMCaller
from ...harness.batch import result_path, run_jobs
from ...models.catalog import load_catalog
from ...models.config import create_provider
from ...retry import RetryPolicy
from ...trace import TraceRecorder
from . import data as D
from .pm3.evaluation.task_registry import DROPPED_TASK_TYPES, PRIMARY_METRIC, normalize_task_type
from .runner import DEFAULT_MEMORY_TOKEN_CAP, PersonaMemRunner
from .scoring import summarize
from .tables import render as render_tables

logger = logging.getLogger("imact.personamem_v3")
BENCH = "personamem_v3"
DEFAULT_MEMORY_URL = "https://paiplusinferencepre.alipay.com/inference/2220c69b1f3f829b_identity_memory/V0.1"
DATA_HINT = "找不到 PersonaMem-v3 数据:先运行 python -X utf8 -m IMAct.benchmarks.personamem_v3.download(或设 IMACT_PERSONAMEM_V3_DATA)"


@dataclass
class Job:
    """批处理的一个 job = 一道题。``id`` 用 query_id(跨 persona 唯一)。"""

    id: str
    user_id: str
    seq: int
    item: dict


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="用 IMAct harness 跑 PersonaMem-v3(llm_memory 模式,外部 Memory 服务)")
    p.add_argument("--models", required=True, help="统一模型配置 models.yaml")
    p.add_argument("--agent-model", help="覆盖 roles.action(作答模型),填模型别名")
    p.add_argument("--judge-model", "--evaluator-llm", dest="judge_model", help="覆盖 roles.judge(裁判模型)")
    p.add_argument("--users", nargs="*", default=None, help="persona 目录名,如 25 36;缺省为已下载的全部")
    p.add_argument("--num-users", type=int, default=None, help="只取前 N 个 persona")
    p.add_argument("--task-types", nargs="*", default=None, help="只跑这些题型(官方 task_type 名)")
    p.add_argument("--num-queries", type=int, default=None, help="每个 persona 只跑前 N 道题(按 test.json 顺序;冒烟用)")
    p.add_argument("--max-concurrency", type=int, default=4, help="同一个 persona 内同时跑几道题(persona 之间按顺序)")
    p.add_argument("--max-task-retries", type=int, default=2, help="一道题整体失败(Memory / 模型请求重试后仍失败)后的重试次数")
    p.add_argument("--model-retries", type=int, default=3, help="单次模型请求的重试次数")
    p.add_argument("--no-judge", action="store_true", help="不调用裁判(依赖裁判的题型没有分数;只用于快速看作答与费用)")
    p.add_argument("--memory-url", default=os.environ.get("IMACT_MEMORY_URL") or DEFAULT_MEMORY_URL, help="Memory 网关地址(环境变量 IMACT_MEMORY_URL)")
    p.add_argument("--memory-app-name", default="imact", help="MPS-app-name 请求头")
    p.add_argument("--memory-timeout", type=float, default=20.0, help="单次 Memory 请求的超时秒数")
    p.add_argument("--memory-retries", type=int, default=2, help="单次 Memory 请求的重试次数(网关偶发超时)")
    p.add_argument("--memory-concurrency", type=int, default=16, help="同时在途的 Memory 请求上限")
    p.add_argument("--memory-token-cap", type=int, default=DEFAULT_MEMORY_TOKEN_CAP,
                   help="注入 prompt 的记忆最多多少 token(官方 llm_memory 为 4096;超出按行截断)")
    p.add_argument("--memory-query-placeholder", choices=("empty", "verbatim"), default="empty",
                   help="没有用户正文的题(test.json 里 user_query 是 '[system prompt] …' 占位):empty=给 Memory 传空字符串(接口约定),verbatim=原样传")
    p.add_argument("--out", default=None, help="结果目录;缺省 results/pm3-<时间戳>")
    p.add_argument("--run-id", default=None)
    p.add_argument("--rerun", action="store_true", help="忽略已有结果文件,重新跑")
    p.add_argument("--check", action="store_true", help="只检查数据 / 密钥 / Memory 网关(发 1 次只读 query),不调用模型")
    return p


def select_models(catalog, args) -> None:
    for role, override in (("action", args.agent_model), ("judge", args.judge_model)):
        alias, _ = catalog.selected(role, override)
        setattr(catalog.roles, role, alias)


def select_users(args) -> list[str]:
    available = D.list_users()
    if not available:
        raise FileNotFoundError(DATA_HINT)
    if args.users:
        missing = [u for u in args.users if u not in available]
        if missing:
            raise ValueError(f"这些 persona 没有数据(未下载?): {', '.join(missing)}")
        users = [u for u in available if u in set(args.users)]
    else:
        users = available
    if args.num_users is not None:
        if args.num_users < 1:
            raise ValueError("--num-users 必须 >= 1")
        users = users[: args.num_users]
    for u in users:
        if D.missing_files(u):
            raise ValueError(f"persona {u} 缺少文件: {', '.join(D.missing_files(u))}")
    return users


def select_jobs(user_id: str, args) -> list[Job]:
    items = D.load_test(user_id)
    wanted = {normalize_task_type(t) for t in args.task_types} if args.task_types else None
    if wanted:
        unknown = sorted(t for t in wanted if t not in PRIMARY_METRIC)
        if unknown:
            raise ValueError(f"未知题型: {', '.join(unknown)}")
    jobs = []
    for seq, item in enumerate(items):
        tt = normalize_task_type(item.get("task_type", ""))
        if tt in DROPPED_TASK_TYPES or (wanted and tt not in wanted):
            continue
        jobs.append(Job(id=item["query_id"], user_id=user_id, seq=seq, item=item))
    if args.num_queries is not None:
        if args.num_queries < 1:
            raise ValueError("--num-queries 必须 >= 1")
        jobs = jobs[: args.num_queries]
    return jobs


def memory_config(args) -> MpsMemoryConfig:
    return MpsMemoryConfig(url=args.memory_url, bench=BENCH, app_name=args.memory_app_name, timeout_s=args.memory_timeout,
                           max_concurrency=args.memory_concurrency)


def check(args, catalog, plan: list[tuple[str, list[Job]]]) -> int:
    problems: list[str] = []
    roles = ["action"] + ([] if args.no_judge else ["judge"])
    for role in roles:
        alias, model = catalog.selected(role)
        try:
            create_provider(model)
            print(f"{role} 角色: {alias} ({model.model}) ✓")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{role} 角色 {alias}: {exc}")
    if args.no_judge:
        print("裁判: 本次不调用(依赖裁判的题型没有分数)")
    total = 0
    for uid, jobs in plan:
        total += len(jobs)
        print(f"  persona {uid:<5}{len(jobs):>5} 道题")
    print(f"共 {len(plan)} 个 persona、{total} 道题")
    if not total:
        problems.append("没有选中任何题")
    print(f"Memory 网关: {args.memory_url}  (bench={BENCH}, 超时 {args.memory_timeout:g}s)")
    if total:
        uid, jobs = next((u, j) for u, j in plan if j)
        job = jobs[0]
        bridge = AsyncBridge("imact-memory-check")
        client = MpsMemoryClient(memory_config(args), retry=RetryPolicy(max_retries=args.memory_retries, backoff_s=0.5))
        try:
            from .runner import memory_user_query

            started = time.time()
            reply = bridge.run(client.query(user_id=uid, query_id=job.item["query_id"],
                                            user_query=memory_user_query(job.item, args.memory_query_placeholder), timestamp=int(job.item["ts"])))
            if reply["status"] in ("ok", "empty"):
                print(f"  试查 {job.item['query_id']}: {reply['status']},context {len(reply.get('context') or '')} 字符,{time.time() - started:.1f}s ✓")
            else:
                problems.append(f"Memory 网关试查 {job.item['query_id']} 失败: {reply.get('error')}")
        finally:
            bridge.run(client.aclose())
            bridge.close()
    for p in problems:
        print(f"  ✗ {p}")
    print("检查通过" if not problems else f"有 {len(problems)} 项需要处理")
    return 0 if not problems else 2


def collect(out: Path, plan: list[tuple[str, list[Job]]]) -> tuple[list[dict], dict[str, list[dict]]]:
    """读回本次选中题目的结果文件(包括以前跑完、这次跳过的)。"""
    runs, per_user = [], {}
    for uid, jobs in plan:
        rows = []
        for job in jobs:
            path = result_path(out / uid, job.id, 0)
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
        per_user[uid] = rows
        runs.extend(rows)
    return runs, per_user



def compose_summary(runs: list[dict], per_user: dict[str, list[dict]], *, failed: int, meta: dict) -> dict:
    summary = {**meta, **summarize(runs, failed=failed), "personas": {uid: summarize(rows) for uid, rows in per_user.items() if rows}}
    for s in summary["personas"].values():  # 每个 persona 只保留头条,明细看总表
        for k in ("by_task", "by_axis"):
            s.pop(k, None)
    return summary


def rebuild(out: Path) -> dict:
    """只读 ``out`` 里的结果文件重新汇总(不调用模型)。``python -m IMAct.report`` 用它把旧目录打印成表。"""
    try:
        old = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    per_user, failed = {}, 0
    for d in sorted((p for p in out.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(d.glob("*.trial*.json"))]
        if not rows:
            continue
        per_user[d.name] = rows
        done = {r["task_id"] for r in rows}
        try:
            prev = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prev = {}
        failed += sum(1 for f in prev.get("failed") or [] if f.get("task_id") not in done)
    meta = {k: old.get(k) for k in ("run_id", "mode", "agent_model", "judge_model")}
    return compose_summary([r for rows in per_user.values() for r in rows], per_user, failed=failed, meta={"bench": BENCH, **meta})


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    load_dotenv()
    out = Path(args.out or f"results/pm3-{time.strftime('%Y%m%d-%H%M%S')}")
    catalog = load_catalog(args.models)
    select_models(catalog, args)
    print(json.dumps(catalog.snapshot(), ensure_ascii=False, indent=2))
    try:
        plan = [(uid, select_jobs(uid, args)) for uid in select_users(args)]
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2
    if args.check:
        return check(args, catalog, plan)
    if not any(jobs for _, jobs in plan):
        print("没有选中任何题", file=sys.stderr)
        return 2
    for role in ["action"] + ([] if args.no_judge else ["judge"]):
        create_provider(catalog.selected(role)[1])  # 任何请求之前先确认密钥都在

    out.mkdir(parents=True, exist_ok=True)
    (out / f"models.resolved.{time.time_ns()}.json").write_text(json.dumps(catalog.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
    file_handler = logging.FileHandler(out / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    policy = RetryPolicy(max_retries=args.model_retries, backoff_s=1.0)
    bridge = AsyncBridge("imact-models")
    trace = TraceRecorder(out / "trace.jsonl")
    answer_llm = LLMCaller(catalog.selected("action")[1], bridge, policy=policy, trace=trace, trace_stage="agent")
    judge_llm = None if args.no_judge else LLMCaller(catalog.selected("judge")[1], bridge, policy=policy, trace=trace, trace_stage="judge")
    memory = MpsMemoryClient(memory_config(args), retry=RetryPolicy(max_retries=args.memory_retries, backoff_s=0.5), trace=trace)

    def memory_query(**kw):
        return bridge.run(memory.query(**kw))

    runner = PersonaMemRunner(backend_dir=D.backend_dir(), answer_llm=answer_llm, judge_llm=judge_llm, memory_query=memory_query,
                              memory_token_cap=args.memory_token_cap, query_placeholder=args.memory_query_placeholder,
                              model_name=answer_llm.cfg.model)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    failed_total = 0
    try:
        for uid, jobs in plan:
            if not jobs:
                continue
            print(f"\n=== persona {uid} ({len(jobs)} 道题) ===", flush=True)
            s = run_jobs(jobs, lambda job, run_id, seed: runner.run_item(job.user_id, job.seq, job.item, run_id=run_id, seed=seed),
                         out=out / uid, run_id=run_id, num_trials=1, max_task_retries=args.max_task_retries,
                         max_concurrency=args.max_concurrency, rerun=args.rerun,
                         row_fields=lambda r: {"task_type": r["task_type"], "status": r["status"], "accuracy_pct": r["accuracy_pct"]},
                         summary_extra={"bench": BENCH, "user_id": uid, "agent_model": answer_llm.cfg.model})
            failed_total += len(s["failed"])
    finally:
        bridge.run(memory.aclose())
        for llm in (answer_llm, judge_llm):
            if llm is not None:
                llm.close()
        bridge.close()
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()

    runs, per_user = collect(out, plan)
    summary = compose_summary(runs, per_user, failed=failed_total, meta={
        "run_id": run_id, "bench": BENCH, "mode": "llm_memory (external memory)", "agent_model": answer_llm.cfg.model,
        "judge_model": None if judge_llm is None else judge_llm.cfg.model})
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + render_tables(summary))
    return 0 if not failed_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
