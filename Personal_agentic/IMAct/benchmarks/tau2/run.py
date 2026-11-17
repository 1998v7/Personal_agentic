"""用 IMAct harness 跑 τ²-bench text-v1(6 语种 × airline / retail / telecom,每个 cell 50 个冻结 task,共 900 个)的入口。

    python -X utf8 -m IMAct.benchmarks.tau2.run --models models.yaml --domain all --language all --check
    python -X utf8 -m IMAct.benchmarks.tau2.run --models models.yaml --domain airline --language en zh --num-tasks 3 --out results/tau2-smoke

agent / 用户模拟器 / 裁判 三个角色统一从 models.yaml 选择。τ² 不涉及 Memory,这里没有 Memory 相关的任何配置:
被测对象就是 agent 模型本身(原生 function-calling,prompt 与上游 τ² 一致)。

结果目录:``<out>/<语言>-<domain>/<task>.trial<N>.json`` 每个 task 一份,``<out>/summary.json`` 是总汇总(含各 cell 的明细)。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from ...exec_layer.bridge import AsyncBridge
from ...harness import LLMCaller
from ...harness.batch import run_jobs, summary_from_results
from ...models.catalog import load_catalog
from ...models.config import create_provider
from ...retry import RetryPolicy
from ...trace import TraceRecorder
from .multilingual import LANGUAGES, Cell, load_base_cell, load_cell, text_v1_cells
from .runner import DEFAULT_MAX_ERRORS, DEFAULT_MAX_STEPS, Tau2Runner
from .tables import render as render_tables
from .tasks import DOMAINS

logger = logging.getLogger("imact.tau2")
DATA_HINT = "找不到 τ² 的任务数据:把 tau2-bench 的 data/tau2 放到 data/tau2,或设置环境变量 IMACT_TAU2_DATA 指向其 data 目录"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="用 IMAct harness 跑 τ²-bench text-v1 (6 语种 × airline / retail / telecom)")
    p.add_argument("--models", required=True, help="统一模型配置 models.yaml")
    p.add_argument("--domain", nargs="+", required=True, choices=(*DOMAINS, "all"), help="可多选;all = 三个 domain")
    p.add_argument("--language", nargs="+", default=["en"], choices=(*LANGUAGES, "all"), help="可多选;all = 六种语言。默认 en")
    p.add_argument("--benchmark", choices=("text-v1", "base"), default="text-v1",
                   help="text-v1:官方 2.0 评测集(每个 cell 50 个冻结 task,英文用审阅过的任务)。base:pre-2.0 的原始英文任务"
                        "(airline 50 / retail 114 / telecom 114),只支持 en,分数不能与 text-v1 榜单对照")
    p.add_argument("--agent-model", help="覆盖 roles.action,填模型别名")
    p.add_argument("--user-model", "--user-llm", dest="user_model", help="覆盖 roles.user")
    p.add_argument("--judge-model", "--evaluator-llm", dest="judge_model", help="覆盖 roles.judge")
    p.add_argument("--task-ids", nargs="*", default=None, help="按 canonical task ID 选(各语言通用,如 3);只给一个 domain 时才有意义")
    p.add_argument("--num-tasks", type=int, default=None, help="每个 cell 只取前 N 个 task(冒烟用)")
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="一段对话的步数上限")
    p.add_argument("--max-errors", type=int, default=DEFAULT_MAX_ERRORS, help="工具回执报错累计上限")
    p.add_argument("--max-task-retries", type=int, default=3, help="任务级重试")
    p.add_argument("--max-concurrency", type=int, default=1, help="同一个 cell 内同时跑几个 task")
    p.add_argument("--model-retries", type=int, default=3, help="单次模型请求的重试次数")
    p.add_argument("--communicate-scorer", choices=("semantic", "substring"), default="semantic",
                   help="必须告知的信息怎么判:semantic 用裁判模型(上游默认),substring 用子串匹配(不调裁判)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="结果目录;缺省为 results/tau2-<时间戳>")
    p.add_argument("--run-id", default=None)
    p.add_argument("--rerun", action="store_true", help="忽略已有结果文件,重新跑")
    p.add_argument("--no-judge", action="store_true", help="只跑对话,不评分(也不调裁判模型)")
    p.add_argument("--check", action="store_true", help="只检查配置 / 密钥 / 数据,不发请求")
    return p


def select_models(catalog, args) -> None:
    for role, override in (("action", args.agent_model), ("user", args.user_model), ("judge", args.judge_model)):
        alias, _ = catalog.selected(role, override)
        setattr(catalog.roles, role, alias)


def selected_cells(args) -> list[Cell]:
    """按命令行选出要跑的 cell(语言 × domain),顺序与官方 manifest 一致。"""
    domains = list(DOMAINS) if "all" in args.domain else list(dict.fromkeys(args.domain))
    languages = list(LANGUAGES) if "all" in args.language else list(dict.fromkeys(args.language))
    if args.task_ids is not None and len(domains) != 1:
        raise ValueError("--task-ids 只能配合单个 --domain 使用(不同 domain 的 task ID 不通用)")
    if args.benchmark == "base":
        if languages != ["en"]:
            raise ValueError("--benchmark base 只支持英文(--language en)")
        return [load_base_cell(d) for d in domains]
    return [load_cell(language, domain) for language, domain in text_v1_cells() if language in languages and domain in domains]


def needs_judge(cells_tasks, communicate_scorer: str) -> bool:
    """这批 task 里有没有哪个要用到裁判模型:NL 断言,或语义模式下的必须告知信息。"""
    for _cell, tasks in cells_tasks:
        for task in tasks:
            criteria = task.evaluation_criteria
            if criteria is None:
                continue
            basis = {getattr(b, "value", b) for b in criteria.reward_basis}
            if "NL_ASSERTION" in basis and criteria.nl_assertions:
                return True
            if communicate_scorer == "semantic" and "COMMUNICATE" in basis and criteria.communicate_info:
                return True
    return False


def check(args, catalog, cells_tasks) -> int:
    problems: list[str] = []
    roles = ["action", "user"] + (["judge"] if not args.no_judge and needs_judge(cells_tasks, args.communicate_scorer) else [])
    for role in roles:
        alias, model = catalog.selected(role)
        try:
            create_provider(model)
            print(f"{role} 角色: {alias} ({model.model}) ✓")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{role} 角色 {alias}: {exc}")
    if "judge" not in roles:
        print("裁判: 这批 task 不需要" if not args.no_judge else "裁判: 本次不评分")
    total = 0
    for cell, tasks in cells_tasks:
        total += len(tasks)
        note = "" if not cell.cross_lingual else f"  (跨语言;币种 {cell.regional['settings']['currency_code']})" if cell.regional else "  (跨语言)"
        print(f"  {cell.name:<14} {cell.label:<8} {len(tasks):>3} 个 task{note}")
    print(f"共 {len(cells_tasks)} 个 cell、{total} 个 task × {args.num_trials} 次 trial")
    if not total:
        problems.append("没有选中任何 task")
    for p in problems:
        print(f"  ✗ {p}")
    print("检查通过" if not problems else f"有 {len(problems)} 项需要处理")
    return 0 if not problems else 2


def summarize(cells_summary: dict[str, dict], *, run_id: str, agent_model: str, benchmark: str) -> dict:
    """总汇总。``avg_reward`` 是所有已评分 task 的平均;各 cell 的明细在 ``cells`` 里。"""
    cells, rewards = {}, []
    for name, s in cells_summary.items():
        rewards += [r["reward"] for r in s["results"] if r["reward"] is not None]
        cells[name] = {"completed": len(s["results"]), "scored": sum(1 for r in s["results"] if r["reward"] is not None),
                       "failed": len(s["failed"]), "skipped": len(s["skipped"]),
                       "evaluation_errors": s.get("num_evaluation_errors", 0), "avg_reward": s["avg_reward"], "failed_tasks": s["failed"]}
    total = lambda key: sum(c[key] for c in cells.values())  # noqa: E731
    return {"bench": "tau2", "run_id": run_id, "benchmark": benchmark, "agent_model": agent_model, "completed": total("completed"), "failed": total("failed"),
            "skipped": total("skipped"), "evaluation_errors": total("evaluation_errors"),
            "avg_reward": sum(rewards) / len(rewards) if rewards else None, "cells": cells}


def rebuild(out: Path) -> dict:
    """只读 ``out`` 里的结果文件重新汇总(不调用模型)。``python -m IMAct.report`` 用它把旧目录打印成表。"""
    old = _read_json(out / "summary.json")
    cells_summary = {}
    for cell_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        if any(cell_dir.glob("*.trial*.json")):
            cells_summary[cell_dir.name] = summary_from_results(cell_dir, previous=_read_json(cell_dir / "summary.json"))
    return summarize(cells_summary, run_id=old.get("run_id", out.name), agent_model=old.get("agent_model", "-"),
                     benchmark=old.get("benchmark", "text-v1"))


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    out = Path(args.out or f"results/tau2-{time.strftime('%Y%m%d-%H%M%S')}")
    catalog = load_catalog(args.models)
    select_models(catalog, args)
    print(json.dumps(catalog.snapshot(), ensure_ascii=False, indent=2))
    try:
        cells_tasks = [(cell, cell.select(args.task_ids, args.num_tasks)) for cell in selected_cells(args)]
    except FileNotFoundError:
        print(DATA_HINT, file=sys.stderr)
        return 2
    except ValueError as exc:  # 参数组合不合法 / 选的 task 不存在:一行说清楚,不打 traceback
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2
    judge_needed = not args.no_judge and needs_judge(cells_tasks, args.communicate_scorer)
    if args.check:
        return check(args, catalog, cells_tasks)
    if not any(tasks for _, tasks in cells_tasks):
        print(DATA_HINT, file=sys.stderr)
        return 2
    # 任何请求之前先确认所需角色的密钥都在,避免跑到一半才发现
    for role in ["action", "user"] + (["judge"] if judge_needed else []):
        create_provider(catalog.selected(role)[1])

    out.mkdir(parents=True, exist_ok=True)
    (out / f"models.resolved.{time.time_ns()}.json").write_text(json.dumps(catalog.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
    file_handler = logging.FileHandler(out / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    policy = RetryPolicy(max_retries=args.model_retries, backoff_s=1.0)
    bridge = AsyncBridge("imact-models")
    trace = TraceRecorder(out / "trace.jsonl")
    agent_llm = LLMCaller(catalog.selected("action")[1], bridge, policy=policy, trace=trace, trace_stage="agent")
    user_llm = LLMCaller(catalog.selected("user")[1], bridge, policy=policy)
    judge = LLMCaller(catalog.selected("judge")[1], bridge, policy=policy, trace=trace, trace_stage="judge") if judge_needed else None
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    cells_summary: dict[str, dict] = {}
    try:
        for cell, tasks in cells_tasks:
            print(f"\n=== {cell.name} ({len(tasks)} 个 task) ===", flush=True)
            runner = Tau2Runner(cell, agent_llm=agent_llm, user_llm=user_llm, judge=judge, score=not args.no_judge, max_steps=args.max_steps,
                                max_errors=args.max_errors, communicate_scorer=args.communicate_scorer, trace=trace)
            cells_summary[cell.name] = run_jobs(
                tasks, lambda task, run_id, seed, runner=runner: runner.run_task(task, run_id=run_id, seed=seed),
                out=out / cell.name, run_id=run_id, num_trials=args.num_trials, seed=args.seed, max_task_retries=args.max_task_retries,
                max_concurrency=args.max_concurrency, rerun=args.rerun,
                summary_extra={"bench": "tau2", "benchmark": cell.label, "language": cell.language, "domain": cell.domain,
                               "agent_model": agent_llm.cfg.model})
    finally:
        for llm in (agent_llm, user_llm, judge):
            if llm is not None:
                llm.close()
        bridge.close()
    summary = summarize(cells_summary, run_id=run_id, agent_model=agent_llm.cfg.model, benchmark=args.benchmark)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + render_tables(summary))
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
