"""用 IMAct harness 跑 VitaBench 2.0 personalization 的入口。

    python -X utf8 -m IMAct.benchmarks.vitabench.run --config config.vitabench.yaml --check
    python -X utf8 -m IMAct.benchmarks.vitabench.run --config config.vitabench.yaml --num-tasks 1 --out results/smoke

Action / 用户模拟器 / 裁判统一从根目录 models.yaml 选择。VitaBench 只提供任务、环境与工具;
用户模拟、对话循环与评分都由 IMAct 自己的 harness 完成。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

from ...exec_layer import ExecRuntime, OuterConfig, build_runtime, load_outer_config
from ...exec_layer.bridge import AsyncBridge
from ...exec_layer.config import apply_memory_overrides
from ...exec_layer.memory_local import LocalMemoryClient
from ...harness import LLMCaller
from ...harness.batch import run_jobs, summary_from_results
from ...models.catalog import load_catalog
from ...models.config import create_provider
from .personalization import PersonalizationRunner
from .tables import render as render_tables
from .tasks import get_tasks

logger = logging.getLogger("imact.vitabench")

DATA_HINT = (
    "找不到 personalization 任务数据。请先下载:\n"
    "  将 personalization 的 tasks.json 放到 data/vitabench/tasks.json\n"
    "完成后应有 data/vitabench/tasks.json"
)


def simulate(task, *, rt: ExecRuntime, run_id: str, user_llm: LLMCaller, judge: LLMCaller, max_steps: int, language: str, seed: int | None):
    """一个 PersonalizationTask(一个用户,多个 subtask)的完整模拟。"""
    return PersonalizationRunner(task, rt=rt, user_llm=user_llm, judge=judge, run_id=run_id, max_steps=max_steps, language=language, seed=seed).run()


def subtask_fields(record: dict) -> dict:
    """汇总里每个用户那一行带上各 subtask 的得分与业务域(结果表按域拆分)。旧结果没有 domain 时只有得分。"""
    info = (record.get("reward_info") or {}).get("info") or {}
    rewards, domains = info.get("subtask_rewards") or {}, info.get("subtask_domains") or {}
    return {"subtasks": [{"index": k.removeprefix("subtask_").removesuffix("_reward"), "reward": v,
                          "domain": domains.get(k.removesuffix("_reward"))} for k, v in rewards.items()]}


def run_all(args: argparse.Namespace, cfg: OuterConfig, rt: ExecRuntime, tasks: list, simulate_fn=simulate) -> dict[str, Any]:
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    return run_jobs(
        tasks,
        lambda task, run_id, seed: simulate_fn(task, rt=rt, run_id=run_id, max_steps=args.max_steps, language=args.language, seed=seed),
        out=Path(args.out), run_id=run_id, num_trials=args.num_trials, seed=args.seed, max_task_retries=args.max_task_retries,
        max_concurrency=args.max_concurrency, rerun=args.rerun, summary_extra={"bench": "vitabench", "model": cfg.model.model},
        row_fields=subtask_fields,
    )


def rebuild(out: Path) -> dict:
    """只读 ``out`` 里的结果文件重新汇总(不调用模型)。``python -m IMAct.report`` 用它把旧目录打印成表。"""
    try:
        old = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    summary = summary_from_results(out, row_fields=subtask_fields, previous=old)
    summary.setdefault("bench", "vitabench")
    return summary


def check(cfg: OuterConfig, args: argparse.Namespace, tasks: list, catalog=None) -> int:
    """不发任何模型请求,只检查配置 / 密钥 / 数据 / 工具是否齐备。"""
    problems: list[str] = []
    print(f"被测模型: provider={cfg.model.provider} model={cfg.model.model!r} base_url={cfg.model.base_url!r}")
    try:
        create_provider(cfg.model)
        print("  ✓ 模型配置与密钥环境变量可用")
    except Exception as exc:
        problems.append(f"模型配置: {exc}")
    if catalog is not None:
        for role in ("user", "judge"):
            alias, model = catalog.selected(role)
            try:
                create_provider(model)
                print(f"{role} 角色: {alias} ({model.model}) ✓")
            except Exception as exc:
                problems.append(f"{role} 角色 {alias}: {exc}")
    if cfg.memory.backend == "local":
        print(f"Memory 本地: {cfg.memory.local.config_path}")

        async def check_local():
            client = LocalMemoryClient(cfg.memory.local)
            try:
                await client.start()
            finally:
                await client.aclose()

        try:
            asyncio.run(check_local())
            print("  ✓ 本地 Memory 代码、配置与任务索引可加载(未调用模型)")
        except Exception as exc:
            problems.append(f"本地 Memory: {exc}")
    elif cfg.memory.backend == "none":
        print("Memory: 本次禁用")
    else:
        gw = cfg.memory.gateway
        print(f"Memory 网关: {'已配置 ' + gw.url if gw.enabled else '未配置(将发送 memory=not_requested,个性化记忆不会生效)'}")
    mode = "raw" if cfg.memory.backend == "local" else cfg.memory.timestamp_mode
    print(f"  预取={cfg.memory.prefetch} 主动查询的 user_query={cfg.memory.active_query_text} timestamp 模式={mode}")
    print(f"任务数据: {len(tasks)} 个用户" + (f",共 {sum(len(t.subtasks) for t in tasks)} 个 subtask" if tasks else ""))
    if not tasks:
        problems.append(DATA_HINT)
    for p in problems:
        print(f"  ✗ {p}")
    print("检查通过" if not problems else f"有 {len(problems)} 项需要处理")
    return 0 if not problems else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="用 IMAct harness 跑 VitaBench 2.0 personalization")
    p.add_argument("--config", required=True, help="外层配置 YAML(见 config.vitabench.example.yaml)")
    p.add_argument("--models", help="覆盖运行配置中的 models_file")
    p.add_argument("--action-model", help="覆盖 models.yaml 的 roles.action,填模型别名")
    p.add_argument("--user-llm", "--user-model", dest="user_llm", help="覆盖 roles.user,填模型别名")
    p.add_argument("--evaluator-llm", "--judge-model", dest="evaluator_llm", help="覆盖 roles.judge,填模型别名")
    memory = p.add_mutually_exclusive_group()
    memory.add_argument("--no-memory", action="store_true", help="本次禁用 Memory")
    memory.add_argument("--memory-url", help="本次启用 Memory 并使用指定地址")
    memory.add_argument("--local-memory", metavar="HARNESS_CONFIG", help="本次使用本地 Memory harness 配置 JSON")
    p.add_argument("--task-ids", nargs="*", default=None)
    p.add_argument("--num-tasks", type=int, default=None)
    p.add_argument("--max-subtasks", type=int, default=None, help="每个用户只跑前 N 个 subtask(冒烟用;评测汇总只覆盖这些)")
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=50, help="每个 subtask 的对话步数上限")
    p.add_argument("--max-task-retries", type=int, default=3, help="任务级重试(VitaBench 约定为 3)")
    p.add_argument("--max-concurrency", type=int, default=1)
    p.add_argument("--language", default="chinese")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="结果目录;缺省为 results/<时间戳>")
    p.add_argument("--run-id", default=None)
    p.add_argument("--rerun", action="store_true", help="忽略已有结果文件,重新跑")
    p.add_argument("--check", action="store_true", help="只检查配置 / 密钥 / 数据,不发请求")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    args.out = args.out or f"results/{time.strftime('%Y%m%d-%H%M%S')}"
    cfg = load_outer_config(args.config, models_file=args.models)
    if not (args.models or cfg.models_file):
        raise ValueError("需要统一模型配置:在运行配置里设置 models_file,或用 --models 指定 models.yaml")
    if cfg.model_call.params:
        raise ValueError("统一模型配置模式下请把 model_call.params 移入 models.yaml")
    catalog = load_catalog(args.models or cfg.models_file)
    for role, override in (("action", args.action_model), ("user", args.user_llm), ("judge", args.evaluator_llm)):
        alias, _ = catalog.selected(role, override)
        setattr(catalog.roles, role, alias)
    cfg.model = catalog.selected("action")[1]
    print(json.dumps(catalog.snapshot(), ensure_ascii=False, indent=2))
    # 检查当前选中角色,未选中的模型无需凭据。任何请求前失败,避免跑半途才发现缺密钥。
    for alias in set(catalog.roles.model_dump().values()):
        create_provider(catalog.models[alias])
    apply_memory_overrides(cfg, no_memory=args.no_memory, memory_url=args.memory_url, local_memory=args.local_memory)

    cfg.exec.db_path = str(Path(args.out) / "exec.sqlite3")  # 执行层的完整轨迹(attempts、证据、状态)跟结果放一起
    tasks = get_tasks(task_ids=args.task_ids, num_tasks=args.num_tasks, language=args.language)
    if args.max_subtasks:
        for t in tasks:
            t.subtasks = t.subtasks[: args.max_subtasks]
    if args.check:
        return check(cfg, args, tasks, catalog)
    if not tasks:
        print(DATA_HINT, file=sys.stderr)
        return 2
    Path(args.out).mkdir(parents=True, exist_ok=True)
    snapshot = Path(args.out) / f"models.resolved.{time.time_ns()}.json"
    snapshot.write_text(json.dumps(catalog.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
    file_handler = logging.FileHandler(Path(args.out) / "run.log", encoding="utf-8")  # 含 Memory 的请求 curl 与返回内容
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    policy = cfg.retry.model_call.to_policy()
    bridge = AsyncBridge("imact-models")
    user = LLMCaller(catalog.selected("user")[1], bridge, policy=policy)
    judge = LLMCaller(catalog.selected("judge")[1], bridge, policy=policy)
    rt = None
    try:
        rt = build_runtime(cfg)
        summary = run_all(args, cfg, rt, tasks, simulate_fn=partial(simulate, user_llm=user, judge=judge))
    finally:
        if rt:
            rt.close()
        user.close()
        judge.close()
        bridge.close()
    print("\n" + render_tables(summary))
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
