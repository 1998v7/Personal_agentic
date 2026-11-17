"""用 IMAct harness 跑 MCP-Universe(web_search / location_navigation / repository_management / financial_analysis / browser_automation)的入口。

    python -X utf8 -m IMAct.benchmarks.mcp_universe.run --models models.yaml --domain all --check
    python -X utf8 -m IMAct.benchmarks.mcp_universe.run --models models.yaml --domain financial_analysis --num-tasks 3 --out results/mcpu-smoke

被测对象是 agent 模型(ReAct 文本协议,prompt 与上游一致)。agent 与评分都经 MCP 网关访问真实服务:
``MCP_GATEWAY_ADDRESS`` / ``MCP_GATEWAY_TOKEN`` 以及各服务的密钥在 ``.env``(或 ``--env-file``)里。
会在 GitHub / Notion 里创建并事后删除资源的 task(共 43 个)默认也跑;只想跑只读的 169 个,加 ``--skip-side-effects``。

指标:SR(成功率:一个 task 的全部评估都通过)、AE(评估通过率:每个 task 通过的评估占比,再取平均)、
AS(成功 task 的平均步数)、平均 ReAct 轮数(所有 task)。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from ...evaluation.mcp_universe import Context, EvaluatorConfig
from ...exec_layer.bridge import AsyncBridge
from ...harness import LLMCaller
from ...harness.batch import run_jobs, summary_from_results
from ...harness.mcp import GatewayConfig, McpError, McpToolset
from ...models.catalog import load_catalog
from ...models.config import create_provider
from ...retry import RetryPolicy
from ...trace import TraceRecorder
from . import tasks as T
from .runner import McpUniverseRunner, needs_judge
from .tables import render as render_tables

logger = logging.getLogger("imact.mcp_universe")
LOCAL_EVAL_PREFIXES = {"yfinance.": "yfinance", "playwright.check_": "playwright", "weather.": "weather"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="用 IMAct harness 跑 MCP-Universe(ReAct agent,真实 MCP 服务)")
    p.add_argument("--models", required=True, help="统一模型配置 models.yaml")
    p.add_argument("--domain", nargs="+", required=True, choices=(*T.domain_names(), "all"), help="可多选;all = 五个领域")
    p.add_argument("--agent-model", help="覆盖 roles.action,填模型别名")
    p.add_argument("--judge-model", "--evaluator-llm", dest="judge_model", help="覆盖 roles.judge(web_search 的答案由它判)")
    p.add_argument("--task-ids", nargs="*", default=None, help="按 task 名选(如 info_search_task_0001);只给一个 domain 时才有意义")
    p.add_argument("--num-tasks", type=int, default=None, help="每个领域只取前 N 个 task(冒烟用)")
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--max-task-retries", type=int, default=3, help="task 整体失败(连不上网关、模型请求持续失败)后的重试次数")
    p.add_argument("--max-concurrency", type=int, default=1, help="同一个领域内同时跑几个 task;会创建资源的 task 始终一次一个")
    p.add_argument("--model-retries", type=int, default=3, help="单次模型请求的重试次数")
    p.add_argument("--call-timeout", type=float, default=60.0, help="单次工具调用的超时秒数")
    p.add_argument("--eval-timeout", type=float, default=900.0, help="单条评估的超时秒数")
    p.add_argument("--skip-side-effects", action="store_true",
                   help="不跑会在真实 GitHub / Notion 里创建并事后删除资源的 task(共 43 个),只跑只读的 169 个。"
                        "不加这个开关时它们会跑:GitHub 的删除只作用于 GITHUB_PERSONAL_ACCOUNT_NAME 账号,务必用专用测试账号")
    p.add_argument("--env-file", action="append", default=[], help="额外加载的 .env 文件(可重复);不覆盖已有的环境变量")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="结果目录;缺省为 results/mcpu-<时间戳>")
    p.add_argument("--run-id", default=None)
    p.add_argument("--rerun", action="store_true", help="忽略已有结果文件,重新跑")
    p.add_argument("--check", action="store_true", help="只检查配置 / 密钥 / 网关连通,不发模型请求、不创建任何资源")
    return p


def select_models(catalog, args) -> None:
    for role, override in (("action", args.agent_model), ("judge", args.judge_model)):
        alias, _ = catalog.selected(role, override)
        setattr(catalog.roles, role, alias)


def selected_domains(args) -> list[str]:
    return T.domain_names() if "all" in args.domain else list(dict.fromkeys(args.domain))


def select_tasks(args, domains: list[str], env) -> list[tuple[str, list[T.McpTask]]]:
    """(领域, task 列表)。"""
    if args.task_ids is not None and len(domains) != 1:
        raise ValueError("--task-ids 只能配合单个 --domain 使用")
    return [(d, T.select(T.load_tasks(d, env), args.task_ids, args.num_tasks)) for d in domains]


def unresolved_variables(task: T.McpTask, env=None) -> list[str]:
    """问题或评估里没被环境变量替换掉的 ``{{ VAR }}``(评估按运行时的方式先渲染)。"""
    context = Context(env)
    rendered = []
    for config in task.evaluators:
        cfg = EvaluatorConfig.model_validate(config)
        cfg.set_environ_variables(context)
        rendered.append(cfg.model_dump())
    text = task.question + json.dumps(rendered, ensure_ascii=False)
    return sorted(set(re.findall(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", text)))


def local_evaluation_needs(tasks: list[T.McpTask], env) -> dict[str, int]:
    """没配远程评估时,哪些评估要在本机联网执行:{依赖: 条数}。"""
    if env.get("EVAL_REMOTE_URL") and env.get("EVAL_REMOTE_PREFIXES"):
        prefixes = tuple(p.strip() for p in env["EVAL_REMOTE_PREFIXES"].split(",") if p.strip())
    else:
        prefixes = ()
    needs: dict[str, int] = {}
    for t in tasks:
        for e in t.evaluators:
            op = e.get("op", "")
            for prefix, dep in LOCAL_EVAL_PREFIXES.items():
                if op.startswith(prefix) and not op.startswith(prefixes or ("\0",)):
                    needs[dep] = needs.get(dep, 0) + 1
    return needs


# 能列出工具不代表服务真的能干活(例如 playwright 服务在,浏览器却没装)。这些 server 在 --check 时做一次无副作用的真实调用。
SERVER_PROBES = {
    "playwright": [("browser_navigate", {"url": "about:blank"}), ("browser_close", {})],
}


def probe_server(server: str, toolset: McpToolset) -> str | None:
    """对 ``SERVER_PROBES`` 里的 server 试调用;失败时返回原因,正常返回 None。"""
    for tool, arguments in SERVER_PROBES.get(server, []):
        result = toolset.call(server, tool, arguments)
        text = result.first_text or ""
        if result.is_error or text.lstrip().startswith("### Error"):
            return f"{tool} → {' '.join(text.split())[:300]}"
    return None


def check(args, catalog, groups, env) -> int:
    problems: list[str] = []
    all_tasks = [t for _, ts in groups for t in ts]
    for role in ["action"] + (["judge"] if needs_judge(all_tasks) else []):
        alias, model = catalog.selected(role)
        try:
            create_provider(model)
            print(f"{role} 角色: {alias} ({model.model}) ✓")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{role} 角色 {alias}: {exc}")
    if not needs_judge(all_tasks):
        print("裁判: 这批 task 不需要")
    try:
        gateway = GatewayConfig.from_env(env)
        print(f"网关: {gateway.address}  {'(带 token)' if gateway.token else '(无 token)'}")
    except McpError as exc:
        problems.append(str(exc))
        gateway = None
    side_effect = [t for t in all_tasks if t.has_side_effects]
    for domain, tasks in groups:
        servers = sorted({s["name"] for t in tasks for s in t.mcp_servers})
        print(f"  {domain:<24}{len(tasks):>4} 个 task   服务: {', '.join(servers)}")
    print(f"共 {len(all_tasks)} 个 task × {args.num_trials} 次 trial;其中会创建资源的 {len(side_effect)} 个")
    if side_effect:
        account = env.get("GITHUB_PERSONAL_ACCOUNT_NAME")
        print(f"  ⚠ 这些 task 会在 GitHub 账号 {account or '(GITHUB_PERSONAL_ACCOUNT_NAME 未设置!)'} 下创建并删除仓库,在 Notion 里创建并归档页面")
        github_tasks = [t for t in side_effect if any(c.get("server") == "github" for c in t.cleanups)]
        notion_tasks = [t for t in side_effect if any(c.get("server") == "notion" for c in t.cleanups)]
        if github_tasks:
            if not env.get("GITHUB_PERSONAL_ACCOUNT_NAME"):
                problems.append("环境变量 GITHUB_PERSONAL_ACCOUNT_NAME 未设置(仓库属于哪个账号、评估查询谁的仓库都靠它)")
            if not env.get("GITHUB_PERSONAL_ACCESS_TOKEN"):
                print(f"  ⚠ 本机没有 GITHUB_PERSONAL_ACCESS_TOKEN:{len(github_tasks)} 个 GitHub task 创建的仓库**不会被自动删除**"
                      "(网关的 GitHub 服务没有删除仓库的工具),跑完需要手动删;同名仓库已存在时 task 会因此失败")
        if notion_tasks:
            if not env.get("NOTION_ROOT_PAGE"):
                problems.append("环境变量 NOTION_ROOT_PAGE 未设置(Notion task 要在这个父页面下建页;页面需已共享给 Notion 集成)")
            if not env.get("NOTION_API_KEY"):
                print("  ℹ 本机没有 NOTION_API_KEY:Notion 页面将通过网关的 Notion 服务归档(API-patch-page)")
    unresolved = {t.id: v for t in all_tasks if (v := unresolved_variables(t, env))}
    if unresolved:
        missing = sorted({x for v in unresolved.values() for x in v})
        problems.append(f"{len(unresolved)} 个 task 里有未替换的环境变量: {', '.join(missing)}")
    for dep, count in local_evaluation_needs(all_tasks, env).items():
        print(f"  ⚠ {count} 条评估需要在本机联网执行({dep});没配 EVAL_REMOTE_URL / EVAL_REMOTE_PREFIXES 时,{dep} 要能直接访问外网"
              + (",且已执行过 `playwright install chromium`" if dep == "playwright" else ""))
    if gateway is not None:
        bridge = AsyncBridge("imact-mcp-check")
        try:
            for server in sorted({s["name"] for t in all_tasks for s in t.mcp_servers}):
                started = time.time()
                try:
                    with McpToolset(gateway, [{"name": server}], bridge, connect_retries=2) as toolset:
                        broken = probe_server(server, toolset)
                        if broken:
                            problems.append(f"网关服务 {server}: 能连上,但实际调用失败: {broken}")
                        else:
                            print(f"  网关服务 {server:<14}✓ {len(toolset.tools[server]):>3} 个工具  {time.time() - started:.1f}s"
                                  + ("  (已试调用)" if server in SERVER_PROBES else ""))
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"网关服务 {server}: {exc}")
        finally:
            bridge.close()
    for p in problems:
        print(f"  ✗ {p}")
    print("检查通过" if not problems else f"有 {len(problems)} 项需要处理")
    return 0 if not problems else 2


def row_fields(record: dict) -> dict:
    info = (record.get("reward_info") or {}).get("info") or {}
    return {"evaluator_score": info.get("evaluator_score"), "iterations": info.get("iterations"), "llm_calls": info.get("llm_calls"), "answered": info.get("answered"),
            "evaluations_passed": info.get("evaluations_passed"), "evaluations_total": info.get("evaluations_total")}


def summarize(domains_summary: dict[str, dict], *, run_id: str, agent_model: str, skipped_side_effects: list[str]) -> dict:
    """总汇总。SR / AE 是百分数(0-100,与论文同口径),按所有已评分的 task 平均;AS 只在成功的 task 上平均(论文口径);各领域的明细在 ``domains`` 里。"""
    domains, rows = {}, []
    for name, s in domains_summary.items():
        r = s["results"]
        rows += r
        domains[name] = {
            "completed": len(r), "failed": len(s["failed"]), "skipped": len(s["skipped"]), "evaluation_errors": s.get("num_evaluation_errors", 0),
            "SR": _pct(s["avg_reward"]), "AE": _pct(_mean([x["evaluator_score"] for x in r])), **_steps(r), "avg_iterations": _mean([x["iterations"] for x in r]),
            "failed_tasks": s["failed"]}
    return {"bench": "mcp_universe", "run_id": run_id, "agent_model": agent_model, "completed": len(rows), "failed": sum(d["failed"] for d in domains.values()),
            "skipped": sum(d["skipped"] for d in domains.values()), "evaluation_errors": sum(d["evaluation_errors"] for d in domains.values()),
            "SR": _pct(_mean([x["reward"] for x in rows])), "AE": _pct(_mean([x["evaluator_score"] for x in rows])), **_steps(rows),
            "avg_iterations": _mean([x["iterations"] for x in rows]), "skipped_side_effects": skipped_side_effects, "domains": domains}


def _steps(rows: list[dict]) -> dict:
    """AS:成功 task(reward == 1)的平均步数。``AS`` 按 ReAct 轮数(论文里 agent 的"步");
    ``AS_llm_calls`` 按模型调用次数(含工具回执摘要),与上游 report.py 的 LLM Call Count 同口径。没有成功的 task 时为 None。"""
    ok = [x for x in rows if x.get("reward") == 1.0]
    return {"num_successful": len(ok), "AS": _mean([x.get("iterations") for x in ok]), "AS_llm_calls": _mean([x.get("llm_calls") for x in ok])}


def _pct(value: float | None) -> float | None:
    """0-1 → 0-100,与论文表格同一口径(SR / AE 都是百分数)。"""
    return None if value is None else 100.0 * value


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None



def rebuild(out: Path) -> dict:
    """只读 ``out`` 里的结果文件重新汇总(不调用模型)。``python -m IMAct.report`` 用它把旧目录打印成表。"""
    old = _read_json(out / "summary.json")
    domains_summary = {}
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        if any(d.glob("*.trial*.json")):
            domains_summary[d.name] = summary_from_results(d, row_fields=row_fields, previous=_read_json(d / "summary.json"))
    return summarize(domains_summary, run_id=old.get("run_id", out.name), agent_model=old.get("agent_model", "-"),
                     skipped_side_effects=old.get("skipped_side_effects") or [])


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    load_dotenv()
    for path in args.env_file:
        if not Path(path).is_file():
            print(f"参数错误: --env-file 找不到 {path}", file=sys.stderr)
            return 2
        load_dotenv(path)
    env = dict(os.environ)
    out = Path(args.out or f"results/mcpu-{time.strftime('%Y%m%d-%H%M%S')}")
    catalog = load_catalog(args.models)
    select_models(catalog, args)
    print(json.dumps(catalog.snapshot(), ensure_ascii=False, indent=2))
    try:
        groups = select_tasks(args, selected_domains(args), env)
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2
    skipped_side_effects: list[str] = []
    if args.skip_side_effects:
        skipped_side_effects = [t.id for _, ts in groups for t in ts if t.has_side_effects]
        groups = [(d, [t for t in ts if not t.has_side_effects]) for d, ts in groups]
    if args.check:
        return check(args, catalog, groups, env)
    if not any(ts for _, ts in groups):
        print("没有选中任何 task(--skip-side-effects 排除了会创建资源的 task)", file=sys.stderr)
        return 2
    all_tasks = [t for _, ts in groups for t in ts]
    judge_needed = needs_judge(all_tasks)
    gateway = GatewayConfig.from_env(env)
    for role in ["action"] + (["judge"] if judge_needed else []):
        create_provider(catalog.selected(role)[1])  # 任何请求之前先确认密钥都在

    out.mkdir(parents=True, exist_ok=True)
    (out / f"models.resolved.{time.time_ns()}.json").write_text(json.dumps(catalog.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
    file_handler = logging.FileHandler(out / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    policy = RetryPolicy(max_retries=args.model_retries, backoff_s=1.0)
    bridge = AsyncBridge("imact-models")
    trace = TraceRecorder(out / "trace.jsonl")
    agent_llm = LLMCaller(catalog.selected("action")[1], bridge, policy=policy, trace=trace, trace_stage="agent")
    judge = LLMCaller(catalog.selected("judge")[1], bridge, policy=policy, trace=trace, trace_stage="judge") if judge_needed else None
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    domains_summary: dict[str, dict] = {}
    try:
        for domain, tasks in groups:
            if not tasks:
                continue
            print(f"\n=== {domain} ({len(tasks)} 个 task) ===", flush=True)
            runner = McpUniverseRunner(T.load_domain(domain), agent_llm, bridge, gateway, judge=judge, env=env, side_effects=not args.skip_side_effects,
                                       eval_timeout=args.eval_timeout, call_timeout=args.call_timeout)
            domains_summary[domain] = run_jobs(
                tasks, lambda task, run_id, seed, runner=runner: runner.run_task(task, run_id=run_id, seed=seed),
                out=out / domain, run_id=run_id, num_trials=args.num_trials, seed=args.seed, max_task_retries=args.max_task_retries,
                max_concurrency=args.max_concurrency, rerun=args.rerun, row_fields=row_fields,
                summary_extra={"bench": "mcp_universe", "domain": domain, "agent_model": agent_llm.cfg.model})
    finally:
        for llm in (agent_llm, judge):
            if llm is not None:
                llm.close()
        bridge.close()
    summary = summarize(domains_summary, run_id=run_id, agent_model=agent_llm.cfg.model, skipped_side_effects=skipped_side_effects)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + render_tables(summary))
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
