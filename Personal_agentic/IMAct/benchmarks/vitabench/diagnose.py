"""真实链路诊断与可选裁判评分;也可对已保存的真实交互单独补评分。"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import shutil
import sqlite3
import time
import traceback
from pathlib import Path

from ...exec_layer import build_runtime, load_outer_config
from ...exec_layer.bridge import AsyncBridge
from ...exec_layer.config import apply_memory_overrides
from ...harness import LLMCaller
from ...models.catalog import load_catalog
from ...models.config import create_provider
from ...trace import TraceRecorder, trace_scope
from .personalization import COMPLETION_GUARD, PersonalizationRunner
from .tasks import get_tasks, tasks_path
from .trace_report import dump, export_trace


def jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


def select_cases(tasks, subtask_ids):
    lookup = {s.subtask_id: (t, s, i) for t in tasks for i, s in enumerate(t.subtasks)}
    missing = set(subtask_ids) - lookup.keys()
    if missing:
        raise ValueError(f"Unknown subtask ids: {sorted(missing)}")
    if len(set(subtask_ids)) != len(subtask_ids):
        raise ValueError("subtask ids must be unique")
    return [lookup[sid] for sid in subtask_ids]


def run_case(task, subtask, original_index, *, cfg, catalog, out, max_steps, seed, user_completion_guard=False, with_reward=False):
    out.mkdir(parents=True, exist_ok=False)
    trace = TraceRecorder(out / "trace.jsonl")
    cfg = cfg.model_copy(deep=True)
    cfg.exec.db_path = str(out / "exec.sqlite3")
    case = {"subtask_id": subtask.subtask_id, "domain": subtask.domain, "user_id": task.user_id,
            "instruction": subtask.instruction, "original_subtask_index": original_index,
            "selection": "isolated original subtask; Memory uses its original fixed snapshot",
            "evaluation": "pending" if with_reward else "skipped", "reward": None,
            "status": "running", "models": catalog.snapshot(), "judge_called": False,
            "memory": cfg.memory.model_dump(), "max_steps": max_steps, "seed": seed,
            "user_completion_guard": COMPLETION_GUARD if user_completion_guard else None,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    dump(out / "case.json", case)
    rt = user_llm = bridge = None
    started = time.perf_counter()
    try:
        rt = build_runtime(cfg, trace=trace)
        bridge = AsyncBridge("imact-user")
        user_llm = LLMCaller(catalog.selected("user")[1], bridge)
        isolated_task = task.model_copy(update={"subtasks": [subtask]})
        runner = PersonalizationRunner(isolated_task, rt=rt, user_llm=user_llm, judge=None, run_id=out.parent.name,
                                       max_steps=max_steps, language="chinese", seed=seed, completion_guard=user_completion_guard)

        def started_case(spec):
            with trace_scope(query_id=subtask.subtask_id, domain=subtask.domain):
                trace.emit("case_start", instruction=subtask.instruction, original_index=original_index,
                           tool_names=[t["function"]["name"] for t in spec.tools])

        with trace_scope(query_id=subtask.subtask_id, domain=subtask.domain):
            result = runner.run_subtask(subtask, original_index, on_prepared=started_case)
            simulation = jsonable(result.as_dict())
            simulation.update(evaluation="skipped", reward=None, reward_info=None)
            dump(out / "simulation.json", simulation)
            case.update(status="conversation_finished", termination_reason=simulation["termination_reason"])
            if with_reward:
                from .reward import grade_simulation, attach_grade, export_grade

                grade = grade_simulation(task, subtask, simulation, catalog=catalog, out=out, trace=trace)
                attach_grade(case, simulation, grade)
                dump(out / "simulation.json", simulation)
                export_grade(out)
            trace.emit("case_end", termination_reason=simulation["termination_reason"], evaluation=case["evaluation"], reward=case["reward"])
    except Exception as exc:
        case.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        trace.emit("case_error", error=case["error"], traceback=traceback.format_exc())
        logging.exception("case failed: %s", subtask.subtask_id)
    finally:
        if rt:
            rt.close()
        if user_llm:
            user_llm.close()
        if bridge:
            bridge.close()
        case["duration_s"] = round(time.perf_counter() - started, 2)
        dump(out / "case.json", case)
        summary = export_trace(out)
        from .episode_report import export_episode
        export_episode(out, refresh=False)
    return summary


def grade_existing_case(source, task, subtask, *, catalog, out):
    """只评分冻结的历史轨迹,完整保留源目录和旧模型/记忆证据。"""
    from .reward import grade_simulation, attach_grade, export_grade
    from .episode_report import export_episode

    source = Path(source).resolve()
    case = json.loads((source / "case.json").read_text())
    simulation = json.loads((source / "simulation.json").read_text())
    if (case["user_id"], case["subtask_id"], simulation["subtask_id"]) != (task.user_id, subtask.subtask_id, subtask.subtask_id):
        raise ValueError("保存的交互与原始用户/题号不一致")
    manifest_path = source.parent / "manifest.json"
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text()).get("dataset_sha256")
        if expected and expected != hashlib.sha256(tasks_path().read_bytes()).hexdigest():
            raise ValueError("当前数据集与轨迹来源哈希不同,不能直接补评分")
    out.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for name in ("simulation.json", "trace.jsonl", "case.json", "memory_result.json", "memory_request.json", "memory_context.txt", "conversation.txt"):
        path = source / name
        if path.exists():
            source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            shutil.copyfile(path, out / name)
    if (source / "exec.sqlite3").exists():
        from contextlib import closing

        with closing(sqlite3.connect(f"file:{source / 'exec.sqlite3'}?mode=ro", uri=True)) as src, closing(sqlite3.connect(out / "exec.sqlite3")) as dest:
            src.backup(dest)
    trace = TraceRecorder(out / "trace.jsonl")
    case.update(source_case=str(source), source_hashes=source_hashes, evaluation="pending", reward=None)
    dump(out / "case.json", case)
    result = grade_simulation(task, subtask, simulation, catalog=catalog, out=out, trace=trace)
    attach_grade(case, simulation, result)
    dump(out / "case.json", case)
    dump(out / "simulation.json", simulation)
    export_grade(out)
    summary = export_trace(out)
    export_episode(out, refresh=False)
    return summary


def write_index(root):
    rows = []
    for p in sorted(root.glob("*/audit.json")):
        audit = json.loads(p.read_text())
        case = json.loads((p.parent / "case.json").read_text())
        rows.append({"subtask_id": audit["subtask_id"], "domain": audit["domain"], "instruction": case["instruction"],
                     "status": audit["status"], "termination": audit["termination_reason"],
                     "action_outputs": audit["model_outputs"], "attempts": audit["attempts"],
                     "failed_attempts": len(audit["validation_errors"]), "tool_calls": audit["tool_calls"],
                     "memory_calls": audit["memory_calls"], "duration_s": case["duration_s"],
                     "trace": f"{p.parent.name}/episode.html", "reward": audit.get("reward"),
                     "evaluation": audit.get("evaluation"), "rubric_pass_rate": case.get("rubric_pass_rate")})
    dump(root / "summary.json", {"cases": rows})
    table = []
    for r in rows:
        vals = [r["domain"], f'<a href="{html.escape(r["trace"])}">{html.escape(r["subtask_id"])}</a>',
                html.escape(r["instruction"]), html.escape(str(r["termination"] or r["status"])),
                f'{r["action_outputs"]}/{r["attempts"]}', str(r["failed_attempts"]), str(r["tool_calls"]), str(r["memory_calls"]),
                (f'<a href="{Path(r["trace"]).parent}/grading.html">{r["reward"]}</a>' if r["evaluation"] in {"completed", "rule_only", "error"} else '未评分'),
                str(r["rubric_pass_rate"])]
        table.append('<tr>'+''.join('<td>'+str(v)+'</td>' for v in vals)+'</tr>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>跨场景链路诊断</title><style>body{font-family:-apple-system,sans-serif;background:#f0f5fa;color:#193552;margin:40px auto;max-width:1400px;padding:20px}h1{font-size:30px}p{line-height:1.8;color:#4f6881}table{width:100%;border-collapse:collapse;background:white;border-radius:12px}th,td{padding:16px;text-align:left;border-bottom:1px solid #dbe4ef;vertical-align:top}th{background:#173c64;color:white}a{color:#2165ad}.box{overflow:auto}</style><h1>VitaBench 跨场景 · 全链路诊断</h1><p>每行是一个原始 subtask 的独立真实运行。点击样本查看：用户问题 → Memory 请求/返回 → Harness 完整输入 → 每次 Action 原始输出 → 校验及工具/用户反馈。<br>评分模式见各行；点击 reward 查看原始 rubric、裁判依据和窗口输入输出。user_stop 只代表对话结束，业务完成与 rubric 得分分别核对。</p><div class="box"><table><thead><tr><th>领域</th><th>样本 / 全链路轨迹</th><th>用户问题</th><th>终止</th><th>输出/尝试</th><th>失败尝试</th><th>工具调用</th><th>Memory 请求</th><th>Reward</th><th>Rubric 通过率</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table></div><p><a href="summary.json">汇总 JSON</a> · <a href="manifest.json">样本与运行来源</a></p></html>'''
    (root / "index.html").write_text(page, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.vitabench.yaml")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--subtask-ids", nargs="+")
    selection.add_argument("--grade-existing", metavar="CASE_DIR", help="仅补评已有 case 的 simulation.json,不重跑交互")
    parser.add_argument("--with-reward", action="store_true", help="交互结束后调用 roles.judge 进行原始 trajectory 评分")
    parser.add_argument("--judge-model", help="本次覆盖 models.yaml 的 roles.judge")
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--memory-url", help="覆盖配置,使用 MPS 网关")
    memory.add_argument("--local-memory", metavar="HARNESS_CONFIG", help="覆盖配置,使用本地 Memory")
    memory.add_argument("--no-memory", action="store_true", help="本次禁用 Memory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--user-completion-guard", action="store_true",
                        help="诊断模式:向用户模拟器说明确认方案不等于执行完成;原始响应仍原样记录")
    parser.add_argument("--resume", action="store_true", help="只跳过已有完整 audit 的样本,不覆盖任何轨迹")
    parser.add_argument("--check", action="store_true", help="只校验配置与样本,不发送请求")
    args = parser.parse_args(argv)
    cfg = load_outer_config(args.config)
    if not cfg.models_file:
        parser.error("diagnose requires models_file")
    catalog = load_catalog(cfg.models_file)
    if args.judge_model:
        catalog.roles.judge = catalog.selected("judge", args.judge_model)[0]
    if args.grade_existing:
        saved = json.loads((Path(args.grade_existing) / "case.json").read_text())
        args.subtask_ids = [saved["subtask_id"]]
    cfg.model = catalog.selected("action")[1]
    apply_memory_overrides(cfg, no_memory=args.no_memory, memory_url=args.memory_url, local_memory=args.local_memory)
    roles = ["judge"] if args.grade_existing else ["action", "user"] + (["judge"] if args.with_reward else [])
    for role in roles:
        create_provider(catalog.selected(role)[1])
    tasks = get_tasks(language="chinese")
    cases = select_cases(tasks, args.subtask_ids)
    if args.check:
        print(json.dumps({"models": catalog.snapshot(), "cases": [dict(subtask_id=s.subtask_id, domain=s.domain, instruction=s.instruction) for _, s, _ in cases], "judge_called": False}, ensure_ascii=False, indent=2))
        return 0
    root = Path(args.out)
    if root.exists() and any(root.iterdir()) and not args.resume:
        parser.error("output directory is not empty; use a new directory or --resume")
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.FileHandler(root / "run.log", encoding="utf-8"), logging.StreamHandler()])
    manifest = {"models": catalog.snapshot(), "requested_subtasks": args.subtask_ids, "memory": cfg.memory.model_dump(),
                "dataset": str(tasks_path()), "dataset_sha256": hashlib.sha256(tasks_path().read_bytes()).hexdigest(),
                "max_steps": args.max_steps, "seed": args.seed,
                "evaluation": "trajectory" if args.with_reward or args.grade_existing else "skipped",
                "grade_existing": args.grade_existing,
                "user_completion_guard": args.user_completion_guard,
                "execution": "sequential, one model request at a time"}
    dump(root / f"manifest.{time.time_ns()}.json", manifest)
    if not (root / "manifest.json").exists():
        dump(root / "manifest.json", manifest)
    for task, subtask, idx in cases:
        out = root / subtask.subtask_id
        if (out / "audit.json").exists() and args.resume:
            continue
        print(f"START {subtask.subtask_id}: {subtask.instruction}", flush=True)
        if args.grade_existing:
            summary = grade_existing_case(args.grade_existing, task, subtask, catalog=catalog, out=out)
        else:
            summary = run_case(task, subtask, idx, cfg=cfg, catalog=catalog, out=out, max_steps=args.max_steps, seed=args.seed,
                               user_completion_guard=args.user_completion_guard, with_reward=args.with_reward)
        write_index(root)
        print(f"DONE {subtask.subtask_id}: {summary['status']} / {summary['termination_reason']} / outputs={summary['model_outputs']} errors={len(summary['validation_errors'])}", flush=True)
    print(f"全链路轨迹: {root / 'index.html'}", flush=True)
    result = json.loads((root / "summary.json").read_text())
    return 1 if any(r["status"] == "failed" or r["evaluation"] == "error" for r in result["cases"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
