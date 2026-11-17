"""批量运行的通用部分:任务级重试、断点续跑、并发、汇总。与 benchmark 无关。

一个 *job* 是 (task, trial)。结果文件 ``<out>/<task_id>.trial<N>.json`` 已存在就跳过(``rerun`` 才重跑);
单个 job 耗尽重试后失败只记录,不拖垮整批。
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger("imact.batch")


def run_with_task_retries(fn: Callable[[], Any], *, retries: int, sleep=time.sleep):
    """整个任务失败就重来(每次调用方都会新建 agent,即全新的会话)。"""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            logger.exception("task failed on attempt %d/%d", attempt + 1, retries + 1)
            if attempt < retries:
                sleep(min(30, 2**attempt))
    assert last is not None
    raise last


def result_path(out: Path, task_id: str, trial: int) -> Path:
    # task id 可能含 [ ] | : 等字符(如 telecom),文件名里不安全
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(task_id))
    if safe != task_id:
        safe += "-" + __import__("hashlib").sha256(str(task_id).encode()).hexdigest()[:8]
    return out / f"{safe}.trial{trial}.json"


def trial_seeds(seed: int | None, num_trials: int) -> list[int | None]:
    if seed is None:
        return [None] * num_trials
    return [random.Random(seed + t).randint(0, 1_000_000) for t in range(num_trials)]


def record_row(record: dict, row_fields: Callable[[dict], dict[str, Any]] | None = None) -> tuple[Any, bool, dict[str, Any]]:
    """一份落盘结果 → (reward, evaluation_error, 额外字段)。"""
    info = record.get("reward_info") or None
    reward = info.get("reward") if isinstance(info, dict) else None
    evaluation_error = bool(isinstance(info, dict) and (info.get("info") or {}).get("evaluation_error"))
    return reward, evaluation_error, row_fields(record) if row_fields else {}


_TRIAL_RE = re.compile(r"\.trial(\d+)\.json$")


def summary_from_results(out: Path, *, row_fields: Callable[[dict], dict[str, Any]] | None = None,
                         previous: dict[str, Any] | None = None) -> dict[str, Any]:
    """只读目录里的结果文件(``*.trialN.json``)重新汇总,形状同 ``run_jobs`` 的返回。不调用任何模型。

    结果文件里没有"失败"的记录,所以失败的 job 取自 ``previous``(旧的 summary.json),但已经有结果文件的会被剔除
    (后来重跑成功了)。读不出的文件跳过并记日志。"""
    results = []
    for path in sorted(Path(out).glob("*.trial*.json")):
        m = _TRIAL_RE.search(path.name)
        if not m:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            logger.warning("结果文件读不出,跳过: %s (%s)", path, exc)
            continue
        reward, evaluation_error, extra = record_row(record, row_fields)
        results.append({"task_id": record.get("task_id") or path.name[: m.start()], "trial": int(m.group(1)), "reward": reward,
                        "evaluation_error": evaluation_error, **extra})
    done = {(str(r["task_id"]), r["trial"]) for r in results}
    failed = [f for f in (previous or {}).get("failed") or [] if (str(f.get("task_id")), f.get("trial", 0)) not in done]
    rewards = [r["reward"] for r in results if r["reward"] is not None]
    extra_keys = {k: v for k, v in (previous or {}).items() if k not in ("results", "failed", "skipped", "avg_reward", "num_evaluation_errors")}
    return {**extra_keys, "results": results, "failed": failed, "skipped": [],
            "avg_reward": sum(rewards) / len(rewards) if rewards else None,
            "num_evaluation_errors": sum(1 for r in results if r["evaluation_error"])}


def run_jobs(
    tasks: Iterable[Any],
    simulate: Callable[..., Any],
    *,
    out: Path,
    run_id: str,
    num_trials: int = 1,
    seed: int | None = None,
    max_task_retries: int = 3,
    max_concurrency: int = 1,
    rerun: bool = False,
    summary_extra: dict[str, Any] | None = None,
    row_fields: Callable[[Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """``simulate(task, run_id=..., seed=...)`` 返回带 ``model_dump`` 与 ``reward_info`` 的结果。
    ``row_fields(record)`` 可选:``record`` 是落盘的结果(``model_dump(mode="json")`` 的 dict),返回要并入汇总里这一行的额外字段。

    已有结果文件的 job 不重跑(``rerun`` 才重跑),但它的结果照样读进汇总(行上标 ``resumed: True``,另记入 ``skipped``),
    所以断点续跑后的汇总覆盖全部 job,而不只是这次新跑的。读不出的旧文件当作没有结果,重新跑。"""
    out.mkdir(parents=True, exist_ok=True)
    seeds = trial_seeds(seed, num_trials)
    jobs = [(task, trial) for task in tasks for trial in range(num_trials)]
    summary: dict[str, Any] = {"run_id": run_id, **(summary_extra or {}), "results": [], "failed": [], "skipped": []}

    def row(record: dict) -> tuple:
        return record_row(record, row_fields)

    def one(task, trial):
        path = result_path(out, task.id, trial)
        if path.exists() and not rerun:
            try:
                return ("skipped", row(json.loads(path.read_text(encoding="utf-8"))))
            except (ValueError, OSError) as exc:
                logger.warning("已有结果文件读不出,重新跑: %s (%s)", path, exc)
        sim = run_with_task_retries(lambda: simulate(task, run_id=f"{run_id}-t{trial}", seed=seeds[trial]), retries=max_task_retries)
        record = sim.model_dump(mode="json")
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return ("done", row(record))

    with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
        futures = {pool.submit(one, t, tr): (t.id, tr) for t, tr in jobs}
        for fut in as_completed(futures):
            tid, trial = futures[fut]
            try:
                status, (reward, evaluation_error, extra) = fut.result()
            except Exception as exc:  # noqa: BLE001 - 单个任务耗尽重试后失败,不拖垮整批
                summary["failed"].append({"task_id": tid, "trial": trial, "error": f"{type(exc).__name__}: {exc}"})
                continue
            entry = {"task_id": tid, "trial": trial, "reward": reward, "evaluation_error": evaluation_error, **extra}
            if status == "skipped":
                entry["resumed"] = True
                summary["skipped"].append({"task_id": tid, "trial": trial})
            summary["results"].append(entry)
    rewards = [r["reward"] for r in summary["results"] if r["reward"] is not None]
    summary["avg_reward"] = sum(rewards) / len(rewards) if rewards else None
    summary["num_evaluation_errors"] = sum(1 for r in summary["results"] if r["evaluation_error"])
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
