"""PersonaMem-v3 的汇总:直接调用上游 ``scripts/aggregate_eval.py`` 的同一套函数(已移植到 pm3/),口径与官方报告一致。

* 每行的 0-100 分:``_accuracy_value``(按 ``task_registry.PRIMARY_METRIC`` 取主指标;status 为 error 等计 0;返回 None 的行不计入);
* 头条:``ALL (micro, row-weighted)``(所有计分行的平均)与 ``ALL-JUDGE``(只含裁判打分的题型);
* 另有每个题型、每个类别(by-class)、每个能力轴(by-axis)的平均,以及主动纠错题的 warn/foil 配对 F1。
* 已被官方弃用的题型(``DROPPED_TASK_TYPES``)在这里排除,与官方 ``_load_rows`` 一致。

这里只额外统计 IMAct 自己的东西:失败(重试用尽、没有分数)的题数,以及 Memory 返回空记忆 / 被截断的题数。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .pm3.evaluation.task_registry import DROPPED_TASK_TYPES, normalize_task_type
from .pm3.scripts.aggregate_eval import _build_token_accuracy_table, _e6_paired_f1


def to_aggregate_row(run: dict) -> dict:
    """一条落盘的运行记录 → 上游汇总函数认的行格式(``results.csv`` 的一行 + 解析后的 ``_metrics``)。"""
    return {"query_id": run["task_id"], "task_type": run["task_type"], "status": run.get("status", "ok"),
            "_metrics": run.get("metrics") or {}, "duration_ms": run.get("duration_ms", 0), "_uid": run.get("user_id")}


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def summarize(runs: Iterable[dict], *, failed: int = 0) -> dict:
    runs = [r for r in runs if normalize_task_type(r.get("task_type", "")) not in DROPPED_TASK_TYPES]
    rows = [to_aggregate_row(r) for r in runs]
    e6 = _e6_paired_f1(rows)
    table = _build_token_accuracy_table(rows, e6)
    by_task, by_class, by_axis, headline = {}, {}, {}, {}
    for t in table:
        name = str(t["task_type"])
        if name.startswith("ALL (micro"):
            headline["ALL"] = _num(t["accuracy_pct"])
        elif name.startswith("ALL-JUDGE"):
            headline["ALL_JUDGE"] = _num(t["accuracy_pct"])
        elif name.startswith("  by-class: "):
            by_class[name.removeprefix("  by-class: ")] = {"n": t["n"], "accuracy_pct": _num(t["accuracy_pct"])}
        elif name.startswith("  by-axis: "):
            by_axis[name.removeprefix("  by-axis: ")] = {"n": t["n"], "accuracy_pct": _num(t["accuracy_pct"])}
        else:
            by_task[name] = {"n": t["n"], "accuracy_pct": _num(t["accuracy_pct"]), "quality_flag": t["quality_flag"],
                             "task_family": t["task_family"], "mean_duration_ms": t["mean_duration_ms"]}
    scored = sum(1 for r in runs if r.get("accuracy_pct") is not None)
    memory = Counter((r.get("memory") or {}).get("status", "?") for r in runs)
    return {
        "accuracy_pct_micro": headline.get("ALL"),
        "accuracy_pct_micro_judge": headline.get("ALL_JUDGE"),
        "n_queries": len(runs),
        "n_scored": scored,
        "n_unscored": len(runs) - scored,
        "n_failed": failed,
        "n_personas": len({r.get("user_id") for r in runs}),
        "status_counts": dict(Counter(r.get("status", "ok") for r in runs)),
        "memory": {"status_counts": dict(memory), "n_truncated": sum(1 for r in runs if (r.get("memory") or {}).get("truncated"))},
        "e6_paired": e6,
        "by_class": by_class,
        "by_axis": by_axis,
        "by_task": by_task,
    }
