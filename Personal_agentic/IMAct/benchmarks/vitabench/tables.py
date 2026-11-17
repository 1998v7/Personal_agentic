"""VitaBench 结果表:总分(用户平均)与按业务域(外卖 / 到店 / 酒旅)的 subtask 平均,以及每个用户的得分。"""

from __future__ import annotations

from ...table import pct, render_table

DOMAIN_LABELS = {"delivery": "外卖", "instore": "到店", "ota": "酒旅"}
PER_USER_LIMIT = 60  # 用户多于这个数时不逐个列出


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def render(summary: dict) -> str:
    results = summary.get("results") or []
    subtasks = [s for r in results for s in (r.get("subtasks") or [])]
    domains = [d for d in DOMAIN_LABELS if any(s.get("domain") == d for s in subtasks)]
    users = [r["reward"] for r in results if r.get("reward") is not None]
    head = render_table(
        ["Model", "用户数", "subtask 数", "总分(用户平均)", "subtask 平均", *[DOMAIN_LABELS[d] for d in domains]],
        [[summary.get("model") or "-", len(users), len(subtasks), pct(_mean(users)), pct(_mean([s["reward"] for s in subtasks])),
          *[pct(_mean([s["reward"] for s in subtasks if s.get("domain") == d])) for d in domains]]],
        title="VitaBench personalization(0-100;一个用户的分数 = 其 subtask 得分的平均)",
    )
    footer = [f"完成 {len(results)} 个用户,失败 {len(summary.get('failed') or [])} 个(没有分数)"
              + (f",其中 {len(summary['skipped'])} 个沿用已有结果" if summary.get("skipped") else "")]
    if summary.get("num_evaluation_errors"):
        footer.append(f"评分错误 {summary['num_evaluation_errors']} 个用户:裁判回复不可用,记 0 分")
    if subtasks and not domains:
        footer.append("这批结果没有记录 subtask 的业务域(旧版本结果),无法按域拆分")
    if len(results) > PER_USER_LIMIT:
        return head + "\n" + "\n".join(f"  {f}" for f in footer)
    rows = []
    for r in sorted(results, key=lambda r: (str(r.get("task_id")), r.get("trial", 0))):
        subs = r.get("subtasks") or []
        name = str(r.get("task_id")) + (f" #{r['trial']}" if r.get("trial") else "")
        rows.append([name, len(subs), sum(1 for s in subs if (s.get("reward") or 0) >= 1), pct(r.get("reward"))])
    detail = render_table(["用户", "subtask 数", "满分 subtask", "得分"], rows, title="各用户", footer=footer)
    return head + "\n\n" + detail
