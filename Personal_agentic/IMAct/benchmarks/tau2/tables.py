"""τ²-bench 结果表:语言 × domain 的 pass^1(%)矩阵,加上题数 / 失败 / 评分错误的矩阵。"""

from __future__ import annotations

from ...table import pct, render_table

LANGUAGE_ORDER = ("en", "es", "pt", "hi", "zh", "ko")
DOMAIN_ORDER = ("airline", "retail", "telecom")


def _split(name: str) -> tuple[str, str]:
    lang, _, domain = name.partition("-")
    return (lang, domain) if domain else ("en", name)


def _order(values, preferred):
    return [v for v in preferred if v in values] + sorted(v for v in values if v not in preferred)


def _weighted(cells: list[dict]) -> float | None:
    """按计分题数加权的平均分(旧汇总没有 scored 时按 completed)。"""
    num = den = 0.0
    for c in cells:
        n = c.get("scored", c.get("completed")) or 0
        if c.get("avg_reward") is not None and n:
            num += c["avg_reward"] * n
            den += n
    return num / den if den else None


def render(summary: dict) -> str:
    cells = summary.get("cells") or {}
    grid = {_split(name): c for name, c in cells.items()}
    langs = _order({l for l, _ in grid}, LANGUAGE_ORDER)
    doms = _order({d for _, d in grid}, DOMAIN_ORDER)
    score_rows = []
    for lang in langs:
        row = [lang] + [pct(grid[(lang, d)].get("avg_reward")) if (lang, d) in grid else None for d in doms]
        row.append(pct(_weighted([grid[(lang, d)] for d in doms if (lang, d) in grid])))
        score_rows.append(row)
    total = ["全部"] + [pct(_weighted([grid[(l, d)] for l in langs if (l, d) in grid])) for d in doms] + [pct(_weighted(list(grid.values())))]
    score_rows.append(total)
    title = f"τ²-bench {summary.get('benchmark', '')} · {summary.get('agent_model', '-')}(pass^1,%)"
    scores = render_table(["语言", *doms, "全部"], score_rows, title=title, separators=[len(score_rows) - 1])

    def count(c: dict | None) -> str:
        if not c:
            return "-"
        text = str(c.get("completed", 0))
        extra = [f"失败{c['failed']}" if c.get("failed") else "", f"评分错误{c['evaluation_errors']}" if c.get("evaluation_errors") else ""]
        extra = [e for e in extra if e]
        return text + (f"({','.join(extra)})" if extra else "")

    count_rows = [[lang] + [count(grid.get((lang, d))) for d in doms] for lang in langs]
    footer = [f"完成 {summary.get('completed', 0)},失败 {summary.get('failed', 0)},评分错误 {summary.get('evaluation_errors', 0)}"
              + (f";其中 {summary['skipped']} 个沿用已有结果(断点续跑)" if summary.get("skipped") else "")]
    if summary.get("failed"):
        footer.append("失败:重试用尽仍出错,没有分数,不计入平均")
    if summary.get("evaluation_errors"):
        footer.append("评分错误:对话跑完但裁判回复不可用,记 0 分;先查原因再看分数")
    counts = render_table(["语言", *doms], count_rows, title="各 cell 的 task 数", footer=footer)
    return scores + "\n\n" + counts
