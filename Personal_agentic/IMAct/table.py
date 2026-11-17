"""终端里的对齐表格。中文等全角字符按两格宽计算,保证列对齐。与具体 benchmark 无关。

    print(render_table(["领域", "SR", "AE"], [["web_search", 0.0, 12.5]], title="MCP-Universe", align="lrr"))

数字统一按 ``digits`` 位小数显示;``None`` 显示为 ``-``。
"""

from __future__ import annotations

import unicodedata
from typing import Any, Iterable, Sequence


def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int, align: str) -> str:
    gap = width - display_width(text)
    if gap <= 0:
        return text
    if align == "r":
        return " " * gap + text
    if align == "c":
        return " " * (gap // 2) + text + " " * (gap - gap // 2)
    return text + " " * gap


def fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_table(headers: Sequence[str], rows: Iterable[Sequence[Any]], *, title: str | None = None, align: str | None = None,
                 digits: int = 2, separators: Iterable[int] = (), footer: Iterable[str] = ()) -> str:
    """``align``:每列一个字符 l / r / c,缺省首列左对齐、其余右对齐。``separators``:在这些行号(从 0 起)之前画一条分隔线。"""
    cells = [[fmt(v, digits) for v in row] for row in rows]
    n = len(headers)
    align = (align or "l" + "r" * (n - 1)).ljust(n, "r")
    widths = [max([display_width(h) for h in [headers[i]]] + [display_width(r[i]) for r in cells if i < len(r)]) for i in range(n)]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    heavy = "+" + "+".join("=" * (w + 2) for w in widths) + "+"

    def render_row(values: Sequence[str], aligns: str) -> str:
        return "| " + " | ".join(_pad(values[i] if i < len(values) else "", widths[i], aligns[i]) for i in range(n)) + " |"

    out = []
    if title:
        out.append(title)
    out += [line, render_row(list(headers), "c" * n), heavy]
    seps = set(separators)
    for i, row in enumerate(cells):
        if i in seps:
            out.append(line)
        out.append(render_row(row, align))
    out.append(line)
    out += [f"  {f}" for f in footer if f]
    return "\n".join(out)


def pct(value: float | None) -> float | None:
    """0-1 → 0-100。"""
    return None if value is None else 100.0 * value
