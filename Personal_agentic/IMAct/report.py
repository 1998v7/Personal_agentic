"""把任意一次运行的结果重新打印成表(不重跑、不调用模型)。

    python -X utf8 -m IMAct.report results/tau2-full results/mcpu-full results/pm3-25

默认从结果目录里的每题结果文件重新汇总(所以旧目录、断点续跑过的目录也能得到完整、口径一致的表);
``--from-summary`` 则只读 ``summary.json``。识别 benchmark 靠 ``summary.json`` 的 ``bench`` 字段,旧汇总没有时按字段推断。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCHES = ("tau2", "mcp_universe", "personamem_v3", "vitabench")


def detect(summary: dict) -> str | None:
    bench = summary.get("bench")
    if bench in BENCHES:
        return bench
    if "cells" in summary:  # 早期 τ² 汇总没有 bench 字段
        return "tau2"
    if "domains" in summary and "SR" in summary:
        return "mcp_universe"
    if "results" in summary and "model" in summary:
        return "vitabench"
    return None


def _module(bench: str, name: str):
    import importlib

    return importlib.import_module(f"IMAct.benchmarks.{bench}.{name}")


def load(out: Path, *, rebuild: bool = True) -> tuple[str, dict]:
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    bench = detect(summary)
    if bench is None:
        raise ValueError("认不出这是哪个 benchmark 的 summary.json")
    if rebuild:
        summary = _module(bench, "run").rebuild(out)
    return bench, summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="把结果目录打印成表(不调用模型)")
    p.add_argument("dirs", nargs="+", help="结果目录(即运行时的 --out)")
    p.add_argument("--from-summary", action="store_true", help="只读 summary.json,不从每题结果文件重新汇总")
    args = p.parse_args(argv)
    code = 0
    for i, d in enumerate(args.dirs):
        out = Path(d)
        out = out.parent if out.name == "summary.json" else out
        if i:
            print()
        print(f"== {out}")
        try:
            bench, summary = load(out, rebuild=not args.from_summary)
            print(_module(bench, "tables").render(summary))
        except (OSError, ValueError) as exc:
            print(f"  读不出 {out}: {exc}", file=sys.stderr)
            code = 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
