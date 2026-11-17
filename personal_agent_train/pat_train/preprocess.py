"""生成 uni-agent 训练用的 parquet:每行一个 episode 的入口(哪个 bench 的哪道题),对话本身在训练时由 Task 现场生成。

    python -m pat_train.preprocess --out ~/pat/data                       # 缺省 tau2 + mcp_universe
    python -m pat_train.preprocess --out ~/pat/data --benches tau2 mcp_universe vitabench   # 内网机器上

* tau2(τ²-bench 格式):训练集 = retail 的 base 划分 + telecom 的 full 划分,都去掉 text-v1 的冻结评测子集
  (airline 的 base 50 题全在冻结子集里,不进训练集);验证集 = 英文 text-v1 冻结子集(retail / telecom / airline 各 50)。
* vitabench(VitaBench 格式,执行层 + Memory):personalization 里每个有评分标准的 subtask 一行。同样没有单独的训练划分,
  且必须带 Memory(内网),这里只生成数据;本机跑不了。
* mcp_universe(deep research 格式):web_search / financial_analysis / location_navigation 中不写 notion / github 的
  135 题。MCP-Universe 没有单独的训练划分,这些就是它的评测题:这里只用来打通格式与流程,正式训练要换成同格式的新题。

每行的 ``prompt`` 只是一句占位说明:uni-agent 用它做长度过滤、日志展示;Task 不读它。样本字段放在
``extra_info.tools_kwargs.task.metadata``(见 ``pat_train.agent.sample_fields``)。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

BENCHES = ("tau2", "mcp_universe", "vitabench")


def row(task_name: str, data_source: str, metadata: dict[str, str], index: int) -> dict[str, Any]:
    metadata = {"domain": "", "split": "", "task_id": "", "subtask": "", **metadata}
    prompt = [{"role": "user", "content": f"[{task_name}] {metadata['domain']} {metadata['task_id']}"}]
    return {
        "data_source": data_source,
        "prompt": prompt,
        "extra_info": {"index": index, "tools_kwargs": {"task": {"name": task_name, "metadata": metadata}}},
    }


def tau2_rows(split: str) -> list[dict[str, Any]]:
    from .tasks.tau2 import NAME, TRAIN_SPLITS, eval_cell, train_cell

    rows = []
    if split == "train":
        for domain, task_split in TRAIN_SPLITS.items():
            for cid in train_cell(domain, task_split).canonical_ids:
                rows.append(row(NAME, f"tau2/{domain}", {"domain": domain, "split": task_split, "task_id": cid}, len(rows)))
    else:
        for domain in ("retail", "telecom", "airline"):
            for cid in eval_cell(domain).canonical_ids:
                rows.append(row(NAME, f"tau2/{domain}", {"domain": domain, "split": "text-v1", "task_id": cid}, len(rows)))
    return rows


def mcp_rows(split: str) -> list[dict[str, Any]]:
    from .tasks.mcp_universe import NAME, train_tasks

    return [row(NAME, f"mcp_universe/{t.domain}", {"domain": t.domain, "task_id": t.id}, i) for i, t in enumerate(train_tasks())]


def vita_rows(split: str) -> list[dict[str, Any]]:
    from .tasks.vitabench import NAME, _tasks

    rows = []
    for task in _tasks("chinese").values():
        for i, sub in enumerate(task.subtasks):
            if sub.evaluation_criteria is not None:
                rows.append(row(NAME, f"vitabench/{sub.domain}", {"domain": sub.domain, "task_id": task.id, "subtask": str(i)}, len(rows)))
    return rows


BUILDERS = {"tau2": tau2_rows, "mcp_universe": mcp_rows, "vitabench": vita_rows}


def build(benches: list[str], split: str, *, seed: int, limit: int | None) -> list[dict[str, Any]]:
    """``limit``:每个 data_source(bench/domain)最多取多少条,避免 telecom 这种大池子淹没其他来源。"""
    rows: list[dict[str, Any]] = []
    for bench in benches:
        part = BUILDERS[bench](split)
        random.Random(f"{seed}:{bench}:{split}").shuffle(part)
        if limit:
            taken: dict[str, int] = {}
            kept = []
            for r in part:
                if taken.get(r["data_source"], 0) < limit:
                    taken[r["data_source"]] = taken.get(r["data_source"], 0) + 1
                    kept.append(r)
            part = kept
        rows += part
    random.Random(f"{seed}:{split}").shuffle(rows)
    for i, r in enumerate(rows):
        r["extra_info"]["index"] = i
    return rows


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--benches", nargs="+", default=["tau2", "mcp_universe"], choices=BENCHES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="", help="输出文件名前缀(如 smoke_ → smoke_train.parquet)")
    p.add_argument("--train-limit", type=int, default=None, help="每个 data_source(bench/domain)最多取多少条训练样本")
    p.add_argument("--val-limit", type=int, default=None, help="每个 data_source 最多取多少条验证样本")
    a = p.parse_args(argv)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for split, limit in (("train", a.train_limit), ("val", a.val_limit)):
        rows = build(a.benches, split, seed=a.seed, limit=limit)
        path = out / f"{a.name}{split}.parquet"
        pd.DataFrame(rows).to_parquet(path)
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["data_source"]] = counts.get(r["data_source"], 0) + 1
        print(f"{split}: {len(rows)} 行 → {path}  {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
