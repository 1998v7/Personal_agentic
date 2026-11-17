"""列出 parquet 里每个 data_source 的前几行行号。用法:python list_rows.py <parquet> [每类几行]"""
import sys
import pandas as pd

df = pd.read_parquet(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
for src, group in df.groupby("data_source"):
    rows = [(i, group.loc[i, "extra_info"]["tools_kwargs"]["task"]["metadata"]["task_id"]) for i in group.index[:n]]
    print(src, rows)
