"""PersonaMem-v3 结果表:头条(ALL / ALL-JUDGE / 各类别)与各题型明细,口径同官方 aggregate_eval。"""

from __future__ import annotations

from ...table import render_table

CLASS_ORDER = ("ranking", "chatbot", "agentic", "over_personalization", "proactive", "other")
AXIS_LABELS = {"explicit_retrieval": "显式检索", "mixed": "混合", "implicit_inference": "隐式推断"}


def render(summary: dict) -> str:
    by_class = summary.get("by_class") or {}
    classes = [c for c in CLASS_ORDER if c in by_class] + sorted(c for c in by_class if c not in CLASS_ORDER)
    model = summary.get("agent_model") or "-"
    head = render_table(
        ["Model", "ALL", "ALL-JUDGE", *classes],
        [[model, summary.get("accuracy_pct_micro"), summary.get("accuracy_pct_micro_judge"), *[by_class[c]["accuracy_pct"] for c in classes]]],
        title=f"PersonaMem-v3 · {summary.get('mode') or 'llm_memory'}(0-100,micro 按题平均;judge = {summary.get('judge_model') or '-'})",
    )
    by_task = summary.get("by_task") or {}
    order = {c: i for i, c in enumerate(CLASS_ORDER)}
    tasks = sorted(by_task.items(), key=lambda kv: (order.get(kv[1].get("task_family") or "other", 99), kv[0]))
    rows, seps, last = [], [], None
    for name, d in tasks:
        fam = d.get("task_family") or "other"
        if last is not None and fam != last:
            seps.append(len(rows))
        last = fam
        flag = d.get("quality_flag")
        rows.append([fam, name, d.get("n"), d.get("accuracy_pct"), "" if flag in (None, "", "ok") else flag])
    axes = summary.get("by_axis") or {}
    footer = [f"{summary.get('n_queries', 0)} 道题有结果({summary.get('n_scored', 0)} 道计分,{summary.get('n_unscored', 0)} 道主指标缺失不计入),"
              f"失败 {summary.get('n_failed', 0)} 道(没有分数),{summary.get('n_personas', 0)} 个 persona"]
    if axes:
        footer.append("能力轴:" + ",".join(f"{AXIS_LABELS.get(k, k)} {v['accuracy_pct']:.2f}({v['n']})" for k, v in axes.items() if v.get("accuracy_pct") is not None))
    e6 = summary.get("e6_paired") or {}
    if e6:
        footer.append(f"主动纠错 warn/foil 配对:{e6.get('n_pairs')} 对,paired F1 = {e6.get('paired_f1')}")
    mem = (summary.get("memory") or {}).get("status_counts") or {}
    if mem:
        footer.append("Memory:" + ",".join(f"{k} {v}" for k, v in mem.items())
                      + (f",截断 {summary['memory']['n_truncated']}" if summary.get("memory", {}).get("n_truncated") else ""))
    footer.append("short_vs_long_term_lifecycle 在官方实现里永远不计分(见 README_PERSONAMEM_V3.md 第 5 节)")
    detail = render_table(["类别", "题型", "题数", "得分", "提示"], rows, title="各题型明细", align="llrrl", separators=seps, footer=footer)
    return head + "\n\n" + detail
