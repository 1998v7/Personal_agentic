"""MCP-Universe 结果表。第一张与论文表格同列(各领域 SR + Overall SR / AE / AS),第二张是各领域明细。"""

from __future__ import annotations

from ...table import render_table

# 论文表格的列顺序(去掉本接入不跑的 3D Designing)
PAPER_COLUMNS = (
    ("location_navigation", "Location Navigation"),
    ("repository_management", "Repository Management"),
    ("financial_analysis", "Financial Analysis"),
    ("browser_automation", "Browser Automation"),
    ("web_search", "Web Searching"),
)


def render(summary: dict) -> str:
    domains = summary.get("domains") or {}
    model = summary.get("agent_model") or "-"
    present = [(k, label) for k, label in PAPER_COLUMNS if k in domains]
    paper = render_table(
        ["Model", *[label for _, label in present], "Overall SR", "AE", "AS"],
        [[model, *[domains[k].get("SR") for k, _ in present], summary.get("SR"), summary.get("AE"), summary.get("AS")]],
        title="MCP-Universe(SR / AE 为百分数,与论文同口径;各领域列为 SR)",
    )
    rows = [[k, d.get("completed"), d.get("failed"), d.get("evaluation_errors"), d.get("num_successful"), d.get("SR"), d.get("AE"),
             d.get("AS"), d.get("avg_iterations")] for k, _ in PAPER_COLUMNS if (d := domains.get(k))]
    rows += [[k, d.get("completed"), d.get("failed"), d.get("evaluation_errors"), d.get("num_successful"), d.get("SR"), d.get("AE"),
              d.get("AS"), d.get("avg_iterations")] for k, d in domains.items() if k not in dict(PAPER_COLUMNS)]
    rows.append(["Overall", summary.get("completed"), summary.get("failed"), summary.get("evaluation_errors"), summary.get("num_successful"),
                 summary.get("SR"), summary.get("AE"), summary.get("AS"), summary.get("avg_iterations")])
    footer = []
    if summary.get("skipped"):
        footer.append(f"其中 {summary['skipped']} 个 task 沿用了已有结果(断点续跑)")
    if summary.get("failed"):
        footer.append(f"失败 {summary['failed']} 个:重试用尽仍出错,没有分数,不计入 SR / AE")
    if summary.get("evaluation_errors"):
        footer.append(f"评分错误 {summary['evaluation_errors']} 个:评估时网关或裁判不可用,记 0 分,先查 infra_errors")
    if summary.get("skipped_side_effects"):
        footer.append(f"按 --skip-side-effects 没跑:{len(summary['skipped_side_effects'])} 个会创建资源的 task")
    footer.append("SR = 全部评估都通过的 task 占比;AE = 每个 task 通过的评估占比的平均;AS = 成功 task 的平均 ReAct 轮数")
    detail = render_table(["领域", "完成", "失败", "评分错误", "成功", "SR", "AE", "AS", "平均轮数"], rows,
                          title="各领域明细", separators=[len(rows) - 1], footer=footer)
    return paper + "\n\n" + detail
