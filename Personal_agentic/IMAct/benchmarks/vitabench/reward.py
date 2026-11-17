"""诊断模式下的评分:对已完成的对话调用裁判,落盘 grading.json / grading.html,不重跑 Action、Memory 或业务工具。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import TypeAdapter

from ...evaluation.base import JudgeError
from ...evaluation.vitabench import RubricTask, VitaBenchEvaluator
from ...exec_layer.bridge import AsyncBridge
from ...harness import LLMCaller
from ...harness.conversation import ConversationResult, TerminationReason
from ...harness.messages import Message
from ...trace import trace_scope
from .personalization import rubrics_of
from .trace_report import dump

_MESSAGES = TypeAdapter(list[Message])


def grade_simulation(task, subtask, simulation, *, catalog, out: Path, trace, language="chinese"):
    """评分已完成的交互;异常保留为 error/null,不会变成正常的零分。"""
    conversation = ConversationResult(
        messages=_MESSAGES.validate_python(simulation["messages"]),
        termination_reason=TerminationReason(simulation["termination_reason"]),
        duration=simulation.get("duration", 0),
        states=simulation.get("states", {"old_states": [], "new_states": []}),
    )
    criteria = subtask.evaluation_criteria
    rubric_task = RubricTask(instruction=subtask.instruction, rubrics=rubrics_of(criteria), env_time=subtask.environment.get("time"))
    # 留一份评分输入,事后可以核对"当时按什么标准评的"
    dump(out / "grading_task.json", {"id": f"{task.id}_subtask_{subtask.subtask_id}", "domain": subtask.domain, "instructions": subtask.instruction,
                                      "environment": subtask.environment,
                                      "user_scenario": (subtask.user_scenario.model_dump(mode="json") if subtask.user_scenario else {"user_profile": task.user_profile}),
                                      "evaluation_criteria": criteria.model_dump(mode="json") if criteria else None})
    result = {"status": "running", "reward": None, "judge": catalog.snapshot()["roles"]["judge"],
              "evaluation_type": "trajectory", "subtask_id": subtask.subtask_id, "user_id": task.user_id,
              "messages_count": len(conversation.messages), "rubric_pass_rate": None}
    dump(out / "grading.json", result)
    started = time.perf_counter()
    bridge = AsyncBridge("imact-judge")
    judge = LLMCaller(catalog.selected("judge")[1], bridge, trace=trace, trace_stage="judge")
    try:
        with trace_scope(query_id=subtask.subtask_id, phase="judge"):
            trace.emit("grading_start", judge=result["judge"], evaluation_type="trajectory", messages_count=len(conversation.messages))
            reward = VitaBenchEvaluator(judge, language).evaluate(conversation, rubric_task)
            dump(out / "reward_info.raw.json", reward.model_dump(mode="json"))
            checks = reward.nl_rubrics or []
            passed = sum(c.met for c in checks)
            result.update(status="completed" if reward.window_evaluations else "rule_only", reward=reward.reward,
                          rubric_passed=passed, rubric_total=len(checks),
                          rubric_pass_rate=passed / len(checks) if checks else None,
                          window_count=len(reward.window_evaluations or []), reward_info=reward.model_dump(mode="json"))
            trace.emit("grading_result", **result)
    except Exception as exc:
        if isinstance(exc, JudgeError) and exc.windows:
            dump(out / "reward_info.raw.json", {"error": str(exc), "window_evaluations": exc.windows})  # 保留出错时裁判的原始响应
        result.update(status="error", error=f"{type(exc).__name__}: {exc}", reward=None, rubric_pass_rate=None)
        trace.emit("grading_error", error=result["error"])
    finally:
        result["judge_called"] = judge.calls > 0
        result["duration_s"] = round(time.perf_counter() - started, 2)
        dump(out / "grading.json", result)
        try:
            judge.close()
        finally:
            bridge.close()
    return result


def attach_grade(case, simulation, result):
    case.update(evaluation=result["status"], reward=result["reward"], judge=result["judge"],
                judge_called=result["judge_called"],
                grading_error=result.get("error"), rubric_pass_rate=result.get("rubric_pass_rate"))
    simulation.update(evaluation=result["status"], reward=result["reward"], reward_info=result.get("reward_info"))


def export_grade(out: Path):
    """导出独立可读评分页,正文全部转义。"""
    import html

    result = json.loads((out / "grading.json").read_text())
    esc = lambda value: html.escape(str(value))
    rows = []
    for i, check in enumerate((result.get("reward_info") or {}).get("nl_rubrics") or [], 1):
        rows.append(f"<tr><td>{i}</td><td>{esc(check['nl_rubric'])}</td><td>{'通过' if check['met'] else '未通过'}</td><td>{esc(check['justification'])}</td></tr>")
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VitaBench 裁判评分</title><style>body{font:16px/1.7 system-ui;background:#eff5fc;color:#153553;max-width:1200px;margin:36px auto;padding:20px}table{border-collapse:collapse;background:white}td,th{border:1px solid #cedeee;padding:12px;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}a{color:#1b65b0}</style>'
    page += f"<h1>{esc(result['subtask_id'])} · 裁判评分</h1><p>模型：{esc(result['judge']['model'])}；评分状态：{esc(result['status'])}；reward：{esc(result['reward'])}；rubric 通过：{esc(result.get('rubric_passed'))}/{esc(result.get('rubric_total'))}</p>"
    page += '<p>VitaBench trajectory 口径：全部 rubric 通过时 reward=1，否则为0；逐项通过比例单独报告。单个 case 不代表整体评测成绩。</p><p><a href="episode.html">完整执行与裁判轨迹</a> · <a href="grading.json">完整评分 JSON</a> · <a href="grading_task.json">原始任务与评分规则</a></p>'
    page += '<table><thead><tr><th>#</th><th>Rubric</th><th>结果</th><th>裁判依据</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    if result.get("error"):
        page += '<pre>'+esc(result['error'])+'</pre>'
    for window in (result.get("reward_info") or {}).get("window_evaluations") or []:
        page += f"<details><summary>窗口 {window['window_idx']} 完整输入输出</summary><pre>{esc(json.dumps(window,ensure_ascii=False,indent=2))}</pre></details>"
    (out / "grading.html").write_text(page+'</html>', encoding="utf-8")
