"""从实际采集记录和 SQLite 导出完整诊断轨迹;不调用任何模型或业务工具。"""
from __future__ import annotations

import argparse
import collections
import html
import json
import sqlite3
from contextlib import closing
from pathlib import Path


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_database(path):
    result = {"attempts": [], "events": [], "tasks": [], "notes": []}
    if not path.exists():
        return result
    with closing(sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        result["integrity_check"] = db.execute("pragma integrity_check").fetchone()[0]
        available = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        for table in ("attempts", "events", "tasks", "notes"):
            if table not in available:
                continue
            for r in db.execute(f"select * from {table} order by rowid"):
                row = dict(r)
                for key in ("context", "decision", "error", "metadata", "input", "output", "workspace", "staged_workspace", "value"):
                    if row.get(key) is not None:
                        try:
                            row[key] = json.loads(row[key])
                        except (ValueError, TypeError):
                            pass
                result[table].append(row)
    return result


def export_trace(out: str | Path):
    out = Path(out)
    records = []
    if (out / "trace.jsonl").exists():
        for line in (out / "trace.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    db = read_database(out / "exec.sqlite3")
    manifest = json.loads((out / "case.json").read_text()) if (out / "case.json").exists() else {}
    simulation = json.loads((out / "simulation.json").read_text()) if (out / "simulation.json").exists() else None
    if simulation is None:
        simulation = [json.loads(p.read_text()) for p in sorted(out.glob("*.trial*.json"))]
    attempts = db["attempts"]
    counts = collections.Counter(r["stage"] for r in records)
    errors = [{"attempt_id": r["id"], "event_id": r["event_id"], "error": r["error"]} for r in attempts if r["error"]]
    outputs = [r for r in attempts if r["decision"] is not None]
    messages = simulation.get("messages", []) if isinstance(simulation, dict) else [m for s in simulation for m in s.get("messages", [])]
    action_inputs = [r for r in records if r["stage"] == "action_input"]
    event_feedback = [r["data"]["messages"] for r in records if r["stage"] == "benchmark_feedback"]
    tool_feedback = [m for group in event_feedback for m in group if m["role"] == "tool"]
    if not messages:  # 中途失败仍统计已逐步落盘的实际反馈
        messages = [m for group in event_feedback for m in group]
    tool_calls = [call for m in messages for call in m.get("tool_calls") or []]
    summary = {
        "domain": manifest.get("domain"), "subtask_id": manifest.get("subtask_id"),
        "status": manifest.get("status"), "termination_reason": simulation.get("termination_reason") if isinstance(simulation, dict) else None,
        "evaluation": manifest.get("evaluation", "see simulation"), "reward": manifest.get("reward"),
        "judge_calls": counts["judge_request"], "judge_responses": counts["judge_response"],
        "trace_records": len(records), "stages": dict(counts), "attempts": len(attempts),
        "model_outputs": len(outputs), "model_outputs_with_request_id": sum(bool((a["metadata"] or {}).get("request_id")) for a in outputs),
        "validation_errors": errors, "tool_calls": len(tool_calls), "tool_names": [c["name"] for c in tool_calls],
        "tool_feedback": len(tool_feedback), "tool_errors": sum(bool(m.get("error")) for m in tool_feedback),
        "input_chars": [len(json.dumps(r["data"]["context"], ensure_ascii=False)) for r in action_inputs],
        "memory_calls": counts["memory_request"],
        "memory_normalized": [r["data"]["payload"] for r in records if r["stage"] == "memory_normalized"],
        "action_usage": {k: sum(((r["metadata"] or {}).get("usage") or {}).get(k, 0) for r in attempts) for k in ("prompt_tokens", "completion_tokens")},
        "sqlite_integrity_check": db.get("integrity_check"),
        "coverage": {"attempt_contexts": sum(r["context"] is not None for r in attempts),
                     "recorded_action_inputs": counts["action_input"], "recorded_action_outputs": counts["action_output"],
                     "recorded_action_errors": counts["action_error"]},
        "final_workspaces": [{"task_id": t["task_id"], "rev": t["rev"], "workspace": t["workspace"]} for t in db["tasks"]],
    }
    dump(out / "audit.json", summary)
    payload = {"manifest": manifest, "summary": summary, "records": records, "sqlite": db, "simulation": simulation}
    dump(out / "full_trace.json", payload)
    embedded = json.dumps(payload, ensure_ascii=False, default=str).replace("<", "\\u003c")
    (out / "trace.html").write_text(HTML.replace("__TRACE_DATA__", embedded), encoding="utf-8")
    return summary


HTML = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IMAct 全链路轨迹</title><style>
:root{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172e48;background:#f1f5fa}*{box-sizing:border-box}body{margin:0}header{background:#102d4e;color:white;padding:32px 5vw}h1{font-size:27px;margin:8px 0}header p{color:#d1e3f5;line-height:1.7}main{max-width:1440px;margin:auto;padding:24px}a{color:#2262a4}header a{color:#bfe0ff}.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.metric,.intro,article{background:white;border:1px solid #d7e2ee;border-radius:10px;padding:18px}.metric strong{display:block;font-size:27px;margin-bottom:6px}.metric span{color:#536a80;font-size:13px}.intro{margin:20px 0;line-height:1.8}.flow{display:flex;gap:8px;flex-wrap:wrap;color:#cfe7ff;font-size:13px}.flow span{background:#224b77;padding:8px 12px;border-radius:6px}.bar{display:flex;gap:12px;position:sticky;top:0;padding:14px 0;background:#f1f5fa;z-index:2;flex-wrap:wrap}input,select,button{border:1px solid #bed0e3;padding:10px;border-radius:6px;background:white;font:inherit}input{min-width:250px;flex:1}button{cursor:pointer}article{margin:0 0 12px;border-left:4px solid #639bd4}article.error{border-left-color:#c25252}.meta{color:#667f98;font-size:12px;margin:7px 0}.label{font-size:15px;font-weight:700}.preview{white-space:pre-wrap;line-height:1.6;font-size:14px;word-break:break-word;margin:10px 0}details{margin-top:12px;border-top:1px solid #e5ebf2;padding-top:10px}summary{cursor:pointer;color:#24629a;font-size:14px}pre{white-space:pre-wrap;word-break:break-word;background:#f5f8fc;padding:14px;border-radius:6px;font:12px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace;max-height:600px;overflow:auto}.count{color:#60758b;font-size:13px}h2{font-size:17px}#empty{display:none;padding:24px;text-align:center}.warn{color:#9a4c22}footer{padding:20px;text-align:center;color:#60758b;font-size:13px}@media(max-width:700px){.grid{grid-template-columns:repeat(2,1fr)}main{padding:12px}header{padding:22px}.bar{position:static}}
</style><header><div>IMAct / VitaBench · 可追溯诊断</div><h1 id="title"></h1><p id="subtitle"></p><div class="flow"><span>用户问题</span>→<span>Memory</span>→<span>Harness 输入</span>→<span>Action 输出</span>→<span>校验 / 工具 / 用户反馈</span>↻</div><p><a href="full_trace.json" download>完整 JSON</a> · <a href="trace.jsonl" download>逐步采集 JSONL</a> · <a href="exec.sqlite3" download>执行层 SQLite</a> · <a href="audit.json" download>审计统计</a></p></header>
<main><div id="metrics" class="grid"></div><section class="intro"><h2>原始用户问题</h2><div id="question"></div><div class="meta" id="scope"></div><details><summary>角色、Memory 与样本来源</summary><pre id="manifest"></pre></details><details><summary>最终执行层状态与订单证据</summary><pre id="workspaces"></pre></details><details><summary>SQLite 每次尝试：含被拒绝的输出及 repair feedback</summary><pre id="attempts"></pre></details><details><summary>完整用户 / 助手 / 工具对话</summary><pre id="conversation"></pre></details></section>
<div class="bar"><select id="stage"><option value="">全部阶段</option></select><input id="search" placeholder="搜索事件 ID、工具名、错误或正文"><button id="errors">只看错误</button><button id="reset">重置</button></div><p id="count" class="count"></p><div id="timeline"></div><div id="empty">没有匹配的记录</div></main><footer>原始正文不截断；折叠项可展开。模型/Memory 为真实外部请求，业务工具在 VitaBench 模拟环境中执行。未运行裁判的样本没有正式分数。</footer>
<script id="data" type="application/json">__TRACE_DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),S=D.summary,M=D.manifest;
const labels={case_start:'样本开始',memory_request:'Memory 请求',memory_response:'Memory 原始返回',memory_error:'Memory 请求错误',memory_normalized:'Memory → Harness',user_reply:'用户回复',benchmark_feedback:'反馈 → Agent',harness_event:'事件 → Harness',action_input:'Harness → Action 完整输入',action_output:'Action 原始输出',action_error:'Action API 错误',harness_result:'Harness 校验通过 / 提交结果',harness_error:'Harness 拒绝 / 重试反馈',conversation_end:'对话结束',case_end:'样本结果',case_error:'样本异常'};
const J=x=>JSON.stringify(x,null,2),set=(id,v)=>document.getElementById(id).textContent=v??'';
set('title',(M.domain||'运行')+' · '+(M.subtask_id||'全链路轨迹'));set('subtitle','执行状态：'+(M.status||'已导出')+' ｜ 终止原因：'+(S.termination_reason||'见异常')+' ｜ 裁判评分：'+(M.evaluation||'见结果'));
set('question',M.instruction||'见每个 start 事件');set('scope','按事件实际发生顺序排列。输入 / 输出 / 反馈各自保留完整正文；没有截掉失败后重试。');set('manifest',J(M));set('workspaces',J(S.final_workspaces));set('attempts',J(D.sqlite.attempts));set('conversation',J(D.simulation));
for(const [value,label] of [[S.model_outputs+'/'+S.attempts,'Action 输出 / 尝试'],[S.validation_errors.length,'失败尝试（含格式与校验）'],[S.memory_calls,'Memory 请求'],[S.tool_feedback,'工具回执'],[S.tool_errors,'业务工具错误']]){let e=document.createElement('div');e.className='metric';let n=document.createElement('strong');n.textContent=value;let l=document.createElement('span');l.textContent=label;e.append(n,l);document.getElementById('metrics').append(e)}
const select=document.getElementById('stage');for(const stage of [...new Set(D.records.map(r=>r.stage))]){let o=document.createElement('option');o.value=stage;o.textContent=labels[stage]||stage;select.append(o)}
const addDetail=(parent,title,value)=>{let d=document.createElement('details'),s=document.createElement('summary'),p=document.createElement('pre');s.textContent=title;p.textContent=typeof value==='string'?value:J(value);d.append(s,p);parent.append(d)};
function short(r){const d=r.data;if(r.stage==='action_output')return d.content;if(r.stage==='user_reply')return d.message.content;if(r.stage==='benchmark_feedback')return d.messages.map(m=>m.role+(m.name?' / '+m.name:'')+': '+(typeof m.content==='string'?m.content:J(m.content))).join('\\n\\n');if(r.stage==='memory_normalized')return d.payload.status+'\\n'+(d.payload.context||d.payload.error||'');if(r.stage==='memory_response'){try{const x=JSON.parse(d.body);return (x.resultMap?.context||x.context||J(x))}catch{return d.body}}if(r.stage==='harness_error'||r.stage.endsWith('_error'))return d.error;if(r.stage==='harness_result')return J(d.output.action);if(r.stage==='harness_event')return d.request.event.type+' / retry='+d.retry_index;if(r.stage==='action_input')return '完整 context '+J(d.context).length+' 字符；事件 '+(r.correlation.event_id||'');return ''}
let onlyErrors=false;function render(){const search=document.getElementById('search').value.toLowerCase(),stage=select.value,box=document.getElementById('timeline');box.replaceChildren();let count=0;for(const r of D.records){const bad=r.stage.endsWith('_error')||(r.stage==='benchmark_feedback'&&r.data.messages.some(m=>m.error));if(stage&&stage!==r.stage||onlyErrors&&!bad||search&&!J(r).toLowerCase().includes(search))continue;count++;let a=document.createElement('article');if(bad)a.classList.add('error');let title=document.createElement('div');title.className='label';title.textContent='#'+r.seq+'  '+(labels[r.stage]||r.stage);let meta=document.createElement('div');meta.className='meta';meta.textContent=r.time+'  '+Object.entries(r.correlation).filter(([k,v])=>v!=null).map(([k,v])=>k+'='+v).join(' · ');a.append(title,meta);let p=document.createElement('div');p.className='preview';let text=short(r)||'';if(text.length>1400){p.textContent=text.slice(0,1400)+'…（下方展开全文）';addDetail(a,'正文全文（'+text.length+' 字符）',text)}else p.textContent=text;a.insertBefore(p,a.children[2]||null);if(r.stage==='action_input'){addDetail(a,'Harness 系统协议全文',r.data.context.system);addDetail(a,'任务策略全文',r.data.context.task_policy);addDetail(a,'Action 实际收到的结构化 input',r.data.context.input);addDetail(a,'传入 provider 的 messages 与参数（system 合并前）',{messages:r.data.messages_before_provider,params:r.data.params})}addDetail(a,'本条采集记录全文',r);box.append(a)}set('count',count+' / '+D.records.length+' 条记录');document.getElementById('empty').style.display=count?'none':'block'}
select.onchange=render;document.getElementById('search').oninput=render;document.getElementById('errors').onclick=()=>{onlyErrors=!onlyErrors;render()};document.getElementById('reset').onclick=()=>{onlyErrors=false;select.value='';document.getElementById('search').value='';render()};render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    args = parser.parse_args()
    summary = export_trace(args.out)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"final_workspaces", "memory_normalized"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
