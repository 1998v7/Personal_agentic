"""PersonaMem-v3 一道题的完整流程(官方 ``llm_memory`` 模式,记忆换成外部 Memory 服务)。

    test.json 的一项 → 构造官方 runner 的 inst → 题型 runner(prompt、作答、评分,均为移植的上游代码 pm3/)
                         └─ 记忆块:向 Memory 服务 query(按 query_id 定位已构建的快照)+ 截至 T 的日历状态

与官方 ``run_eval.py`` 的对应:
* inst 的构造同 ``_load_queries``(``instance_full`` + 补回顶层 ``user_query``);
* 调度同 ``run_eval_dispatch.dispatch_single``;结果打包同 ``_pack_rec``(``metrics`` / ``status`` / ``agent_response``);
* 记忆块:官方 llm_memory 是本地 LLM 按天压缩出的 ≤4096 token 文本;这里是 Memory 服务返回的 ``context``,同样截到
  ``memory_token_cap``,再接上官方的日历块(日历是实时状态,不进 memory,统一调用接口 §4.1 Adapter.2)。

失败的处理:Memory 服务出错、模型请求在重试后仍失败,都抛出,交给批处理重试 / 记为失败(没有分数);
题型代码自身的异常与官方一样记为 ``status=error``(汇总时按 0 分计)。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from ...evaluation.base import RewardInfo
from ...harness.llm import LLMCaller
from ...harness.messages import UserMessage, now_str
from .pm3.evaluation.backend_query import BackendQuery
from .pm3.evaluation.inference_utils import SnapshotCache, count_tokens
from .pm3.evaluation.run_eval_dispatch import DispatchContext, dispatch_single
from .pm3.evaluation.task_registry import DROPPED_TASK_TYPES, normalize_task_type
from .pm3.scripts.aggregate_eval import _accuracy_value

logger = logging.getLogger("imact.personamem_v3")

MODE = "llm_memory"
DEFAULT_MEMORY_TOKEN_CAP = 4096  # 官方 DEFAULT_MEMORY_TOKEN_CAP(evaluation/prompts.py)
SYSTEM_PROMPT_PLACEHOLDER = "[system prompt]"  # test.json 里"没有用户正文"的题,user_query 是上游填的这种占位文本
REPETITION_TASKS = frozenset({"over_personalization_repetition_recsys", "over_personalization_repetition_chatbot"})
_AGENT_RESPONSE_TRUNCATE_BYTES = 4096  # 官方 results.csv 的截断长度;这里只用于 summary 里的短文本,完整回答另存


class MemoryUnavailable(RuntimeError):
    """Memory 服务这次没能给出记忆(网关错误、快照未构建…)。不能当成"没有记忆"继续答题。"""


# ---------------------------------------------------------------------------------------------- 适配器
class QueryLLMAdapter:
    """官方 runner 调用模型的接口是 ``client.query_llm(prompt) -> str``(``query_llm.QueryLLM``)。这里转到 IMAct 的 LLMCaller。

    与官方一致:一条 user 消息、不另加 system;空正文 / 被截断照样返回给评分(官方会把空答案计为非实质回答)。
    每道题一个实例,统计这道题的调用次数与 token。
    """

    def __init__(self, llm: LLMCaller, stage: str):
        self.llm = llm
        self.stage = stage
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def query_llm(self, prompt, use_history=False, image=None, image_path=None, verbose=False, thread_id=None, temperature=None):
        if use_history or image or image_path:
            raise NotImplementedError("PersonaMem-v3 的 llm_memory 路径不使用多轮历史 / 图片输入")
        params = {} if temperature is None else {"temperature": temperature}
        reply = self.llm.chat([UserMessage(role="user", content=prompt)], allow_incomplete=True, **params)
        self.calls += 1
        self.prompt_tokens += int((reply.usage or {}).get("prompt_tokens", 0) or 0)
        self.completion_tokens += int((reply.usage or {}).get("completion_tokens", 0) or 0)
        return reply.content or ""

    def usage(self) -> dict[str, int]:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


@dataclass
class MemoryResult:
    status: str  # ok / empty
    context: str
    tokens: int = 0
    truncated: bool = False
    raw_chars: int = 0
    elapsed_ms: float = 0.0


def cap_tokens(text: str, cap: int) -> tuple[str, bool]:
    """按行截到 ``cap`` token 以内(与官方 mem0 渲染同法:逐行累加,装不下就停)。单行就超长时按字符截。"""
    if count_tokens(text) <= cap:
        return text, False
    out, used = [], 0
    for line in text.split("\n"):
        t = count_tokens(line)
        if used + t > cap:
            if not out:  # 第一行就装不下:按 token 比例截字符
                out.append(line[: max(1, int(len(line) * cap / max(1, t)))])
            break
        out.append(line)
        used += t
    return "\n".join(out), True


class GatewayMemoryCache(SnapshotCache):
    """官方 ``SnapshotCache`` 的 llm_memory 分支:``_memory_block`` 不再查本地 ledger,改为这道题向 Memory 服务 query 一次。

    一道题一个实例(题目信息在构造时给定)。runner 可能用与题目 ts 不同的时间取记忆(例如重复疲劳题用 cluster 最后一问的时间),
    Memory 服务按 query_id 定位快照,所以记忆本身不随这个时间变;日历块仍按 runner 传入的时间折叠,与官方一致。
    """

    def __init__(self, fetch: Callable[[], MemoryResult]):
        super().__init__(mode=MODE)
        self._fetch = fetch
        self.result: MemoryResult | None = None
        self.memory_ts_seen: list[int] = []

    def _memory_block(self, t_test: int) -> tuple[str, dict]:
        self.memory_ts_seen.append(int(t_test))
        if self.result is None:
            self.result = self._fetch()
        text = self.result.context
        return text, {"total_tokens": count_tokens(text) if text else 0, "memory_mode": True, "n_checkpoints": 1, "per_app": {}}


# ---------------------------------------------------------------------------------------------- 结果
class PM3Run(BaseModel):
    """一道题的运行记录(落盘)。``id`` / ``task_id`` / ``reward_info`` 是批处理框架需要的字段。"""

    id: str
    task_id: str  # = query_id
    user_id: str
    seq: int
    task_type: str
    ts: int
    timestamp: str = Field(default_factory=now_str)
    duration_ms: int
    status: str
    error: str = ""
    metrics: dict[str, Any]
    accuracy_pct: float | None  # 官方 _accuracy_value:0-100,None = 这行不计入
    agent_response: Any = None
    memory: dict[str, Any]
    usage: dict[str, Any]
    reward_info: RewardInfo | None  # 不计分的行(官方 _accuracy_value 返回 None)为 None,不进平均
    trial: int | None = None
    seed: int | None = None


def build_inst(item: dict) -> dict:
    """同官方 ``run_eval._load_queries``:runner 读 ``instance_full``;顶层 ``user_query`` 没在里面时补回(复制,不改原对象)。"""
    inst_full = item.get("instance_full") or item
    if isinstance(inst_full, dict) and item.get("user_query") and "user_query" not in inst_full:
        inst_full = {**inst_full, "user_query": item["user_query"]}
    return json.loads(json.dumps(inst_full, ensure_ascii=False))  # 官方经 instance_json 字符串往返,这里同样深拷贝


def memory_user_query(item: dict, placeholder_mode: str = "empty") -> str:
    """传给 Memory 服务的 ``task.user_query``。没有用户正文的题,test.json 里是 ``[system prompt] …`` 占位文本:
    统一调用接口约定"原数据没有正文时传空字符串",默认传空;``verbatim`` 则原样传。"""
    q = item.get("user_query") or ""
    if placeholder_mode == "empty" and q.lstrip().startswith(SYSTEM_PROMPT_PLACEHOLDER):
        return ""
    return q


def pack_record(result: dict | None) -> tuple[dict, str, str, Any]:
    """官方 ``run_eval._pack_rec`` 的打包规则:合并 subagent_stats 的 token 计数,取回答字段。返回 (metrics, status, error, agent_response)。"""
    if result is None:
        return {}, "no_result", "", ""
    metrics = dict(result.get("metrics") or {})
    sub = result.get("subagent_stats") or {}
    if isinstance(sub, list):  # 重复疲劳题:每次调用一条,累加
        agg: dict = {}
        for s in sub:
            if not isinstance(s, dict):
                continue
            for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cost_usd"):
                v = s.get(k)
                if isinstance(v, (int, float)):
                    agg[k] = agg.get(k, 0) + v
        sub = agg
    elif not isinstance(sub, dict):
        sub = {}
    for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cost_usd"):
        v = sub.get(k)
        if v is not None and k not in metrics:
            metrics[k] = v
    resp = result.get("agent_response")
    if resp is None:
        resp = result.get("agent_response_raw")
    if resp is None:
        resp = result.get("response_for_row")
    if resp is None and isinstance(result.get("responses"), list):
        resp = result.get("responses")
    return metrics, result.get("status", "ok"), result.get("error", "") or "", resp


# ---------------------------------------------------------------------------------------------- runner
class PersonaMemRunner:
    def __init__(
        self,
        *,
        backend_dir: Path,
        answer_llm: LLMCaller | None,
        judge_llm: LLMCaller | None,
        memory_query: Callable[..., dict[str, Any]],
        memory_token_cap: int = DEFAULT_MEMORY_TOKEN_CAP,
        query_placeholder: str = "empty",
        model_name: str = "",
        dry_run: bool = False,
    ):
        """``memory_query(user_id=, query_id=, user_query=, timestamp=)`` 返回 ``{status: ok|empty|error, context, error}``
        (即 ``MpsMemoryClient.query`` 的同步版本)。``judge_llm`` 为 None 时不调用裁判(依赖裁判的分数缺失)。"""
        self.backend_dir = Path(backend_dir)
        self.answer_llm = answer_llm
        self.judge_llm = judge_llm
        self.memory_query = memory_query
        self.memory_token_cap = memory_token_cap
        self.query_placeholder = query_placeholder
        self.model_name = model_name
        self.dry_run = dry_run
        self._local = threading.local()  # 每个线程一份 BackendQuery(官方是每个进程一份)
        self._cluster_locks: dict[str, threading.Lock] = {}
        self._cluster_locks_guard = threading.Lock()

    def _bq(self) -> BackendQuery:
        bq = getattr(self._local, "bq", None)
        if bq is None:
            bq = self._local.bq = BackendQuery(self.backend_dir)
        return bq

    def _cluster_lock(self, inst: dict) -> threading.Lock | None:
        """重复疲劳题:同一 cluster 的多行共用一次 cluster 级运行(官方用模块级缓存去重)。并发时加锁,保证 cluster 只跑一次。"""
        cid = inst.get("cluster_id")
        if not cid:
            return None
        with self._cluster_locks_guard:
            return self._cluster_locks.setdefault(cid, threading.Lock())

    def fetch_memory(self, user_id: str, item: dict) -> MemoryResult:
        started = time.perf_counter()
        reply = self.memory_query(user_id=str(user_id), query_id=item["query_id"], user_query=memory_user_query(item, self.query_placeholder),
                                  timestamp=int(item["ts"]))
        elapsed = (time.perf_counter() - started) * 1000
        status = reply.get("status")
        if status not in ("ok", "empty"):
            raise MemoryUnavailable(f"Memory 服务没有返回记忆({item['query_id']}): {reply.get('error') or status}")
        raw = reply.get("context") or ""
        text, truncated = cap_tokens(raw, self.memory_token_cap) if raw else ("", False)
        return MemoryResult(status="ok" if text.strip() else "empty", context=text, tokens=count_tokens(text) if text else 0,
                            truncated=truncated, raw_chars=len(raw), elapsed_ms=elapsed)

    def run_item(self, user_id: str, seq: int, item: dict, *, run_id: str = "", seed: int | None = None) -> PM3Run:
        task_type = normalize_task_type(item.get("task_type", ""))
        if task_type in DROPPED_TASK_TYPES:
            raise ValueError(f"{item['query_id']}: 题型 {task_type} 已被官方弃用,不应进入评测")
        inst = build_inst(item)
        answer = QueryLLMAdapter(self.answer_llm, "agent") if self.answer_llm is not None else None
        judge = QueryLLMAdapter(self.judge_llm, "judge") if self.judge_llm is not None else None
        cache = GatewayMemoryCache(lambda: self.fetch_memory(user_id, item))
        ctx = DispatchContext(
            user_id=str(user_id), bq=self._bq(), llm_client=answer, judge_client=judge, mode=MODE, snapshot_cache=cache,
            model_name=self.model_name, claude_model="", context_budget=None, enable_llm_judge=judge is not None, dry_run=self.dry_run,
        )
        started = time.time()
        lock = self._cluster_lock(inst) if task_type in REPETITION_TASKS else None
        try:
            if lock:
                lock.acquire()
            try:
                result = dispatch_single(task_type, inst, ctx)
            finally:
                if lock:
                    lock.release()
            metrics, status, error, resp = pack_record(result)
        except (MemoryUnavailable, NotImplementedError):
            raise
        except Exception as exc:  # noqa: BLE001
            if _is_infrastructure_error(exc):
                raise
            # 与官方一致:题型代码自身的异常记为 status=error(汇总按 0 分),并保留位置便于排查
            last = next((l.strip() for l in reversed(traceback.format_exc().splitlines()) if l.startswith("  File ")), "")
            logger.exception("题型代码出错 %s", item["query_id"])
            metrics, status, error, resp = {}, "error", f"{type(exc).__name__}: {exc} | {last}", ""
        duration = int((time.time() - started) * 1000)
        acc = _accuracy_value(task_type, metrics, status)
        mem = cache.result
        memory = {"status": mem.status, "tokens": mem.tokens, "raw_chars": mem.raw_chars, "truncated": mem.truncated,
                  "elapsed_ms": round(mem.elapsed_ms), "context": mem.context, "requested_at_ts": cache.memory_ts_seen} if mem else {"status": "not_requested"}
        usage = {"answer": answer.usage() if answer else None, "judge": judge.usage() if judge else None}
        reward = None if acc is None else acc / 100.0
        return PM3Run(
            id=f"{run_id}:{item['query_id']}", task_id=item["query_id"], user_id=str(user_id), seq=seq, task_type=task_type,
            ts=int(item["ts"]), duration_ms=duration, status=status, error=error, metrics=metrics, accuracy_pct=acc,
            agent_response=resp, memory=memory, usage=usage,
            reward_info=None if reward is None else RewardInfo(reward=reward, info={"accuracy_pct": acc, "status": status}),
            seed=seed,
        )


def _is_infrastructure_error(exc: BaseException) -> bool:
    """模型 / 网关请求在重试后仍失败:不是题型逻辑的问题,交给批处理重试。"""
    from ...retry import CallError

    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (CallError, MemoryUnavailable)):
            return True
        exc = exc.__cause__ or exc.__context__
    return False
