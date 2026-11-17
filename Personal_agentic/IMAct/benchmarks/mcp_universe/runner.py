"""MCP-Universe 一个 task 的完整流程:连上它指定的 MCP server → ReAct agent 作答 → 逐条评估 → 清理创建的资源。

被测对象是 **agent 模型本身**(经 ReAct 文本协议,prompt 与上游一致),不涉及 Memory 与执行层。
工具与评分都经 MCP 网关访问真实服务,所以结果会随外部服务的状态变化。
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Mapping

from pydantic import BaseModel, Field

from ...evaluation.base import RewardInfo
from ...evaluation.mcp_universe import Context, EvaluationResult, Evaluator
from ...exec_layer.bridge import AsyncBridge
from ...harness.llm import LLMCaller
from ...harness.mcp import GatewayConfig, McpToolset
from ...harness.messages import now_str
from ...harness.react import ReActAgent, ReActResult
from . import checks  # noqa: F401  向评分引擎登记各领域的比较函数
from .cleanups import run_cleanups
from .prompts import TOOL_RESPONSE_SUMMARIZER_PROMPT, question_with_format, render_react_prompt
from .tasks import Domain, McpTask

JUDGE_OP = "google_search.llm_as_a_judge"
DEFAULT_EVAL_TIMEOUT = 900.0


class McpRun(BaseModel):
    """一个 task 的运行记录(落盘成结果文件)。字段 ``id`` / ``task_id`` / ``reward_info`` 是批处理框架需要的。"""

    id: str
    task_id: str
    domain: str
    timestamp: str = Field(default_factory=now_str)
    start_time: str
    end_time: str
    duration: float
    question: str
    response: str
    answered: bool
    iterations: int
    usage: dict[str, int]
    steps: list[dict[str, Any]]
    history: list[str]
    tool_calls: list[dict[str, Any]]
    evaluations: list[EvaluationResult]
    cleanups: list[str]
    infra_errors: list[str]
    reward_info: RewardInfo
    trial: int | None = None
    seed: int | None = None


def needs_judge(tasks: list[McpTask]) -> bool:
    return any(e.get("op") == JUDGE_OP for t in tasks for e in t.evaluators)


class McpUniverseRunner:
    def __init__(
        self,
        domain: Domain,
        agent_llm: LLMCaller,
        bridge: AsyncBridge,
        gateway: GatewayConfig,
        *,
        judge: LLMCaller | None = None,
        env: Mapping[str, str] | None = None,
        side_effects: bool = True,
        eval_timeout: float = DEFAULT_EVAL_TIMEOUT,
        call_timeout: float = 60.0,
    ):
        self.domain = domain
        self.agent_llm = agent_llm
        self.bridge = bridge
        self.gateway = gateway
        self.judge = judge
        self.env = dict(env or {})
        self.side_effects = side_effects
        self.eval_timeout = eval_timeout
        self.call_timeout = call_timeout
        # 会在真实服务里创建资源的 task 一次只跑一个:它们可能用同样的仓库名 / 页面名,并发会互相踩。
        self._side_effect_lock = threading.Lock()

    # ------------------------------------------------------------------ 批处理入口
    def run_task(self, task: McpTask, *, run_id: str, seed: int | None = None) -> McpRun:
        if task.has_side_effects:
            if not self.side_effects:
                raise RuntimeError(f"{task.id} 会在真实服务里创建资源,但这次运行不允许(--skip-side-effects)")
            with self._side_effect_lock:
                return self._run(task, run_id=run_id, seed=seed)
        return self._run(task, run_id=run_id, seed=seed)

    def _run(self, task: McpTask, *, run_id: str, seed: int | None) -> McpRun:
        if not task.use_specified_server or not task.mcp_servers:
            raise ValueError(f"{task.id}: task 没有指定 MCP server")
        started = time.time()
        context = Context(self.env, gateway=self.gateway, judge=self.judge)
        run = ReActResult(response="", answered=False, iterations=0)
        evaluations: list[EvaluationResult] = []
        cleanup_outcomes: list[str] = []
        toolset = McpToolset(self.gateway, task.mcp_servers, self.bridge, call_timeout=self.call_timeout)
        try:
            toolset.open()
            agent = ReActAgent(
                self.agent_llm, toolset,
                build_prompt=lambda q, h: render_react_prompt(self.domain.instruction, q, self.domain.max_iterations, h, toolset.tools),
                summary_prompt=lambda c, r: TOOL_RESPONSE_SUMMARIZER_PROMPT.format(context=c, tool_response=r),
                max_iterations=self.domain.max_iterations, summarize_tool_response=self.domain.summarize_tool_response)
            agent.run(question_with_format(task.question, task.output_format), result=run)
            evaluations = self.bridge.run(self._evaluate(task, run.response, context))
        finally:
            toolset.close()
            if task.cleanups:
                cleanup_outcomes = self.bridge.run(run_cleanups(task.cleanups, run.tool_calls, context))
        passed = sum(1 for e in evaluations if e.passed)
        success = bool(evaluations) and passed == len(evaluations)
        info = {
            "success": success, "evaluator_score": passed / len(evaluations) if evaluations else 0.0,
            "evaluations_passed": passed, "evaluations_total": len(evaluations),
            "iterations": run.iterations, "llm_calls": run.usage["llm_calls"], "num_tool_calls": len(run.tool_calls), "answered": run.answered,
            "evaluation_error": bool(context.infra_errors), "infra_errors": list(context.infra_errors),
        }
        ended = time.time()
        return McpRun(
            id=f"{run_id}:{task.id}", task_id=task.id, domain=task.domain,
            start_time=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)), end_time=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ended)),
            duration=ended - started, question=task.question, response=run.response, answered=run.answered, iterations=run.iterations, usage=run.usage,
            steps=[{k: v for k, v in s.__dict__.items() if k != "tool_result"} for s in run.steps],  # 工具返回的全文已在 history 里
            history=run.history,
            tool_calls=[{k: v for k, v in c.items() if k != "response"} for c in run.tool_calls],
            evaluations=evaluations, cleanups=cleanup_outcomes, infra_errors=list(context.infra_errors),
            reward_info=RewardInfo(reward=1.0 if success else 0.0, info=info), seed=seed)

    async def _evaluate(self, task: McpTask, response: str, context: Context) -> list[EvaluationResult]:
        """逐条评估(按任务给定的顺序,串行:有的评估依赖外部状态)。单条评估超时按"出错"处理,记为没通过。"""
        results: list[EvaluationResult] = []
        for config in task.evaluators:
            evaluator = Evaluator(config, context)
            try:
                results.append(await asyncio.wait_for(evaluator.evaluate(response), timeout=self.eval_timeout))
            except asyncio.TimeoutError:
                context.infra_errors.append(f"evaluator {config.get('op') or config.get('func')}: timeout after {self.eval_timeout}s")
                results.append(EvaluationResult(config=evaluator.config, response=response, passed=False, reason="Execution error",
                                                error=f"evaluation timed out after {self.eval_timeout}s"))
        return results
