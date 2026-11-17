"""ReAct agent:模型每轮输出一段 JSON(``thought`` + ``action`` 或 ``answer``),文本里直接给出要调的工具,没有原生 function calling。

循环、历史格式与错误处理对齐上游 MCP-Universe 的 ``ReAct``:

* 每轮把"系统 prompt + 到目前为止的全部历史"作为**一条** user 消息发给模型,模型回一段 JSON;
* ``answer`` 结束;``action`` 调一次工具,把结果写进历史;``result`` 原样写进历史;
* JSON 解析失败、格式不对、动作不合法、工具调用出错都不终止,只是在历史里记一笔,让模型下一轮自己修正;
* 用满 ``max_iterations`` 还没有 ``answer``,返回固定的道歉文案。

具体的 prompt 文本属于 benchmark,由调用方通过 ``Protocol`` 的两个函数提供;工具来自 ``McpToolset``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .llm import LLMCaller
from .mcp import McpError, McpResult, McpTool
from .messages import UserMessage

GIVE_UP_RESPONSE = "I'm sorry, but I couldn't find a satisfactory answer within the allowed number of iterations."
PARSE_ERROR_NOTE = "I encountered an error in parsing LLM response. Let me try again."
UNEXPECTED_ERROR_NOTE = "I encountered an unexpected error. Let me try a different approach."
RESULT_ERROR_LIMIT = 300  # 工具出错时写进历史的错误文字长度上限


class Toolset(Protocol):
    """agent 能用的工具:按 server 分组,以及调用入口。"""

    tools: dict[str, list[McpTool]]

    def call(self, server: str, tool: str, arguments: dict[str, Any]) -> McpResult: ...


@dataclass
class ReActStep:
    """一轮的记录,供事后排查。"""

    iteration: int
    llm_response: str
    kind: str  # answer / action / invalid_action / result / parse_error / error
    thought: str | None = None
    action: Any = None
    tool_result: str | None = None
    note: str | None = None


@dataclass
class ReActResult:
    response: str  # 交给评估器的最终回答(字符串)
    answered: bool  # 模型是否在步数内给出了 answer
    iterations: int  # 实际用了几轮
    steps: list[ReActStep] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    # 实际执行过的工具调用,按发生顺序:{"seq", "server", "tool_name", "arguments", "response"(MCP 结果的 JSON 形式,出错时为 ""), "error"}。
    # 清理阶段据此找出 agent 创建了哪些资源。
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    first_prompt: str = ""
    usage: dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0})


def answer_to_str(answer: Any) -> str:
    """把模型给的 ``answer`` 变成评估器收到的字符串。字符串原样;dict 序列化成 JSON;其它用 ``str()``(与上游一致)。"""
    if isinstance(answer, dict):
        return json.dumps(answer)
    return str(answer)


def _strip_json(text: str) -> str:
    text = text.strip().strip("`").strip()
    if text.startswith("json"):
        text = text[4:].strip()
    return text


class ReActAgent:
    def __init__(
        self,
        llm: LLMCaller,
        toolset: Toolset,
        *,
        build_prompt: Callable[[str, list[str]], str],
        summary_prompt: Callable[[str, str], str] | None = None,
        max_iterations: int = 5,
        summarize_tool_response: bool = False,
    ):
        """``build_prompt(question, history)`` 给出完整的系统 prompt(含工具说明与历史);
        ``summary_prompt(context, tool_response)`` 给出工具回执摘要的 prompt(``summarize_tool_response`` 为真时需要)。"""
        if summarize_tool_response and summary_prompt is None:
            raise ValueError("summarize_tool_response 需要提供 summary_prompt")
        self.llm = llm
        self.toolset = toolset
        self.build_prompt = build_prompt
        self.summary_prompt = summary_prompt
        self.max_iterations = max_iterations
        self.summarize_tool_response = summarize_tool_response

    # ------------------------------------------------------------------ 一次完整运行
    def run(self, question: str, result: ReActResult | None = None) -> ReActResult:
        """跑一次。调用方可以传入 ``result``:模型请求中途失败抛出时,已执行的工具调用仍留在里面(清理需要)。"""
        history: list[str] = []
        if result is None:
            result = ReActResult(response=GIVE_UP_RESPONSE, answered=False, iterations=0)
        result.response, result.answered = GIVE_UP_RESPONSE, False  # 传入的 result 也一样:没等到 answer 就是这句
        result.history = history

        def add(kind: str, message: Any) -> None:
            history.append(f"{kind.title()}: {message}")

        for iter_num in range(self.max_iterations):
            result.iterations = iter_num + 1
            prompt = self.build_prompt(question, history)
            if iter_num == 0:
                result.first_prompt = prompt
            raw = self._ask(prompt, result)
            step = ReActStep(iteration=iter_num + 1, llm_response=raw, kind="error")
            result.steps.append(step)
            try:
                parsed = json.loads(_strip_json(raw))
                if "thought" not in parsed:
                    raise ValueError("Invalid response format")
                add(f"Step {iter_num + 1}", "")
                step.thought = parsed["thought"]
                if "answer" in parsed:
                    add("answer", parsed["answer"])
                    step.kind = "answer"
                    result.response, result.answered = answer_to_str(parsed["answer"]), True
                    return result
                if "action" in parsed:
                    add("thought", parsed["thought"])
                    self._act(parsed["action"], add, step, result)
                elif "result" in parsed:
                    add("thought", parsed["thought"])
                    add("result", parsed["result"])
                    step.kind = "result"
                else:
                    raise ValueError("Invalid response format")
            except json.JSONDecodeError as exc:
                add("error", PARSE_ERROR_NOTE)
                step.kind, step.note = "parse_error", f"Failed to parse response: {exc}"
            except Exception as exc:  # noqa: BLE001 - 与上游一致:模型输出的任何毛病都不终止,记一笔让它重来
                add("error", UNEXPECTED_ERROR_NOTE)
                step.kind, step.note = "error", f"Failed to process response: {exc}"
        return result

    # ------------------------------------------------------------------ 内部
    def _ask(self, prompt: str, result: ReActResult) -> str:
        """发一次模型请求并计入用量。思考与工具回执摘要都走这里,所以 ``usage["llm_calls"]`` 是全部模型调用次数。"""
        reply = self.llm.chat([UserMessage(role="user", content=prompt)])
        result.usage["llm_calls"] += 1
        for key in ("prompt_tokens", "completion_tokens"):
            result.usage[key] += int((reply.usage or {}).get(key, 0) or 0)
        return reply.content or ""

    def _act(self, action: Any, add: Callable[[str, Any], None], step: ReActStep, run: ReActResult) -> None:
        step.action = action
        if not isinstance(action, dict) or "server" not in action or "tool" not in action:
            add("action", str(action))
            add("result", "Invalid action")
            step.kind = "invalid_action"
            return
        step.kind = "action"
        add("action", f"Using tool `{action['tool']}` in server `{action['server']}`")
        add("action input", str(action.get("arguments", "none")))
        try:
            result = self._call_tool(action, run)
            first = result.content[0]  # 空内容会 IndexError,和上游一样落到下面的 except
            if first.get("type") != "text":
                raise ValueError("Tool output is not a text")
            text = first.get("text") or ""
            step.tool_result = text
            if self.summarize_tool_response:
                summary = self._ask(self.summary_prompt(json.dumps(action, indent=2), text), run)
                add("result", summary)
            else:
                add("result", text)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)[:RESULT_ERROR_LIMIT]
            step.note = str(exc)
            add("result", message)

    def _call_tool(self, call: dict[str, Any], run: ReActResult) -> McpResult:
        """校验并执行一次工具调用,错误文字与上游一致(模型会在历史里看到它们)。通过校验、真正发出的调用才会记入 ``run.tool_calls``。"""
        if "server" in call and "tool" in call and "arguments" in call:
            server, tool = call["server"], call["tool"]
            if server not in self.toolset.tools:
                raise RuntimeError(f"Not found server {server}")
            for candidate in self.toolset.tools[server]:
                if candidate.name != tool:
                    continue
                record = {"seq": len(run.tool_calls), "server": server, "tool_name": tool, "arguments": call["arguments"], "response": "", "error": ""}
                run.tool_calls.append(record)
                try:
                    result = self.toolset.call(server, tool, call["arguments"])
                except Exception as exc:  # noqa: BLE001
                    record["error"] = str(exc)
                    raise RuntimeError(f"Error occurred during executing tool {tool}") from exc
                record["response"] = result.model_dump()
                return result
            raise RuntimeError(f"Server {server} has no tool {tool}")
        raise RuntimeError("The input of `call_tool` function has a wrong format")
