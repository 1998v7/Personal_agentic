"""MCP-Universe 的 agent prompt:工具说明、输出格式、工具回执摘要、ReAct 系统 prompt。模板文本取自上游,逐字不变。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml
from jinja2 import Environment

from ...harness.mcp import McpTool

TEMPLATES = Path(__file__).with_name("prompt_templates")

OUTPUT_FORMAT_PROMPT = """
The final answer should follow this JSON format:
{output_format}

You must respond with valid JSON only, with no triple backticks. No markdown formatting.
No extra text. Do not wrap in ```json code fences. Property names must be enclosed in double quotes.
""".strip()

TOOL_RESPONSE_SUMMARIZER_PROMPT = """
Extract all information from the tool response that is relevant to the context.

Tool Call Context:
{context}

Tool Response:
{tool_response}

Directly output the extracted information. Try to maintain the original response structure. Use fewer than 500 words.
""".strip()


@lru_cache(maxsize=None)
def _template(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8")


def tools_description(tools: dict[str, list[McpTool]]) -> str:
    """每个工具的 server、名称、说明和参数(参数用 YAML 展开)。"""
    descriptions = []
    for server_name, tool_list in tools.items():
        for tool in tool_list:
            args = []
            schema = tool.input_schema
            if "properties" in schema:
                for param_name, param_info in schema["properties"].items():
                    info = "\n".join("    " + line for line in yaml.dump(param_info, sort_keys=False, indent=2).split("\n"))
                    arg = f"- {param_name}:\n{info}".strip()
                    if param_name in schema.get("required", []):
                        arg += "\n    required: true"
                    args.append(arg.strip())
            lines = [line for line in tool.description.split("\n") if line.strip()]
            arguments = f"\n{chr(10).join(args)}" if args else " No arguments"
            descriptions.append(
                f"Server: {server_name}\nTool: {tool.name}\nDescription:\n{chr(10).join(lines)}\nArguments:{arguments}"
            )
    return "\n\n".join(descriptions).strip()


def output_format_prompt(output_format: dict | str | None) -> str:
    if output_format is None:
        return ""
    if isinstance(output_format, dict):
        return OUTPUT_FORMAT_PROMPT.format(output_format=json.dumps(output_format, indent=2)).strip()
    return output_format.strip()


def render_react_prompt(instruction: str, question: str, max_steps: int, history: list[str], tools: dict[str, list[McpTool]]) -> str:
    """每一轮发给模型的完整 prompt:角色说明 + 工具说明 + 问题 + 到目前为止的历史。"""
    params = {"INSTRUCTION": instruction, "QUESTION": question, "MAX_STEPS": max_steps}
    if history:
        params["HISTORY"] = "\n\n".join(history)
    description = tools_description(tools) if tools else ""
    tools_prompt = ""
    if description:
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        tools_prompt = env.from_string(_template("tools_prompt.j2")).render(**params, TOOLS_DESCRIPTION=description)
    if tools_prompt:
        params["TOOLS_PROMPT"] = tools_prompt
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(_template("react_prompt.j2")).render(**params).strip()


def question_with_format(question: str, output_format: dict | None) -> str:
    """问题后面接上"最终答案的 JSON 格式"要求(任务给了 output_format 时)。"""
    if output_format is not None:
        return question + "\n\n" + output_format_prompt(output_format)
    return question
