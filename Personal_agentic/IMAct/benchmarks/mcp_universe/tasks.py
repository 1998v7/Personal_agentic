"""MCP-Universe 的任务与领域。任务是 ``task_data/<目录>/<名字>.json``,领域(含 agent 设置与 task 顺序)在 ``domains.yaml``。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from ...evaluation.mcp_universe import Context, render_env

ROOT = Path(__file__).parent
TASKS = ROOT / "task_data"


@dataclass(frozen=True)
class Domain:
    name: str
    instruction: str
    max_iterations: int
    summarize_tool_response: bool
    task_paths: tuple[str, ...]  # 相对 task_data/,如 ``web_search/info_search_task_0001.json``


@dataclass
class McpTask:
    id: str  # 文件名(不含 .json);同一个 task 只属于一个文件,不会重名
    path: str
    domain: str
    category: str
    question: str  # 已渲染 {{ENV_VAR}}
    output_format: dict | None
    mcp_servers: list[dict[str, Any]]
    evaluators: list[dict[str, Any]]
    cleanups: list[dict[str, Any]] = field(default_factory=list)
    use_specified_server: bool = False

    @property
    def has_side_effects(self) -> bool:
        """会在真实服务里创建东西(GitHub 仓库、Notion 页面),需要事后清理的 task。"""
        return bool(self.cleanups)


@lru_cache(maxsize=1)
def _domains_file() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "domains.yaml").read_text(encoding="utf-8"))


def domain_names() -> list[str]:
    return list(_domains_file())


def load_domain(name: str) -> Domain:
    raw = _domains_file()
    if name not in raw:
        raise ValueError(f"不支持的领域 {name!r};可选: {', '.join(raw)}")
    d = raw[name]
    return Domain(name=name, instruction=d["instruction"], max_iterations=int(d["max_iterations"]),
                  summarize_tool_response=bool(d["summarize_tool_response"]), task_paths=tuple(d["tasks"]))


def load_task(path: str, domain: str, env: Mapping[str, str] | None = None) -> McpTask:
    """读一个任务,把问题文本里的 ``{{ENV_VAR}}`` 渲染成环境变量(没定义的原样保留)。"""
    raw = json.loads((TASKS / path).read_text(encoding="utf-8"))
    context = Context(env)
    return McpTask(
        id=Path(path).stem,
        path=path,
        domain=domain,
        category=raw.get("category", "general"),
        question=render_env(raw.get("question", ""), context),
        output_format=raw.get("output_format") or None,
        mcp_servers=list(raw.get("mcp_servers", [])),
        evaluators=list(raw.get("evaluators", [])),
        cleanups=list(raw.get("cleanups", [])),
        use_specified_server=bool(raw.get("use_specified_server", False)),
    )


def load_tasks(domain: str, env: Mapping[str, str] | None = None) -> list[McpTask]:
    return [load_task(p, domain, env) for p in load_domain(domain).task_paths]


def select(tasks: list[McpTask], task_ids: list[str] | None = None, num_tasks: int | None = None) -> list[McpTask]:
    if task_ids is not None:
        known = {t.id for t in tasks}
        unknown = [i for i in task_ids if i not in known]
        if unknown:
            raise ValueError(f"这个领域里没有这些 task: {', '.join(unknown)}")
        wanted = set(task_ids)
        tasks = [t for t in tasks if t.id in wanted]
    if num_tasks is not None:
        if num_tasks < 1:
            raise ValueError("--num-tasks 必须 >= 1")
        tasks = tasks[:num_tasks]
    return tasks
