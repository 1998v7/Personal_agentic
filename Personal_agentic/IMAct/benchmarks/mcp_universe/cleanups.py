"""任务结束后的清理:把 agent 在真实服务里创建的东西(GitHub 仓库、Notion 页面)删掉。

对应上游的 ``Task.reset`` 与 ``benchmark/cleanups.py``。每个 task 的 ``cleanups`` 说明"哪个工具创建了资源、用哪个清理函数、
参数从那次调用的参数或返回值里取":

    {"server": "github", "tool": "create_repository", "cleanup_func": "delete_repository", "cleanup_args": {"repo": "$name"}}

``$name`` 取自那次工具调用的参数 ``name``;``$return -> get(content) -> array(0) -> get(text) -> json -> get(id)``
对那次调用的返回值依次做 get / array / json 操作。清理按调用时间从晚到早进行;清理失败只记日志,不影响 task 的得分。

清理直接调用 GitHub / Notion 的 REST API,用环境变量里的令牌。本机没有 ``NOTION_API_KEY`` 时,Notion 页面改经网关的 Notion 服务归档;
GitHub 没有这个退路(网关的 GitHub 服务不能删仓库),缺 ``GITHUB_PERSONAL_ACCESS_TOKEN`` 时清理失败并在结果里明说。**GitHub 的删除只会作用于
``GITHUB_PERSONAL_ACCOUNT_NAME`` 这个账号下的仓库**,所以请使用专用的测试账号。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from typing import Any, Awaitable, Callable

import httpx

from ...evaluation.mcp_universe import Context
from ...harness.mcp import GatewayConfig, call_once

logger = logging.getLogger("imact.mcp_universe.cleanup")

CleanupFunc = Callable[..., Awaitable[str]]
CLEANUP_FUNCTIONS: dict[tuple[str, str], CleanupFunc] = {}


def cleanup_func(server_name: str, name: str):
    def _decorator(func: CleanupFunc) -> CleanupFunc:
        assert (server_name, name) not in CLEANUP_FUNCTIONS, f"Duplicated cleanup function ({server_name}, {name})"
        CLEANUP_FUNCTIONS[(server_name, name)] = func
        return func

    return _decorator


async def _with_retries(request: Callable[[], Awaitable[httpx.Response]], ok: int, what: str, not_found: str, denied: str,
                        max_retries: int = 3) -> str:
    delay = 1.0
    for _ in range(max_retries):
        response = await request()
        if response.status_code == ok:
            return what
        if response.status_code == 403:
            raise RuntimeError(denied)
        if response.status_code == 404:
            raise RuntimeError(not_found)
        await asyncio.sleep(delay)
        delay *= random.uniform(1, 1.5)
    raise RuntimeError("Reached the max retries")


@cleanup_func("github", "delete_repository")
async def github_delete_repository(repo: str, owner: str = "", **kwargs: Any) -> str:
    """https://docs.github.com/en/rest/repos/repos#delete-a-repository"""
    context: Context = kwargs.get("context") or Context()
    owner = owner or context.get_env("GITHUB_PERSONAL_ACCOUNT_NAME")
    if not owner:
        raise ValueError("Repository owner is empty")
    if not context.get_env("GITHUB_PERSONAL_ACCESS_TOKEN"):
        raise RuntimeError(f"没有 GITHUB_PERSONAL_ACCESS_TOKEN,仓库 {owner}/{repo} 没有被删除,请手动删除")
    if not repo or "/" in repo:
        raise ValueError(f"Refusing to delete repository with an unexpected name: {repo!r}")
    headers = {
        "Authorization": f"Bearer {context.get_env('GITHUB_PERSONAL_ACCESS_TOKEN')}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "MCPUniverse",
    }
    async with httpx.AsyncClient(timeout=float(kwargs.get("timeout", 30))) as client:
        return await _with_retries(
            lambda: client.delete(f"https://api.github.com/repos/{owner}/{repo}", headers=headers), 204,
            f"Repository {owner}/{repo} has been successfully deleted", f"Repository {owner}/{repo} not found.",
            f"Permission denied. You may not have delete permissions for {owner}/{repo}.", int(kwargs.get("max_retries", 3)))


@cleanup_func("notion", "delete_page")
async def notion_delete_page(page: str, _owner: str = "", **kwargs: Any) -> str:
    """https://developers.notion.com/reference/archive-a-page"""
    context: Context = kwargs.get("context") or Context()
    if not context.get_env("NOTION_API_KEY"):
        # 本机没有 Notion 密钥(密钥在网关那一侧):经网关的 Notion 服务把页面移入回收站
        gateway = context.gateway or GatewayConfig.from_env({**os.environ, **context.env})
        result = await call_once(gateway, "notion", "API-patch-page", {"page_id": page, "in_trash": True})
        if result.is_error:
            raise RuntimeError(f"网关的 Notion 服务没能归档页面 {page}: {(result.first_text or '')[:200]}")
        return f"Page {page} has been moved to trash (via gateway)"
    headers = {
        "Authorization": f"Bearer {context.get_env('NOTION_API_KEY')}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    async with httpx.AsyncClient(timeout=float(kwargs.get("timeout", 30))) as client:
        return await _with_retries(
            lambda: client.patch(f"https://api.notion.com/v1/pages/{page}", headers=headers, content='{"in_trash": true}'), 200,
            f"Page {page} has been successfully moved to trash", f"Page {page} not found.",
            f"Permission denied. You may not have delete permissions for {page}.", int(kwargs.get("max_retries", 3)))


def parse_cleanup_args(cleanup_args: Any, tool_call: dict[str, Any] | None) -> Any:
    """把 ``$name`` / ``$return -> ...`` 这样的取值写法按那次工具调用解析成具体值(与上游一致)。"""
    if isinstance(cleanup_args, dict):
        return {key: parse_cleanup_args(value, tool_call) for key, value in cleanup_args.items()}
    if isinstance(cleanup_args, list):
        return [parse_cleanup_args(arg, tool_call) for arg in cleanup_args]
    if not isinstance(cleanup_args, str):
        return cleanup_args
    arg = cleanup_args.strip()
    if "$" not in arg or not tool_call:
        return cleanup_args
    if arg[0] != "$":
        raise ValueError(f"Format error in `cleanup_args`: {arg}")
    items = [f.strip() for f in arg.split("->") if f.strip()]
    source = items[0][1:]
    if source != "return" and source not in tool_call["arguments"]:
        raise ValueError(f"Invalid input argument in `cleanup_args`: {source}")
    value = tool_call["response"] if source == "return" else tool_call["arguments"][source]
    for item in items[1:]:
        name = item.split("(")[0].strip().lower()
        match = re.search(r"\((.*?)\)", item)
        operand = match.group(1).strip() if match else ""
        if name == "get":
            value = value[operand]
        elif name == "array":
            value = value[int(operand)]
        elif name == "json":
            value = json.loads(value)
        else:
            raise ValueError(f"Invalid operation in `cleanup_args`: {name}")
    if isinstance(value, str) and "->" in value:
        logger.warning("The parsed `cleanup_args` contains `->`: %s", value)
    return value


def call_created_nothing(call: dict[str, Any]) -> str | None:
    """这次工具调用明显没有成功(没有可清理的资源)时返回原因,否则 None。

    调用本身抛错,或 MCP 结果标了 ``isError``,或结果正文是服务端的错误体(``{"object": "error", ...}``,Notion 的 400 就是这样回的)。
    """
    if call.get("error"):
        return f"调用出错: {str(call['error'])[:100]}"
    response = call.get("response")
    if isinstance(response, dict):
        if response.get("isError"):
            return "工具返回了错误"
        content = response.get("content") or []
        text = content[0].get("text") if content and isinstance(content[0], dict) else None
        if isinstance(text, str) and text.lstrip().startswith("{"):
            try:
                body = json.loads(text)
            except ValueError:
                return None
            if isinstance(body, dict) and body.get("object") == "error":
                return f"服务端返回错误: {body.get('code') or body.get('message') or ''}"[:120]
    return None


async def run_cleanups(cleanups: list[dict[str, Any]], tool_calls: list[dict[str, Any]], context: Context) -> list[str]:
    """按 task 的 ``cleanups`` 清理 agent 创建的资源。``tool_calls`` 是 agent 实际执行过的调用,按发生顺序,
    每项 ``{"server", "tool_name", "arguments", "response", "error"}``。返回每次清理的结果说明(成功或失败原因)。"""
    outcomes: list[str] = []

    async def execute(config: dict[str, Any], call: dict[str, Any] | None) -> None:
        label = f"{config.get('server')}.{config.get('cleanup_func')}"
        if call is not None and (why := call_created_nothing(call)):
            outcomes.append(f"{label}: SKIPPED (这次 {call.get('tool_name')} 调用没有建出资源: {why})")
            return
        try:
            args = parse_cleanup_args(config.get("cleanup_args", {}), call)
            func = CLEANUP_FUNCTIONS.get((config.get("server", ""), config.get("cleanup_func", "")))
            if func is None:
                raise RuntimeError(f"没有清理函数 {label}")
            outcomes.append(f"{label}: {await func(**args, context=context)}")
        except Exception as exc:  # noqa: BLE001 - 清理失败不影响得分,只记录
            logger.error("清理失败 %s: %s", label, exc)
            outcomes.append(f"{label}: FAILED {exc}")

    for config in cleanups:
        if not any(c.get("server") == config.get("server") and (c.get("tool_name") == config.get("tool") or config.get("tool", "") == "")
                   for c in tool_calls):
            logger.warning("No tool %s of server %s executed", config.get("tool"), config.get("server"))
    for call in sorted(tool_calls, key=lambda c: c["seq"], reverse=True):
        for config in cleanups:
            if call.get("server") == config.get("server") and call.get("tool_name") == config.get("tool"):
                await execute(config, call)
    for config in cleanups:
        if config.get("tool", "") == "" and config.get("server"):
            await execute(config, None)
    return outcomes
