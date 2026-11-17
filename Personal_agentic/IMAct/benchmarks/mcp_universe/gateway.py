"""评估器里要查询真实服务(GitHub、Google Maps、Notion …)时用的入口,接口形状与上游 ``MCPManager.execute`` 一致,
这样 ``checks/`` 里从上游移植的评估函数可以保持原样。底层是 ``IMAct.harness.mcp`` 的通用 MCP 客户端,走 MCP 网关。

两点与上游不同:
* 每次调用的基础设施故障(连不上网关、超时)除了照常抛出,还记到 ``context.infra_errors``。
  上游的评估函数大多把这类异常吞掉当作"没找到",结果是网关挂了和 agent 答错分不开;
  runner 据此把这个 task 标为评估错误,而不是静悄悄记成 0 分。
* 工具自己返回的错误(``isError``)是正常结果,照旧交给评估函数判断。
"""

from __future__ import annotations

import os
from typing import Any

from ...evaluation.mcp_universe import Context
from ...harness.mcp import GatewayConfig, McpError, McpResult, call_once


class _View:
    """让 MCP 结果的 JSON 形式可以像 SDK 的对象那样用属性访问(``out.content[1].resource.text``),缺字段时抛 ``AttributeError``。"""

    def __init__(self, data: dict[str, Any]):
        self.__dict__.update({k: _wrap(v) for k, v in data.items()})


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return _View(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


class ToolOutput(_View):
    """工具调用结果:``isError``、``content``。"""

    def __init__(self, result: McpResult):
        super().__init__({"isError": result.is_error, "content": result.content})
        self.raw = result.raw


def default_transport() -> str:
    return "sse"  # 评估器只经网关访问服务


class MCPManager:
    def __init__(self, context: Context | None = None):
        self._context = context if context is not None else Context()

    async def execute(self, server_name: str, tool_name: str, arguments: dict[str, Any], transport: str | None = None, permissions: Any = None) -> ToolOutput:
        gateway = self._context.gateway or GatewayConfig.from_env({**os.environ, **self._context.env})
        try:
            return ToolOutput(await call_once(gateway, server_name, tool_name, arguments))
        except McpError as exc:
            self._context.infra_errors.append(f"{server_name}.{tool_name}: {exc}")
            raise
