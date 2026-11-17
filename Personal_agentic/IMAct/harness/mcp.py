"""通用的 MCP 客户端:连接一个 MCP server,列工具、调工具。与具体 benchmark 无关。

两层接口:

* ``McpConnection``:异步上下文管理器,一次连接。适合"连一下、调一次、关掉"(评估器查询真实服务)。
* ``McpClient``:同步句柄,长连接放在 ``AsyncBridge`` 的事件循环里,供同步的 agent 循环使用。

MCP SDK 的传输层在 anyio 的 task group 里运行,进入与退出必须在同一个 asyncio task 里。
``McpClient`` 因此让一个常驻 task 持有整个连接的生命周期,其它线程只通过 ``run_coroutine_threadsafe`` 提交调用。

只支持经网关的 SSE 传输(``<网关>/<server 名>/sse``,可带 Bearer token)。
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping

from ..exec_layer.bridge import AsyncBridge

logger = logging.getLogger("imact.mcp")

DEFAULT_CALL_TIMEOUT = 60.0
DEFAULT_CONNECT_RETRIES = 5
DEFAULT_CONNECT_RETRY_DELAY = 2.0
CALL_RETRIES = 5
CALL_RETRY_DELAY = 1.0


class McpError(RuntimeError):
    """连接或调用 MCP server 失败(基础设施层面,不是工具自己返回的错误结果)。"""


@dataclass(frozen=True)
class GatewayConfig:
    """网关地址与鉴权。环境变量 ``MCP_GATEWAY_ADDRESS`` / ``MCP_GATEWAY_TOKEN``。"""

    address: str
    token: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "GatewayConfig":
        env = os.environ if env is None else env
        address = (env.get("MCP_GATEWAY_ADDRESS") or "").strip()
        if not address:
            raise McpError("MCP_GATEWAY_ADDRESS 未设置:在 .env 里填 MCP 网关地址(以及 MCP_GATEWAY_TOKEN)")
        return cls(address=address.rstrip("/"), token=(env.get("MCP_GATEWAY_TOKEN") or "").strip())

    def url(self, server: str) -> str:
        return f"{self.address}/{server}/sse"

    @property
    def headers(self) -> dict[str, str] | None:
        return {"Authorization": f"Bearer {self.token}"} if self.token else None


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class McpResult:
    """一次工具调用的结果。``raw`` 是 SDK 结果的 JSON 形式(``content`` / ``isError`` / ``structuredContent``…)。"""

    raw: dict[str, Any]

    @property
    def is_error(self) -> bool:
        return bool(self.raw.get("isError"))

    @property
    def content(self) -> list[dict[str, Any]]:
        return list(self.raw.get("content") or [])

    @property
    def first_text(self) -> str | None:
        """第一个内容块的文本;第一块不是文本时为 ``None``。"""
        blocks = self.content
        if blocks and blocks[0].get("type") == "text":
            return blocks[0].get("text") or ""
        return None

    def model_dump(self, mode: str = "json") -> dict[str, Any]:  # 与 SDK 的 CallToolResult 同名,评估器沿用
        return self.raw


def _result_from_sdk(result: Any) -> McpResult:
    if not hasattr(result, "model_dump"):
        raise McpError(f"工具调用返回了意外的结果类型: {type(result).__name__}")
    return McpResult(raw=result.model_dump(mode="json", by_alias=True, exclude_none=False))


def _tool_from_sdk(tool: Any) -> McpTool:
    schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}  # SDK 1.x / 2.x 字段名不同
    return McpTool(name=tool.name, description=tool.description or "", input_schema=dict(schema))


class McpConnection:
    """``async with McpConnection(...) as conn``:连接一个 server。"""

    def __init__(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        call_timeout: float = DEFAULT_CALL_TIMEOUT,
        connect_timeout: float = 20.0,
        connect_retries: int = DEFAULT_CONNECT_RETRIES,
        connect_retry_delay: float = DEFAULT_CONNECT_RETRY_DELAY,
    ):
        self.url = url
        self.headers = dict(headers) if headers else None
        self.call_timeout = call_timeout
        self.connect_timeout = connect_timeout
        self.connect_retries = connect_retries
        self.connect_retry_delay = connect_retry_delay
        self._stack: AsyncExitStack | None = None
        self._session: Any = None

    async def __aenter__(self) -> "McpConnection":
        from mcp import ClientSession  # 延迟导入:不用 MCP 的 benchmark 不需要装 mcp
        from mcp.client.sse import sse_client

        last: BaseException | None = None
        for attempt in range(self.connect_retries):
            stack = AsyncExitStack()
            try:
                streams = await stack.enter_async_context(
                    sse_client(self.url, headers=self.headers, timeout=self.connect_timeout, sse_read_timeout=max(1800.0, self.call_timeout * 2)))
                read, write = streams[0], streams[1]
                session = await stack.enter_async_context(
                    ClientSession(read, write, read_timeout_seconds=_read_timeout(max(60.0, self.call_timeout))))
                await asyncio.wait_for(session.initialize(), timeout=max(self.connect_timeout, 30.0))
                self._stack, self._session = stack, session
                return self
            except BaseException as exc:  # noqa: BLE001 - 含 ExceptionGroup;失败的尝试要收拾干净再重试
                last = exc
                try:
                    await stack.aclose()
                except BaseException:  # noqa: BLE001
                    pass
                if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                    raise
                if attempt + 1 < self.connect_retries:
                    await asyncio.sleep(self.connect_retry_delay * (1.5**attempt))
        raise McpError(f"连接 MCP server 失败 {self.url}: {_describe(last)}") from last

    async def __aexit__(self, *exc_info: Any) -> None:
        stack, self._stack, self._session = self._stack, None, None
        if stack is not None:
            try:
                await stack.aclose()
            except BaseException as exc:  # noqa: BLE001 - 关闭阶段的传输层噪声不应盖过真正的结果
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                logger.debug("关闭 MCP 连接 %s 时出错(忽略): %s", self.url, _describe(exc))

    async def list_tools(self) -> list[McpTool]:
        self._require_session()
        tools: list[McpTool] = []
        cursor = None
        while True:
            page = await self._session.list_tools() if cursor is None else await self._session.list_tools(params=_page(cursor))
            tools.extend(_tool_from_sdk(t) for t in page.tools)
            cursor = getattr(page, "nextCursor", None)
            if not cursor:
                return tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> McpResult:
        """调一次工具。普通异常重试;超时与连接错误不重试,直接抛 ``McpError``。"""
        self._require_session()
        last: BaseException | None = None
        for attempt in range(CALL_RETRIES):
            try:
                result = await asyncio.wait_for(self._session.call_tool(name, arguments or {}), timeout=self.call_timeout)
                return _result_from_sdk(result)
            except asyncio.TimeoutError as exc:
                raise McpError(f"Connection lost: timeout after {self.call_timeout}s calling {name}") from exc
            except McpError:
                raise
            except Exception as exc:  # noqa: BLE001
                last = exc
                if _is_connection_error(exc):
                    raise McpError(f"Connection lost: {_describe(exc)}") from exc
                logger.warning("调用工具 %s 失败(第 %d/%d 次): %s", name, attempt + 1, CALL_RETRIES, _describe(exc))
                if attempt + 1 < CALL_RETRIES:
                    await asyncio.sleep(CALL_RETRY_DELAY)
        raise McpError(f"调用工具 {name} 失败: {_describe(last)}") from last

    def _require_session(self) -> None:
        if self._session is None:
            raise McpError("MCP 连接未建立或已关闭")


async def call_once(gateway: GatewayConfig, server: str, tool: str, arguments: dict[str, Any], **options: Any) -> McpResult:
    """连一下、调一次、关掉。评估器查询真实服务用它。"""
    async with McpConnection(gateway.url(server), headers=gateway.headers, **options) as conn:
        return await conn.call_tool(tool, arguments)


class McpClient:
    """同步句柄:一个 server 的长连接。由 ``AsyncBridge`` 的事件循环承载。"""

    def __init__(self, name: str, url: str, bridge: AsyncBridge, *, headers: Mapping[str, str] | None = None, **options: Any):
        self.name = name
        self.url = url
        self.bridge = bridge
        self._conn = McpConnection(url, headers=headers, **options)
        self._ready: "asyncio.Future[None] | None" = None
        self._close: "asyncio.Event | None" = None
        self._owner: "asyncio.Task[None] | None" = None
        self._tools: list[McpTool] | None = None

    @classmethod
    def for_server(cls, gateway: GatewayConfig, server: str, bridge: AsyncBridge, **options: Any) -> "McpClient":
        return cls(server, gateway.url(server), bridge, headers=gateway.headers, **options)

    def connect(self) -> "McpClient":
        self.bridge.run(self._start())
        return self

    async def _start(self) -> None:
        loop = asyncio.get_running_loop()
        self._ready, self._close = loop.create_future(), asyncio.Event()
        self._owner = loop.create_task(self._own())
        await self._ready

    async def _own(self) -> None:
        try:
            async with self._conn:
                self._ready.set_result(None)
                await self._close.wait()
        except BaseException as exc:  # noqa: BLE001
            if not self._ready.done():
                self._ready.set_exception(exc if isinstance(exc, Exception) else McpError(str(exc)))
            elif not isinstance(exc, asyncio.CancelledError):
                logger.warning("MCP 连接 %s 异常结束: %s", self.name, _describe(exc))

    def list_tools(self) -> list[McpTool]:
        if self._tools is None:
            self._tools = self.bridge.run(self._conn.list_tools())
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> McpResult:
        return self.bridge.run(self._conn.call_tool(name, arguments))

    def close(self) -> None:
        if self._owner is None:
            return
        owner, self._owner = self._owner, None

        async def _stop() -> None:
            self._close.set()
            try:
                await asyncio.wait_for(asyncio.shield(owner), timeout=10)
            except BaseException:  # noqa: BLE001
                owner.cancel()

        try:
            self.bridge.run(_stop())
        except Exception as exc:  # noqa: BLE001
            logger.debug("关闭 MCP 客户端 %s 出错(忽略): %s", self.name, exc)

    def __enter__(self) -> "McpClient":
        return self.connect()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class McpToolset:
    """一组 server 的长连接和它们的工具列表,给 agent 用。

    ``servers`` 每项是 ``{"name": ..., "tools": [可选,只暴露这些工具]}``。连接并行建立,任何一个失败就全部关闭并抛出。
    """

    def __init__(self, gateway: GatewayConfig, servers: list[dict[str, Any]], bridge: AsyncBridge, **options: Any):
        self.gateway = gateway
        self.server_specs = servers
        self.bridge = bridge
        self.options = options
        self.clients: dict[str, McpClient] = {}
        self.tools: dict[str, list[McpTool]] = {}

    def open(self) -> "McpToolset":
        from concurrent.futures import ThreadPoolExecutor

        names = [spec["name"] for spec in self.server_specs]
        if len(set(names)) != len(names):
            raise McpError(f"server 重复: {names}")
        clients = {name: McpClient.for_server(self.gateway, name, self.bridge, **self.options) for name in names}

        def start(name: str) -> list[McpTool]:
            clients[name].connect()
            return clients[name].list_tools()

        try:
            with ThreadPoolExecutor(max_workers=max(1, len(names))) as pool:
                listed = dict(zip(names, pool.map(start, names)))  # map 按顺序返回,任何一个抛错都会在这里重新抛出
        except BaseException:
            for client in clients.values():
                client.close()
            raise
        self.clients = clients
        for spec in self.server_specs:
            selected = spec.get("tools")
            tools = listed[spec["name"]]
            self.tools[spec["name"]] = tools if selected is None else [t for t in tools if t.name in selected]
        return self

    def call(self, server: str, tool: str, arguments: dict[str, Any]) -> McpResult:
        return self.clients[server].call_tool(tool, arguments)

    def close(self) -> None:
        for client in reversed(list(self.clients.values())):
            client.close()
        self.clients = {}

    def __enter__(self) -> "McpToolset":
        return self.open()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _read_timeout(seconds: float) -> Any:
    """SDK 1.x 要 ``timedelta``,2.x 要秒数。"""
    from importlib.metadata import version

    return seconds if int(version("mcp").split(".")[0]) >= 2 else timedelta(seconds=seconds)


def _page(cursor: str) -> Any:
    from mcp import types

    return types.PaginatedRequestParams(cursor=cursor)


def _describe(exc: BaseException | None) -> str:
    if exc is None:
        return "未知错误"
    subs = getattr(exc, "exceptions", None)  # ExceptionGroup:取出真正的原因
    if subs:
        return "; ".join(_describe(e) for e in subs)
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _is_connection_error(exc: BaseException) -> bool:
    import httpx

    return isinstance(exc, (ConnectionError, httpx.TransportError, httpx.ConnectError, httpx.ReadError, BrokenPipeError, EOFError))
