"""测试用的本地 MCP server(SSE)。``python tests/mcp_test_server.py <port>``。只给 tests/test_mcp.py 用。"""

import sys
import time

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("imact-test")


@server.tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo:{text}"


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool()
def boom() -> str:
    """Always fails with a tool-level error."""
    raise ToolError("kaboom")


@server.tool()
def sleepy(seconds: float) -> str:
    """Sleep, then answer."""
    time.sleep(seconds)
    return "awake"


if __name__ == "__main__":
    uvicorn.run(server.sse_app(), host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
