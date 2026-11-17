"""MCP-Universe 各领域的评分函数(比较函数)。导入本包即向 ``IMAct.evaluation.mcp_universe`` 的注册表登记。

这些函数从上游 MCP-Universe 移植,除导入与 google_search 的裁判调用外保持原样,所以保留上游的编码风格。
它们要查询真实服务:经 MCP 网关(github、google_maps、notion),或直接联网(yfinance、playwright、weather)。
设了 ``EVAL_REMOTE_URL`` 时,yfinance / playwright 的比较函数可以交给远端执行,见 ``IMAct.evaluation.mcp_universe``。
"""

from . import github, google_maps, google_search, notion, playwright, weather, yfinance  # noqa: F401
