"""IMAct: 上游 evaluation/claude_subagent.py 的桩。

上游用它启动 Claude Code 子进程(agent_tools / mcp_agent 模式)。本接入只跑 llm_memory 模式,这些分支走不到;
保留同名入口只是为了让 runner 模块的顶层导入不变。真的被调用时直接报错,不会悄悄返回空答案。
"""

_OVERPERS_RAIL = ""


def run_subagent(*args, **kwargs):
    raise NotImplementedError("Claude Code agent 模式(agent_tools / mcp_agent)不在 IMAct 的 PersonaMem-v3 接入范围内,只支持 llm_memory")
