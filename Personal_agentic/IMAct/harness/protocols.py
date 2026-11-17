"""harness 与 benchmark 之间的接口。benchmark 只需提供这些东西,不需要知道对话循环怎么跑。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .messages import ToolCall, ToolMessage


class Environment(Protocol):
    """一个对话片段(episode)里助手可以操作的世界:工具 + 状态。"""

    def tool_schemas(self) -> list[dict[str, Any]]:
        """OpenAI function 格式的工具定义。"""

    def get_response(self, call: ToolCall) -> ToolMessage:
        """执行一次工具调用并返回回执;业务报错也以 ``error=True`` 的回执返回,不抛异常。"""

    def get_states(self) -> dict[str, Any]:
        """对话结束时交给评估器的环境状态快照。"""

    # 可选:用户也能调工具的环境额外提供
    #   user_tool_schemas() -> list[dict]   用户可用工具的 OpenAI function 定义
    #   sync_tools() -> None                每个工具往返之后让环境同步两边的状态


@dataclass
class EpisodeSpec:
    """一个对话片段(VitaBench 里是一个 subtask)交给 agent 的全部输入。"""

    bench: str
    user_id: str
    episode_id: str  # Memory 的 query_id
    domain: str
    instruction: str  # 预取记忆用的任务描述
    timestamp: Any  # Memory 需要的任务时间,由 benchmark 决定口径
    system_prompt: str
    tools: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
