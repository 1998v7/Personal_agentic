"""把 τ² 的 ``Environment`` 接到 harness 的环境协议上。"""

from __future__ import annotations

from typing import Any, Optional

from ...harness.messages import ToolCall, ToolMessage
from .data_model.tasks import Task
from .environment.environment import Environment


class Tau2HarnessEnvironment:
    """一个 task 一个实例:构造时就应用 task 的初始状态(初始数据、初始动作、预置历史)。"""

    def __init__(self, env: Environment, task: Task):
        self.env = env
        self.task = task
        state = task.initial_state
        env.set_state(
            initialization_data=state.initialization_data if state else None,
            initialization_actions=state.initialization_actions if state else None,
            message_history=list(state.message_history) if state and state.message_history else [],
        )

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema for tool in self.env.get_tools()]

    def user_tool_schemas(self) -> Optional[list[dict[str, Any]]]:
        """这个 task 里用户自己能调用的工具;没有(或环境根本没有用户工具)就是 None。"""
        try:
            tools = self.env.get_user_tools(include=self.task.user_tools)
        except Exception:  # noqa: BLE001 - 没有用户工具的 domain 会抛 ValueError
            return None
        return [tool.openai_schema for tool in tools] or None

    def get_response(self, call: ToolCall) -> ToolMessage:
        return self.env.get_response(call)

    def sync_tools(self) -> None:
        self.env.sync_tools()

    def get_states(self) -> dict[str, Any]:
        return {}  # 评分时会在新环境里重放整段对话,不需要这里的快照
