"""在 uni-agent 的 agent 注册表里登记 ``imact``:被训练的 agent 就是 IMAct 的 harness 本身。

uni-agent 的 Task 配置要求 ``agent`` 是一个注册过的 AgentConfig;框架只往 ``agent.model`` 里填当前策略的 Gateway
地址(``base_url``)与模型名。IMAct 的对话循环、工具调用、用户模拟都在各 Task 的 ``run`` 里完成,所以这里的
``ImactAgent.run`` 不会被调用。
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from uni_agent.agents import Agent, AgentConfig, AgentResult
from uni_agent.agents.registry import AGENT_REGISTRY, register_agent

NAME = "imact"


class ImactAgentConfig(AgentConfig):
    name: str = NAME
    max_tokens_per_turn: int | None = Field(default=None, description="每次生成的 max_tokens;缺省由 Gateway 按剩余长度决定")
    timeout_s: float = Field(default=900.0, description="单次请求的超时(本地 rollout 排队时可能很慢)")


class ImactAgent(Agent):
    config_model = ImactAgentConfig

    async def run(self, *, sandbox: Any, messages: list[dict[str, Any]], workdir: str | None = None) -> AgentResult:
        raise NotImplementedError("imact agent 由 pat_train 的 Task 直接驱动,不经过 Agent.run")


def sample_fields(config: Any, *keys: str) -> tuple[str, ...]:
    """一条样本自己的字段(domain / task_id / split …)都放在 ``metadata`` 里,运行参数放在 Task Config YAML。

    训练数据是 parquet:不同 bench 的嵌套字段会被合并成同一个 struct,缺的键补 null。所以样本只用一组
    各 bench 通用的字符串键,不往 TaskConfig 的顶层字段里写(那里 ``extra="forbid"``,null 也过不了校验)。
    """
    missing = [k for k in keys if not config.metadata.get(k)]
    if missing:
        raise ValueError(f"{config.name}: 样本 metadata 缺少 {missing}")
    return tuple(str(config.metadata[k]) for k in keys)


if NAME not in AGENT_REGISTRY:
    register_agent(NAME)(ImactAgent)
