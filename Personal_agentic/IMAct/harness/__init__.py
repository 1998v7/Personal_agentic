"""通用交互层:用户模拟器、agent(对接执行层)、对话循环。不依赖任何具体 benchmark。

benchmark(``IMAct.benchmarks.*``)提供任务数据、环境与工具、prompt;评分在 ``IMAct.evaluation``。
"""

from .agent import AgentState, HarnessAgent
from .conversation import Conversation, ConversationResult, TerminationReason
from .episode import run_episode
from .llm import LLMCaller
from .protocols import Environment, EpisodeSpec
from .simulation import SimulationRun
from .user import UserSimulator

__all__ = [
    "AgentState", "Conversation", "ConversationResult", "Environment", "EpisodeSpec", "HarnessAgent", "LLMCaller",
    "SimulationRun", "TerminationReason", "UserSimulator", "run_episode",
]
