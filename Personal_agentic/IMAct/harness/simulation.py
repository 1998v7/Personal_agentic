"""一次完整模拟的结果记录。字段与历史结果文件同形。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ..evaluation.base import RewardInfo
from .conversation import TerminationReason
from .messages import Message, now_str


class SimulationRun(BaseModel):
    id: str
    task_id: str
    timestamp: str = Field(default_factory=now_str)
    start_time: str
    end_time: str
    duration: float
    termination_reason: TerminationReason
    agent_cost: Optional[float] = None
    user_cost: Optional[float] = None
    reward_info: Optional[RewardInfo] = None
    messages: list[Message]
    states: Dict[str, Any] = Field(default_factory=lambda: {"old_states": [], "new_states": []})
    trial: Optional[int] = None
    seed: Optional[int] = None
