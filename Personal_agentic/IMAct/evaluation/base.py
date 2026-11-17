"""评估的通用类型。各 benchmark 的评估器放在本目录下的独立模块里,实现 ``Evaluator``。"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Optional, Protocol

from pydantic import BaseModel, Field


class RewardType(str, Enum):
    """决定最终得分的评分分量。各 benchmark 用其中的一部分。"""

    NL_ASSERTION = "NL_ASSERTION"
    DB = "DB"
    ENV_ASSERTION = "ENV_ASSERTION"
    ACTION = "ACTION"
    COMMUNICATE = "COMMUNICATE"


class NLRubricCheck(BaseModel):
    """一条自然语言得分点的判定。"""

    nl_rubric: Optional[str] = None
    met: bool
    justification: str


class DBCheck(BaseModel):
    """环境数据库的终态是否和参考终态一致。"""

    db_match: bool
    db_reward: float


class EnvAssertionCheck(BaseModel):
    env_assertion: dict  # 断言本身(函数名、参数、期望值),原样保存
    met: bool
    reward: float


class ActionCheck(BaseModel):
    action: dict  # 期望的动作,原样保存
    action_match: bool
    action_reward: float
    tool_type: Optional[str] = None  # read / write / think / generic


class NLAssertionCheck(BaseModel):
    nl_assertion: str
    met: bool
    justification: str


class CommunicateCheck(BaseModel):
    info: str
    met: bool
    justification: str


class RewardInfo(BaseModel):
    reward: Annotated[float, Field(description="最终得分")]
    nl_rubrics: Annotated[Optional[list[NLRubricCheck]], Field(default=None)]
    reward_breakdown: Annotated[Optional[dict[RewardType, float]], Field(default=None)]
    info: Annotated[Optional[dict], Field(default=None)]
    window_evaluations: Annotated[Optional[list[dict]], Field(default=None, description="逐窗口的裁判输入输出")]
    # 以下是环境状态类评分(τ² 一类 benchmark)用的明细
    db_check: Annotated[Optional[DBCheck], Field(default=None)]
    env_assertions: Annotated[Optional[list[EnvAssertionCheck]], Field(default=None)]
    action_checks: Annotated[Optional[list[ActionCheck]], Field(default=None)]
    nl_assertions: Annotated[Optional[list[NLAssertionCheck]], Field(default=None)]
    communicate_checks: Annotated[Optional[list[CommunicateCheck]], Field(default=None)]
    reward_basis: Annotated[Optional[list[RewardType]], Field(default=None, description="哪些分量参与相乘")]


class JudgeError(RuntimeError):
    """裁判响应不可用(格式错误、缺项、重复项)。调用方决定记为失败还是 0 分,评估器不把它悄悄当成正常零分。

    ``windows`` 是出错前已拿到的逐窗口裁判输入输出(含出错的那个),供事后排查。
    """

    def __init__(self, message: str, windows: list[dict] | None = None):
        super().__init__(message)
        self.windows = windows or []


class Evaluator(Protocol):
    """给一段对话打分。``task`` 是该 benchmark 的评估输入(如评分点列表),由对应评估模块定义。"""

    def evaluate(self, result: Any, task: Any) -> RewardInfo: ...
