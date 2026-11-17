"""外层配置(执行层接入方式)。所有子配置 extra=forbid,拼错的键直接报错。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..retry import DEFAULT_RETRY_ON, RetryPolicy
from ..models.config import ModelConfig
from .memory_mps import MpsMemoryConfig
from .memory_local import LocalMemoryConfig


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class PolicyConfig(_Strict):
    max_retries: int = 3
    backoff_s: float = 1.0
    backoff_factor: float = 2.0
    max_backoff_s: float = 30.0

    def to_policy(self) -> RetryPolicy:
        return RetryPolicy(self.max_retries, self.backoff_s, self.backoff_factor, self.max_backoff_s, DEFAULT_RETRY_ON)


class ExecSettings(_Strict):
    src: str | None = Field(default=None, description="执行层 src 目录;缺省走环境变量或仓库内默认位置")
    db_path: str = "data/imact_exec.sqlite3"  # 必须是文件(执行层每个事务单开连接,不支持 :memory:)
    context_max_bytes: int = 240_000
    decision_timeout_s: float | None = Field(default=None, description="缺省取 max(120, 模型超时+30)")


class ModelCallConfig(_Strict):
    response_format: str = "text"  # text | json_object(网关支持时)
    params: dict[str, Any] = Field(default_factory=dict, description="如 {max_completion_tokens: 4096};GPT-5 系列用 max_completion_tokens")


class MemoryRouting(_Strict):
    backend: Literal["mps", "local", "none"] = "mps"  # 兼容原有未声明 backend 的网关配置
    prefetch: bool = True  # start 时按 subtask 的指令预取一次记忆
    require_prefetch: bool = False  # 严格联调:首轮 Memory 无有效返回则停止,不调用 Action
    timestamp_mode: Literal["date", "raw"] = Field(
        default="date",
        description="MPS 模式的时间格式:date=只取日期,raw=原样。本地模式始终保留原始 current_time。",
    )
    active_query_text: str = Field(
        default="question",
        description="执行层主动 query_memory 时,传给网关的 user_query:question=模型提的问题;instruction=沿用任务指令",
    )
    gateway: MpsMemoryConfig = Field(default_factory=MpsMemoryConfig)
    local: LocalMemoryConfig = Field(default_factory=LocalMemoryConfig)
    retry: PolicyConfig = Field(default_factory=lambda: PolicyConfig(max_retries=1, backoff_s=0.5))


class RetryConfig(_Strict):
    """任务级重试由 benchmark 负责(VitaBench 每个 task 重试 3 次)。这里是模型决策级。"""

    model_call: PolicyConfig = Field(default_factory=PolicyConfig)


class OuterConfig(_Strict):
    models_file: str | None = Field(default=None, description="统一模型配置,相对本运行 YAML 所在目录解析")
    model: ModelConfig = Field(default_factory=ModelConfig)
    model_call: ModelCallConfig = Field(default_factory=ModelCallConfig)
    exec: ExecSettings = Field(default_factory=ExecSettings)
    memory: MemoryRouting = Field(default_factory=MemoryRouting)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    max_turns: int = Field(default=200, description="每个 task(= 一个 subtask)内执行层允许的决策轮数")
    max_memory_hops: int = 3


def load_outer_config(path: str | Path, *, models_file: str | Path | None = None) -> OuterConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if models_file is not None:
        raw["models_file"] = str(Path(models_file).resolve())
    cfg = OuterConfig.model_validate(raw)
    for field in ("client_path", "config_path"):
        value = Path(getattr(cfg.memory.local, field)).expanduser()
        setattr(cfg.memory.local, field, str((Path(path).resolve().parent / value).resolve()))
    if cfg.models_file:
        if "model" in raw or cfg.model_call.params:
            raise ValueError("models_file 不能与内联 model / model_call.params 混用;模型参数统一放入 models.yaml")
        cfg.models_file = str((Path(path).resolve().parent / cfg.models_file).resolve())
        from ..models.catalog import load_catalog

        cfg.model = load_catalog(cfg.models_file).selected("action")[1]
    return cfg


def apply_memory_overrides(cfg: OuterConfig, *, no_memory=False, memory_url=None, local_memory=None) -> None:
    """CLI 显式选模式;--no-memory 也关闭严格预取。路径相对当前工作目录。"""
    if no_memory:
        cfg.memory.backend = "none"
        cfg.memory.prefetch = cfg.memory.require_prefetch = False
    elif memory_url is not None:
        cfg.memory.backend = "mps"
        cfg.memory.gateway.url = memory_url
        cfg.memory.prefetch = True
    elif local_memory is not None:
        cfg.memory.backend = "local"
        cfg.memory.local.config_path = str(Path(local_memory).expanduser().resolve())
        cfg.memory.prefetch = True
