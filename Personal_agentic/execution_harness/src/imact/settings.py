"""模型与内部服务的 TOML 配置；密钥不进入日志、快照或配置导出。"""

import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .errors import HarnessError
from .model import ChatModel


def positive_seconds(value, name: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise HarnessError("invalid_config", f"{name} 必须为有限正数") from exc
    if isinstance(value, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise HarnessError("invalid_config", f"{name} 必须为有限正数")
    return seconds


def resolve_decision_timeout(model_timeout, configured=None) -> float:
    model_timeout = positive_seconds(model_timeout, "模型请求超时")
    if configured is None:
        return max(120.0, model_timeout + 30.0)
    value = positive_seconds(configured, "决策超时")
    if value <= model_timeout:
        raise HarnessError("invalid_config", "决策超时必须大于模型请求超时，以便接收结果并处理错误")
    return value


@dataclass
class Settings:
    path: Path
    values: dict

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        with path.open("rb") as stream:
            values = tomllib.load(stream)
        if not isinstance(values.get("model"), dict):
            raise HarnessError("invalid_config", "配置缺少 [model]")
        return cls(path, values)

    def resolve(self, section, key, default=None):
        value = self.values.get(section, {}).get(key, default)
        if not value:
            raise HarnessError("invalid_config", f"缺少 {section}.{key}")
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.path.parent / p).resolve()

    def model(self, transport=None):
        c = self.values["model"]
        base = c.get("base_url") or os.getenv("IMACT_MODEL_BASE_URL", "")
        name = c.get("name") or os.getenv("IMACT_MODEL_NAME", "")
        if not base or not name:
            raise HarnessError(
                "model_not_configured",
                f"请在 {self.path} 填写 model.base_url 和 model.name，或设置对应环境变量",
                503,
            )
        if not base.startswith(("http://", "https://")):
            raise HarnessError("invalid_config", "model.base_url 必须为 HTTP(S) 地址")
        timeout = positive_seconds(c.get("timeout_seconds", 90), "model.timeout_seconds")
        tokens = c.get("max_output_tokens", 4096)
        if type(tokens) is not int or tokens < 1:
            raise HarnessError("invalid_config", "输出 token 上限必须为正整数")
        return ChatModel(
            base,
            name,
            os.getenv(c.get("api_key_env", "IMACT_MODEL_API_KEY"), ""),
            timeout=timeout,
            max_output_tokens=tokens,
            response_format=c.get("response_format", "text"),
            transport=transport,
        )
