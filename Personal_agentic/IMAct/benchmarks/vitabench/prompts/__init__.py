"""VitaBench 的 agent / 用户模拟器 prompt。每个 YAML 按语言分段,文件名即属性名。"""

import os
from functools import lru_cache

import yaml

from ..config import DEFAULT_LANGUAGE

_DIR = os.path.dirname(__file__)


class Prompts:
    def __init__(self, language: str = DEFAULT_LANGUAGE):
        self.language = language
        for name in sorted(os.listdir(_DIR)):
            if not name.endswith(".yaml"):
                continue
            with open(os.path.join(_DIR, name), encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if language not in data:
                raise ValueError(f"Language {language} not found in prompt {name}")
            setattr(self, name.split(".")[0], data[language])


@lru_cache(maxsize=None)
def get_prompts(language: str | None = None) -> Prompts:
    return Prompts(language or DEFAULT_LANGUAGE)
