"""按扩展名读写 json / yaml / toml / 文本。统一用 UTF-8,不依赖系统默认编码。"""

import json
import os
from pathlib import Path
from typing import Any

import toml
import yaml


def load_file(path: str | Path, **kwargs: Any) -> Any:
    path = Path(path)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp, **kwargs)
    if path.suffix in (".yaml", ".yml"):
        with open(path, "r", encoding="utf-8") as fp:
            return yaml.load(fp, Loader=yaml.SafeLoader, **kwargs)
    if path.suffix == ".toml":
        with open(path, "r", encoding="utf-8") as fp:
            return toml.load(fp, **kwargs)
    if path.suffix in (".txt", ".md"):
        if kwargs:
            raise ValueError(f"Unsupported keyword arguments: {kwargs}")
        with open(path, "r", encoding="utf-8") as fp:
            return fp.read()
    raise ValueError(f"Unsupported file extension: {path}")


def dump_file(path: str | Path, data: Any, **kwargs: Any) -> None:
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    if path.suffix == ".json":
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, **kwargs)
    elif path.suffix in (".yaml", ".yml"):
        with open(path, "w", encoding="utf-8") as fp:
            yaml.dump(data, fp, **kwargs)
    elif path.suffix == ".toml":
        # toml 写不了 Enum,先走一遍 json 转成基本类型
        with open(path, "w", encoding="utf-8") as fp:
            toml.dump(json.loads(json.dumps(data)), fp, **kwargs)
    elif path.suffix in (".txt", ".md"):
        if kwargs:
            raise ValueError(f"Unsupported keyword arguments: {kwargs}")
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(data)
    else:
        raise ValueError(f"Unsupported file extension: {path}")
