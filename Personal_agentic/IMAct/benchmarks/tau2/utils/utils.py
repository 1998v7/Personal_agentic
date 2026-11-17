"""数据目录与哈希。"""

import hashlib
import json
import os
from pathlib import Path

ENV_VAR = "IMACT_TAU2_DATA"
REPO_ROOT = Path(__file__).resolve().parents[4]


def _find_data_dir() -> Path:
    """``<DATA_DIR>/tau2/domains/<domain>/...``。环境变量优先,否则用本仓库的 data/。"""
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).expanduser()
    return REPO_ROOT / "data"


DATA_DIR = _find_data_dir()


def get_dict_hash(obj: dict) -> str:
    """dict 的 sha256 十六进制摘要(键排序,不认识的类型转字符串)。"""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
