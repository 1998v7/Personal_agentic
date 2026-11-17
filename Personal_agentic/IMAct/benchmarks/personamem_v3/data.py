"""PersonaMem-v3 数据位置与读取。数据不随源码提交(``python -m IMAct.benchmarks.personamem_v3.download`` 下载)。

查找顺序:环境变量 ``IMACT_PERSONAMEM_V3_DATA``(指向含 ``backend/`` 的目录)→ 本仓库 ``data/personamem_v3/``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ENV_VAR = "IMACT_PERSONAMEM_V3_DATA"
REPO_ROOT = Path(__file__).resolve().parents[3]
APPS = ("instagram", "facebook", "threads", "chatbot", "ai_studio")  # 官方 BackendQuery 的五个 app,顺序同上游
EVAL_FILES = (*(f"{a}.json" for a in APPS), "calendar.json", "profile.json", "test.json")


def data_dir() -> Path:
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).expanduser()
    return REPO_ROOT / "data" / "personamem_v3"


DATA_DIR = data_dir()


def backend_dir() -> Path:
    return data_dir() / "backend"


def list_users() -> list[str]:
    """有 ``test.json`` 的 persona,按数字顺序。"""
    base = backend_dir()
    if not base.is_dir():
        return []
    users = [p.name for p in base.iterdir() if (p / "test.json").is_file()]
    return sorted(users, key=lambda u: (len(u), u))


def load_test(user_id: str) -> list[dict]:
    path = backend_dir() / str(user_id) / "test.json"
    return json.loads(path.read_text(encoding="utf-8"))


def missing_files(user_id: str) -> list[str]:
    folder = backend_dir() / str(user_id)
    return [f for f in EVAL_FILES if not (folder / f).is_file()]
