"""VitaBench personalization 任务的读取。

任务文件按以下顺序查找:环境变量 ``IMACT_VITABENCH_TASKS``(仅中文)→ 仓库内 ``data/vitabench/tasks[_en].json``。
数据集本身不随源码提交。
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .data_model.personalization_task import PersonalizationTask

logger = logging.getLogger(__name__)

ENV_VAR = "IMACT_VITABENCH_TASKS"
REPO_ROOT = Path(__file__).resolve().parents[3]


def tasks_path(language: str | None = None) -> Path:
    name = "tasks_en.json" if language == "english" else "tasks.json"
    override = os.environ.get(ENV_VAR)
    if override and language != "english":
        return Path(override).expanduser()
    candidate = REPO_ROOT / "data" / "vitabench" / name
    if candidate.is_file():
        return candidate
    # 英文数据缺失时回退到中文文件,与上游行为一致
    fallback = REPO_ROOT / "data" / "vitabench" / "tasks.json"
    return fallback if language == "english" and fallback.is_file() else REPO_ROOT / "data" / "vitabench" / name


def load_tasks(language: str | None = None, path: str | os.PathLike | None = None) -> list[PersonalizationTask]:
    path = Path(path) if path else tasks_path(language)
    if not path.exists():
        logger.warning("Personalization task file not found: %s", path)
        return []
    tasks = [PersonalizationTask.model_validate(t) for t in json.loads(path.read_text(encoding="utf-8"))]
    logger.info("Loaded %d personalization tasks from %s", len(tasks), path)
    return tasks


def get_tasks(task_ids: Optional[list[str]] = None, num_tasks: Optional[int] = None, language: str | None = None) -> list[PersonalizationTask]:
    if task_ids is None and num_tasks is None:
        return load_tasks(language)
    tasks: list[PersonalizationTask] = []
    if task_ids is not None:
        tasks = [t for t in load_tasks(language) if t.id in task_ids]
        if len(tasks) != len(task_ids):
            raise ValueError(f"Not all tasks were found: {set(task_ids) - {t.id for t in tasks}}")
    if num_tasks is not None:
        tasks = load_tasks(language)[:num_tasks]
    return tasks


@lru_cache(maxsize=4)
def _current_times(path: str) -> dict[str, str]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {s["subtask_id"]: s["current_time"] for u in raw for s in u.get("subtasks", []) if s.get("current_time") and s.get("subtask_id")}


def current_time_map(path: str | os.PathLike | None = None) -> dict[str, str]:
    """subtask_id -> 原任务的 current_time。

    Memory 接口要求 ``task.timestamp`` 等于原任务的 ``current_time``,但 ``SubTask`` 模型不声明该字段,加载时会丢掉;
    ``environment.time`` 也不能替代(数据里有 7 个 subtask 两者日期不同,其中 3 个 environment.time 为空),所以直接读任务文件。
    """
    return _current_times(str(path or tasks_path()))
