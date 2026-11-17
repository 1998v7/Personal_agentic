"""τ² 三个 domain(airline / retail / telecom)的任务读取与环境构造。"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Optional

from .data_model.tasks import Task
from .environment.environment import Environment

DOMAINS = ("airline", "retail", "telecom")
DEFAULT_SPLIT = "base"  # 官方评测集:airline 50 / retail 114 / telecom 114


def _module(domain: str):
    if domain not in DOMAINS:
        raise ValueError(f"不支持的 domain {domain!r};可用: {list(DOMAINS)}")
    return importlib.import_module(f"{__package__}.domains.{domain}.environment")


def get_environment(domain: str) -> Environment:
    """一个干净的、尚未应用 task 初始状态的环境。"""
    return _module(domain).get_environment()


@lru_cache(maxsize=None)
def _tasks(domain: str, split: Optional[str]) -> tuple[Task, ...]:
    return tuple(_module(domain).get_tasks(split))


def get_tasks(
    domain: str, split: Optional[str] = DEFAULT_SPLIT, task_ids: Optional[list[str]] = None, num_tasks: Optional[int] = None
) -> list[Task]:
    """按划分取 task;``task_ids`` 给了就只取这些(缺哪个报错),``num_tasks`` 再截取前 N 个。"""
    tasks = list(_tasks(domain, split))
    if task_ids is not None:
        missing = set(task_ids) - {t.id for t in tasks}
        if missing:
            raise ValueError(f"{domain} 的 {split!r} 划分里没有这些 task: {sorted(missing)}")
        tasks = [t for t in tasks if t.id in set(task_ids)]
    return tasks[:num_tasks] if num_tasks is not None else tasks
