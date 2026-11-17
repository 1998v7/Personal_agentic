# Copyright Sierra
from pathlib import Path
from typing import Optional

from ...data_model.tasks import Task
from ...domains.airline.data_model import FlightDB
from ...domains.airline.region import AirlineRegionalSettings
from ...domains.airline.tools import AirlineTools
from ...domains.airline.utils import (
    AIRLINE_DB_PATH,
    AIRLINE_POLICY_PATH,
    AIRLINE_TASK_SET_PATH,
)
from ...environment.environment import Environment
from ...utils import load_file


def get_environment(
    db: Optional[FlightDB] = None,
    solo_mode: bool = False,
    policy: str | None = None,
    regional_settings: AirlineRegionalSettings | None = None,
) -> Environment:
    if solo_mode:
        raise ValueError("Airline domain does not support solo mode")
    if db is None:
        db = FlightDB.load(AIRLINE_DB_PATH)
    else:
        db = db.model_copy(deep=True)
    tools = AirlineTools(db, regional_settings=regional_settings)
    if policy is None:
        with open(AIRLINE_POLICY_PATH, "r") as fp:
            policy = fp.read()
    return Environment(
        domain_name="airline",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks = load_file(AIRLINE_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits are: {task_splits.keys()}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        Path(AIRLINE_TASK_SET_PATH).parent
        / f"split_{Path(AIRLINE_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
