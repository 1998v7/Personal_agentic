"""τ² 的语言包与评测单元(cell):一个 cell = 一种语言 × 一个 domain,带着它自己的任务、数据库、policy 和人设。

``text-v1`` 由 18 个 cell(6 语种 × 3 domain)组成,每个 cell 用同一批冻结的 50 个 task(按 canonical ID 对应)。
英文 cell 直接用原始数据库,并用上游审阅过的 retail / telecom 任务替换同 ID 的原任务;其他 5 种语言用
各自发布的本地化数据库包(``regional/<domain>.json``:本地化的 DB、policy、任务,以及货币 / 时区 / 机场目录)。

每个 task 在 cell 里有两个 ID:本语言的 ID(如 ``6_zh_identity``)和跨语言一致的 canonical ID(如 ``6``)。
命令行和统计都以 canonical ID 为准,这样同一个 task 在不同语言下的表现可以直接对照。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from .data_model.tasks import Task
from .environment.regional import REGIONAL_RUNTIME_VERSION, DomainRegionalSettings
from .tasks import DOMAINS, get_environment as default_environment, get_tasks as registry_tasks
from .utils import DATA_DIR, load_file

LANGUAGES = ("en", "es", "hi", "ko", "pt", "zh")
SCENARIO_LANGUAGE = "en"  # 原始数据库与任务的语言;其他语言需要"跨语言"的额外说明
PERSONA_SEED = 300  # 每种语言每个性别只有一个人设,所以人设与 seed 无关;保留只为复现上游的分配规则
MULTILINGUAL = DATA_DIR / "tau2" / "multilingual"
BENCHMARKS = DATA_DIR / "tau2" / "benchmarks"
SUBSETS = DATA_DIR / "tau2" / "task_subsets"


# ---------------------------------------------------------------------- 语言包
@dataclass(frozen=True)
class Persona:
    id: str
    gender: str
    locale: str
    variety: str
    display_language: str
    gender_guidance: str
    pragmatics_clauses: tuple[str, ...]
    agent_background_clause: str
    origin: Optional[str]  # 跨语言时:说话人来自哪里(进语言指令);原语言为 None


@lru_cache(maxsize=None)
def load_pack(language: str) -> dict[str, Any]:
    if language not in LANGUAGES:
        raise ValueError(f"不支持的语言 {language!r};可用: {list(LANGUAGES)}")
    return yaml.safe_load((MULTILINGUAL / language / "pack.yaml").read_text(encoding="utf-8"))


def cross_lingual(language: str) -> bool:
    return language.split("-")[0] != SCENARIO_LANGUAGE


def _persona(pack: dict[str, Any], persona_id: str) -> Persona:
    p = pack["personas"][persona_id]
    if (p.get("verbosity") or "standard") != "standard":
        raise NotImplementedError(f"人设 {persona_id} 的 verbosity={p['verbosity']!r} 还没有移植")
    directive = p.get("language_directive")
    return Persona(
        id=persona_id, gender=p["gender"], locale=p["locale"], variety=p["variety"], display_language=pack["display_name"],
        gender_guidance=p["gender_guidance"], pragmatics_clauses=tuple(p.get("pragmatics_clauses") or ()),
        agent_background_clause=p.get("agent_background_clause") or "", origin=directive["origin"] if directive else None,
    )


def assign_persona(pack: dict[str, Any], entry: dict[str, Any], seed: int = PERSONA_SEED) -> str:
    """确定性地、按性别匹配地给一个 task 选人设:元数据里指定了就用它,否则按 sha256(seed:canonical_task_id) 取模。"""
    if entry.get("persona_id") is not None:
        return entry["persona_id"]
    gender = entry.get("caller_gender")
    eligible = sorted(pid for pid, p in pack["personas"].items() if gender is None or p["gender"] == gender)
    index = int.from_bytes(hashlib.sha256(f"{seed}:{entry['canonical_task_id']}".encode()).digest()[:8], "big") % len(eligible)
    return eligible[index]


# ---------------------------------------------------------------------- cell
@dataclass
class Cell:
    language: str
    domain: str
    tasks: list[Task]  # 按 canonical_ids 的顺序
    canonical_ids: list[str]
    local_id: dict[str, str]  # canonical -> 本语言 task id
    metadata: dict[str, dict[str, Any]]  # 本语言 task id -> {canonical_task_id, caller_gender, persona_id}
    pack: dict[str, Any]
    regional: Optional[dict[str, Any]] = None  # 本地化数据库包;原语言为 None
    label: str = "text-v1"
    _env_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return f"{self.language}-{self.domain}"

    @property
    def cross_lingual(self) -> bool:
        return cross_lingual(self.language)

    @property
    def binding(self) -> dict[str, Any]:
        return self.pack["domains"].get(self.domain, {})

    def environment(self):
        """一个全新的环境(各自独立的数据库副本)。原语言用原始数据;其他语言用本语言的数据库、policy 与区域设置。"""
        if self.regional is None:
            return default_environment(self.domain)
        from .tasks import _module

        return _module(self.domain).get_environment(**self._env_kwargs)

    def task_by_canonical(self, canonical_id: str) -> Task:
        return self.tasks[self.canonical_ids.index(canonical_id)]

    def canonical_of(self, task: Task) -> str:
        return (self.metadata.get(task.id) or {"canonical_task_id": task.id})["canonical_task_id"]

    def persona_id(self, task: Task) -> str:
        return assign_persona(self.pack, self.metadata.get(task.id) or {"canonical_task_id": task.id})

    def persona(self, task: Task) -> Persona:
        return _persona(self.pack, self.persona_id(task))

    def greeting(self, task: Task) -> str:
        return task.agent_opener or self.pack["agent_greeting"]

    def select(self, task_ids: Optional[list[str]] = None, num_tasks: Optional[int] = None) -> list[Task]:
        """按 canonical ID 选 task(缺哪个报错),再取前 N 个,与上游的选择语义一致。"""
        ids = list(self.canonical_ids)
        if task_ids is not None:
            missing = set(task_ids) - set(ids)
            if missing:
                raise ValueError(f"{self.name} 的冻结子集里没有这些 task: {sorted(missing)}")
            ids = [i for i in ids if i in set(task_ids)]
        if num_tasks is not None:
            if not 1 <= num_tasks <= len(ids):
                raise ValueError(f"--num-tasks 必须在 1 到 {len(ids)} 之间({self.name})")
            ids = ids[:num_tasks]
        return [self.task_by_canonical(i) for i in ids]


# ---------------------------------------------------------------------- 加载
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _manifest() -> dict[str, Any]:
    return _read_json(BENCHMARKS / "text.json")


def text_v1_cells() -> list[tuple[str, str]]:
    """官方 text-v1 的 18 个 (语言, domain),按 manifest 顺序。"""
    return [(c["language"], c["domain"]) for c in _manifest()["cells"]]


@lru_cache(maxsize=None)
def frozen_canonical_ids(domain: str) -> tuple[str, ...]:
    """冻结子集的 canonical task ID(顺序即评测顺序)。文件内容必须与 manifest 里钉死的摘要一致。"""
    ref = _manifest()["task_subsets"][domain]
    path = SUBSETS / f"{ref['name']}.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
        raise ValueError(f"冻结子集 {ref['name']} 与 manifest 钉死的摘要不一致,数据可能被改动")
    return tuple(_read_json(path)["task_ids"])


def _metadata(language: str, domain: str) -> dict[str, dict[str, Any]]:
    name = load_pack(language)["domains"][domain].get("task_metadata_file")
    return _read_json(MULTILINGUAL / language / name) if name else {}


def _canonical_tasks(domain: str, ids: list[str], replacements: dict[str, Task]) -> list[Task]:
    """原始 tasks.json 里取指定 ID 的 task;审阅过的替换任务覆盖同 ID 的原任务。只校验用到的那几个。"""
    from .domains.airline.utils import AIRLINE_TASK_SET_PATH
    from .domains.retail.utils import RETAIL_TASK_SET_PATH
    from .domains.telecom.utils import TELECOM_TASK_SET_PATH

    path = {"airline": AIRLINE_TASK_SET_PATH, "retail": RETAIL_TASK_SET_PATH, "telecom": TELECOM_TASK_SET_PATH}[domain]
    raw = {t["id"]: t for t in load_file(path)}
    missing = [i for i in ids if i not in raw and i not in replacements]
    if missing:
        raise ValueError(f"{domain} 的原始任务里没有: {missing[:5]}")
    return [replacements.get(i) or Task.model_validate(raw[i]) for i in ids]


@lru_cache(maxsize=None)
def _regional_bundle(language: str, domain: str) -> dict[str, Any]:
    rel = load_pack(language)["domains"][domain]["regional_bundle_file"]
    bundle = _read_json(MULTILINGUAL / language / rel)
    if bundle["domain"] != domain or bundle["settings"]["language"] != language:
        raise ValueError("本地化数据库包与语言 / domain 不匹配")
    if bundle["runtime_rules_version"] != REGIONAL_RUNTIME_VERSION:
        raise ValueError(f"本地化数据库包要求另一版区域工具规则: {bundle['runtime_rules_version']}")
    return bundle


def _regional_env_kwargs(bundle: dict[str, Any]) -> dict[str, Any]:
    """本地化数据库包 → 环境构造参数。数据库先校验成模型一次,每个环境再深拷贝一份。"""
    from .domains.airline.data_model import FlightDB
    from .domains.airline.region import AirlineRegionalSettings
    from .domains.retail.data_model import RetailDB
    from .domains.telecom.data_model import TelecomDB
    from .domains.telecom.region import TelecomRegionalSettings
    from .domains.telecom.user_data_model import TelecomUserDB

    domain = bundle["domain"]
    kwargs: dict[str, Any] = {"db": {"airline": FlightDB, "retail": RetailDB, "telecom": TelecomDB}[domain].model_validate(bundle["database"])}
    if domain == "telecom":
        kwargs["user_db"] = TelecomUserDB.model_validate(bundle["user_database"])
        kwargs["policy"] = ("<main_policy>\n" + bundle["policies"]["main_policy"] + "\n</main_policy>\n<tech_support_policy>\n"
                            + bundle["policies"]["tech_support_manual"] + "\n</tech_support_policy>")
    else:
        kwargs["policy"] = bundle["policies"]["policy"]
    s = {k: bundle["settings"][k] for k in ("country_code", "currency_code", "currency_scale", "time_zone")}
    if domain == "airline":
        kwargs["regional_settings"] = AirlineRegionalSettings(
            **s, current_datetime=datetime(2024, 5, 15, 15), airports=[{"iata": c, "city": city} for c, city in bundle["airports"].items()])
    elif domain == "telecom":
        kwargs["regional_settings"] = TelecomRegionalSettings(**s, current_datetime=datetime(2025, 2, 25, 12, 8))
    else:
        kwargs["regional_settings"] = DomainRegionalSettings(**s)
    return kwargs


@lru_cache(maxsize=None)
def _cached_regional_env_kwargs(language: str, domain: str) -> dict[str, Any]:
    return _regional_env_kwargs(_regional_bundle(language, domain))


def load_cell(language: str, domain: str) -> Cell:
    """官方 text-v1 里 (语言, domain) 这个 cell。"""
    if domain not in DOMAINS:
        raise ValueError(f"不支持的 domain {domain!r};可用: {list(DOMAINS)}")
    if (language, domain) not in set(text_v1_cells()):
        raise ValueError(f"text-v1 里没有 ({language}, {domain})")
    pack = load_pack(language)
    binding = pack["domains"][domain]
    ids = list(frozen_canonical_ids(domain))
    metadata = _metadata(language, domain)

    if "regional_bundle_file" in binding:
        bundle = _regional_bundle(language, domain)
        by_id = {t["id"]: t for t in bundle["tasks"]}
        metadata = bundle["metadata"]
        if set(by_id) != set(metadata):
            raise ValueError("本地化数据库包的任务与元数据不一致")
        local_id = {m["canonical_task_id"]: tid for tid, m in metadata.items()}
        if not set(ids) <= set(local_id):
            raise ValueError("本地化数据库包缺少冻结子集里的 task")
        tasks = [Task.model_validate(by_id[local_id[c]]) for c in ids]
        return Cell(language, domain, tasks, ids, local_id, metadata, pack, regional=bundle,
                    _env_kwargs=_cached_regional_env_kwargs(language, domain))

    replacements: dict[str, Task] = {}
    if binding.get("tasks_file"):
        replacements = {t["id"]: Task.model_validate(t) for t in _read_json(MULTILINGUAL / language / binding["tasks_file"])}
    tasks = _canonical_tasks(domain, ids, replacements)
    metadata = {tid: m for tid, m in metadata.items() if tid in set(ids)} or {i: {"canonical_task_id": i} for i in ids}
    return Cell(language, domain, tasks, ids, {i: i for i in ids}, metadata, pack)


def load_base_cell(domain: str) -> Cell:
    """pre-2.0 的原始英文口径:registry 的 base 划分,原始任务(没有审阅替换)。airline 50 / retail 114 / telecom 114。

    与 text-v1 不同,它的分数不能和 text-v1 榜单对照。只用于回归与和旧结果对比。
    """
    pack = load_pack("en")
    tasks = registry_tasks(domain, "base")
    ids = [t.id for t in tasks]
    published = _metadata("en", domain)
    metadata = {i: published.get(i) or {"canonical_task_id": i} for i in ids}
    return Cell("en", domain, tasks, ids, {i: i for i in ids}, metadata, pack, label="base")
