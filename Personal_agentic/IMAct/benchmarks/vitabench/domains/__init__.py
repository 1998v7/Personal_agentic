"""VitaBench 的三个业务域:外卖(delivery)、到店(instore)、酒旅(ota)。"""

from typing import Callable

from ..environment.environment import Environment, get_cross_environment


def _loaders() -> dict[str, Callable[..., Environment]]:
    from .delivery.environment import get_environment as delivery
    from .instore.environment import get_environment as instore
    from .ota.environment import get_environment as ota

    return {"delivery": delivery, "instore": instore, "ota": ota}


def get_domain_environment(domain: str, env_data: dict | None = None, language: str | None = None) -> Environment:
    """单个业务域的环境。``env_data`` 是任务里的 environment 字段。"""
    loaders = _loaders()
    if domain not in loaders:
        raise ValueError(f"未知业务域 {domain!r};可用: {sorted(loaders)}")
    return loaders[domain](env_data, language)


def build_environment(domain: str, env_data: dict | None = None, language: str | None = None) -> Environment:
    """域名含逗号表示跨域任务,合并各域的工具与数据库。"""
    if "," in domain:
        return get_cross_environment(domain, env_data or {}, language)
    return get_domain_environment(domain, env_data, language)
