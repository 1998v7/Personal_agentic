"""进程内共享的资源:一个 AsyncBridge(事件循环线程)与 models.yaml 里的辅助角色(用户模拟器、裁判)。

辅助角色不是被训练的对象,照常重试(失败属于基础设施问题)。同一进程里并发的多个 episode 共用它们:
``LLMCaller`` 的请求都在 bridge 的事件循环上执行,可以跨线程共享。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from IMAct.exec_layer.bridge import AsyncBridge
from IMAct.harness.llm import LLMCaller
from IMAct.models.catalog import load_catalog
from IMAct.retry import RetryPolicy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_FILE = ROOT / "configs" / "models.yaml"
HELPER_RETRY = RetryPolicy(max_retries=3, backoff_s=2.0)


def imact_root() -> Path:
    """IMAct(Personal_agentic)所在目录:缺省是本项目的同级目录,可用 ``PAT_IMACT_ROOT`` 指到别处。

    服务器上 clone 下来的目录名不一定还叫 ``Personal_agentic``(例如仓库名是 my_project),所以不写死。
    """
    return Path(os.environ.get("PAT_IMACT_ROOT") or ROOT.parent / "Personal_agentic")

_lock = threading.RLock()
_bridge: AsyncBridge | None = None
_roles: dict[tuple[str, str], LLMCaller] = {}


def models_file(path: str | None = None) -> Path:
    return Path(path or os.environ.get("PAT_MODELS_FILE") or DEFAULT_MODELS_FILE)


def bridge() -> AsyncBridge:
    global _bridge
    with _lock:
        if _bridge is None:
            _bridge = AsyncBridge("pat-train")
        return _bridge


def role(name: str, path: str | None = None) -> LLMCaller:
    """models.yaml 里 ``roles.<name>`` 选中的模型(``user`` / ``judge``)。"""
    file = str(models_file(path))
    key = (file, name)
    with _lock:
        if key not in _roles:
            _, cfg = load_catalog(file).selected(name)
            _roles[key] = LLMCaller(cfg, bridge(), policy=HELPER_RETRY)
        return _roles[key]

