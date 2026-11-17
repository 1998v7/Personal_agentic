"""定位并导入执行层(`imact` 包)。

执行层是别的团队交付的独立项目:我们不修改它,也不把它安装进它自己的目录
(editable 安装会在它的源码树里留下文件),只把它的 ``src`` 加进 ``sys.path``。
查找顺序:显式参数 > 环境变量 ``IMACT_EXEC_HARNESS_SRC`` > 仓库内的 ``execution_harness/src``。
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ENV_VAR = "IMACT_EXEC_HARNESS_SRC"
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = _REPO_ROOT / "execution_harness" / "src"


class ExecLayerNotFound(ImportError):
    pass


def ensure_importable(src: str | os.PathLike | None = None):
    """保证 ``import imact`` 可用,返回 ``imact`` 模块。"""
    candidates = [src, os.environ.get(ENV_VAR), DEFAULT_SRC]
    for cand in candidates:
        if not cand:
            continue
        path = Path(cand).expanduser().resolve()
        if (path / "imact" / "__init__.py").exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            mod = importlib.import_module("imact")
            _check_not_ours(mod, path)
            return mod
    try:  # 已经由外部环境安装好了
        return importlib.import_module("imact")
    except ImportError as exc:
        raise ExecLayerNotFound(
            f"找不到执行层 `imact` 包。请设置环境变量 {ENV_VAR} 指向其 src 目录,或把它放在 {DEFAULT_SRC}。"
            "进程内调用只需要 jsonschema、httpx2、pydantic>=2.14;fastapi / uvicorn 只在起 HTTP 服务时才需要。"
        ) from exc


def _check_not_ours(mod, expected_src: Path) -> None:
    """我们的包叫 ``IMAct``,执行层叫 ``imact``,只差大小写。设了 PYTHONCASEOK=1(或文件系统/解释器忽略大小写)时,
    ``import imact`` 可能解析到我们自己的目录,把执行层顶掉。这里确认导入到的确实是执行层。"""
    file = Path(getattr(mod, "__file__", "") or "").resolve()
    if expected_src.resolve() not in file.parents:
        raise ExecLayerNotFound(
            f"`import imact` 解析到了 {file},而不是执行层 {expected_src}。"
            "我们的包名 IMAct 与执行层的 imact 只差大小写;请取消环境变量 PYTHONCASEOK。"
        )
