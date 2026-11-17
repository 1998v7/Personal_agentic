"""MCP-Universe 的评分引擎:一条评估是 ``func(...) op value``。

``func`` 是用 ``->`` 串起来的链(``json -> get(routes) -> foreach -> get(x) -> len``),把 agent 的最终回答变成一个或一组值;
``op`` 是比较函数,与 ``value`` / ``op_args`` 比较并给出 ``(passed, reason)``。
各领域的比较函数(``github.check_repository``、``google_maps.city_name_match`` …)不在这里:它们要查询真实服务,
属于 benchmark 专属代码,放在 ``IMAct.benchmarks.mcp_universe.checks``,导入时向本模块的注册表登记。

行为与上游 MCP-Universe 的 evaluator 对齐:任何一个比较不通过就整条不通过;链里出错记为 ``Execution error``
(JSON 解析失败记为 ``JSON decoding error``),不抛出。``value`` / ``op_args`` 里的 ``{{ENV_VAR}}`` 在构造时渲染。

需要远程评估时(``EVAL_REMOTE_URL``,以及前缀白名单 ``EVAL_REMOTE_PREFIXES``),命中前缀的比较函数不在本地执行,
POST 到远端的 ``/eval/compare`` 由那边执行并返回 ``(passed, reason)``。
"""

from __future__ import annotations

import json
import os
import re
import traceback
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

import httpx
from jinja2 import Environment, meta
from pydantic import BaseModel

EVALUATION_FUNCTIONS: dict[str, Callable[..., Awaitable[Any]]] = {}
COMPARISON_FUNCTIONS: dict[str, Callable[..., Awaitable[tuple[bool, str]]]] = {}


class Context:
    """评估时能用到的外部信息:环境变量(优先于 ``os.environ``)、MCP 网关、裁判模型。

    比较函数通过关键字参数 ``context`` 拿到它。``gateway`` 是 ``IMAct.harness.mcp.GatewayConfig``,
    ``judge`` 是 ``IMAct.harness.LLMCaller``;都只在用到的函数里才需要,所以允许为空。
    """

    def __init__(self, env: Mapping[str, str] | None = None, *, gateway: Any = None, judge: Any = None):
        self.env: dict[str, str] = dict(env or {})
        self.gateway = gateway
        self.judge = judge
        # 评估过程中遇到的基础设施故障(连不上网关、裁判不可用)。比较函数照上游的习惯可能把它们吞成"没找到",
        # 这里留一份记录,让调用方能把"服务不可用"和"agent 答错"分开。
        self.infra_errors: list[str] = []

    def get_env(self, var_name: str, default_value: str = "") -> str:
        return self.env.get(var_name, os.environ.get(var_name, default_value))


class FunctionResult(BaseModel):
    result: Any


def eval_func(name: str):
    def _decorator(func: Callable):
        assert name not in EVALUATION_FUNCTIONS, "Duplicated evaluation function name"
        EVALUATION_FUNCTIONS[name] = func
        return func

    return _decorator


def compare_func(name: str):
    def _decorator(func: Callable):
        assert name not in COMPARISON_FUNCTIONS, "Duplicated comparison function name"
        COMPARISON_FUNCTIONS[name] = func
        return func

    return _decorator


# ---- 取值函数 ----------------------------------------------------------------------------------


@eval_func(name="json")
async def json_decode(x: Any, *args, **kwargs) -> Any:
    if isinstance(x, FunctionResult):
        assert isinstance(x.result, str), "The input is not a string"
        x = x.result.strip().strip("`").strip()
        if x.startswith("json"):
            x = x[4:].strip()
        return FunctionResult(result=json.loads(x))
    if isinstance(x, (list, tuple)):
        return [await json_decode(y, *args, **kwargs) for y in x]
    raise NotImplementedError(f"`json_decode` doesn't support type {type(x)}")


@eval_func(name="get")
async def get(x: Any, key: str, *args, **kwargs) -> Any:
    if isinstance(x, FunctionResult):
        assert isinstance(x.result, dict), "The input is not a dict"
        return FunctionResult(result=x.result.get(key, None))
    if isinstance(x, (list, tuple)):
        return [await get(y, key, *args, **kwargs) for y in x]
    raise NotImplementedError(f"`get` doesn't support type {type(x)}")


@eval_func(name="len")
async def length(x: Any, *args, **kwargs) -> Any:
    if isinstance(x, FunctionResult):
        assert isinstance(x.result, (list, tuple)), "The input is not a list"
        return FunctionResult(result=len(x.result))
    if isinstance(x, (list, tuple)):
        return [await length(y, *args, **kwargs) for y in x]
    raise NotImplementedError(f"`len` doesn't support type {type(x)}")


@eval_func(name="foreach")
async def foreach(x: List, *args, **kwargs) -> Any:
    if isinstance(x, FunctionResult):
        assert isinstance(x.result, (list, tuple)), "The input is not a list"
        return [FunctionResult(result=y) for y in x.result]
    if isinstance(x, (list, tuple)):
        return [await foreach(y, *args, **kwargs) for y in x]
    raise NotImplementedError(f"`foreach` doesn't support type {type(x)}")


@eval_func(name="raw")
async def raw_decode(x: Any, *args, **kwargs) -> Any:
    if isinstance(x, FunctionResult):
        return FunctionResult(result=x)
    if isinstance(x, (list, tuple)):
        return [await raw_decode(y, *args, **kwargs) for y in x]
    raise NotImplementedError(f"`raw_decode` doesn't support type {type(x)}")


@eval_func(name="list")
async def to_list(x: Any, *args, **kwargs) -> Any:
    if isinstance(x, FunctionResult):
        return FunctionResult(result=[x.result])
    if isinstance(x, (list, tuple)):
        return [await to_list(y, *args, **kwargs) for y in x]


# ---- 比较函数 ----------------------------------------------------------------------------------


def _plain(x: Any) -> Any:
    return x.result if isinstance(x, FunctionResult) else x


@compare_func(name="=")
async def equal(a: Any, b: Any, *args, **kwargs) -> tuple[bool, str]:
    return (True, "") if _plain(a) == _plain(b) else (False, "output is not equal to ground-truth")


@compare_func(name="<")
async def less_than(a: Any, b: Any, *args, **kwargs) -> tuple[bool, str]:
    return (True, "") if _plain(a) < _plain(b) else (False, "output is not less than ground-truth")


@compare_func(name="<=")
async def less_equal(a: Any, b: Any, *args, **kwargs) -> tuple[bool, str]:
    return (True, "") if _plain(a) <= _plain(b) else (False, "output is not less than or equal to ground-truth")


@compare_func(name=">")
async def greater_than(a: Any, b: Any, *args, **kwargs) -> tuple[bool, str]:
    return (True, "") if _plain(a) > _plain(b) else (False, "output is not greater than ground-truth")


@compare_func(name=">=")
async def greater_equal(a: Any, b: Any, *args, **kwargs) -> tuple[bool, str]:
    return (True, "") if _plain(a) >= _plain(b) else (False, "output is not greater than or equal to ground-truth")


@compare_func(name="in")
async def is_in(a: Any, b: Any, *args, **kwargs) -> tuple[bool, str]:
    a, b = _plain(a), _plain(b)
    if not isinstance(b, (str, list, tuple)):
        raise ValueError("The second argument in comparison function `in` is not a list or a str")
    return (True, "") if a in b else (False, "ground-truth doesn't contain output")


@compare_func(name="contain")
async def contain(a: Any, b: Any, *args, **kwargs) -> tuple[bool, str]:
    a, b = _plain(a), _plain(b)
    if not isinstance(a, (str, list, tuple)):
        raise ValueError("The first argument in comparison function `contain` is not a list or a str")
    return (True, "") if b in a else (False, "output doesn't contain ground-truth")


# ---- 远程评估 ----------------------------------------------------------------------------------


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, FunctionResult):
        return _to_jsonable(x.result)
    if isinstance(x, BaseModel):
        return _to_jsonable(x.model_dump())
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def is_remote_op(op: str, context: Context) -> bool:
    """这个比较函数要不要交给远端执行:设了 ``EVAL_REMOTE_URL``,且 op 命中 ``EVAL_REMOTE_PREFIXES`` 里的某个前缀。"""
    if not context.get_env("EVAL_REMOTE_URL"):
        return False
    prefixes = tuple(p.strip() for p in context.get_env("EVAL_REMOTE_PREFIXES").split(",") if p.strip())
    return bool(prefixes) and op.startswith(prefixes)


async def remote_compare(op: str, a: Any, value: Any, op_args: Any, context: Context) -> tuple[bool, str]:
    headers = {}
    token = context.get_env("MCP_GATEWAY_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = context.get_env("EVAL_REMOTE_URL").rstrip("/") + "/eval/compare"
    payload = {"op": op, "a": _to_jsonable(a), "value": _to_jsonable(value), "op_args": _to_jsonable(op_args)}
    async with httpx.AsyncClient(timeout=float(context.get_env("EVAL_REMOTE_TIMEOUT", "600"))) as client:
        resp = await client.post(url, json=payload, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"remote evaluation of `{op}` failed ({resp.status_code}): {resp.text[:300]}")
    data = resp.json()
    return data["passed"], data.get("reason", "")


# ---- 评估器 ------------------------------------------------------------------------------------


def render_env(value: Any, context: Context | None = None) -> Any:
    """递归渲染字符串里的 ``{{VAR}}``;没有定义的变量原样保留。"""
    if isinstance(value, list):
        return [render_env(v, context) for v in value]
    if isinstance(value, dict):
        return {k: render_env(v, context) for k, v in value.items()}
    if not isinstance(value, str):
        return value
    params = dict(os.environ)
    if context:
        params.update(context.env)
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    undefined = meta.find_undeclared_variables(env.parse(value))
    return env.from_string(value).render(**{var: params.get(var, f"{{{{ {var} }}}}") for var in undefined})


class EvaluatorConfig(BaseModel):
    """``func(...) op value``。"""

    func: str  # 取值链,如 ``get(key1) -> foreach -> get(key2)``
    op: str = ""  # 比较符,如 ``=``、``<``、``github.check_repository``;空表示只要取值链不出错就算通过
    value: Any = None
    op_args: Any = None
    desc: str = ""

    def set_environ_variables(self, context: Optional[Context] = None) -> None:
        self.value = render_env(self.value, context)
        self.op_args = render_env(self.op_args, context)


class EvaluationResult(BaseModel):
    config: EvaluatorConfig
    response: str | Dict
    passed: bool
    reason: str = ""
    error: str = ""


class Evaluator:
    def __init__(self, config: str | Dict | EvaluatorConfig, context: Optional[Context] = None):
        self._context = context if context else Context()
        if isinstance(config, str):
            config = json.loads(config)
        self._config = config if isinstance(config, EvaluatorConfig) else EvaluatorConfig.model_validate(config)
        self._config.set_environ_variables(context=context)
        self._funcs = self._parse_func(self._config.func)
        assert self._config.op == "" or self._config.op in COMPARISON_FUNCTIONS, f"Unknown comparison op: {self._config.op}"

    @property
    def config(self) -> EvaluatorConfig:
        return self._config

    @staticmethod
    def _parse_func(func: str) -> List[Dict[str, Any]]:
        funcs = []
        for item in (f.strip() for f in func.split("->") if f.strip()):
            info: Dict[str, Any] = {"name": item.split("(")[0].strip()}
            assert info["name"] in EVALUATION_FUNCTIONS, f"Unknown func `{info['name']}`"
            match = re.search(r"\((.*?)\)", item)
            if match:
                info["args"] = [arg.strip() for arg in match.group(1).split(",")]
            funcs.append(info)
        return funcs

    async def execute(self, x: Any) -> Any:
        res: Any = FunctionResult(result=x)
        for func in self._funcs:
            res = await EVALUATION_FUNCTIONS[func["name"]](res, *func.get("args", []))
        return res

    async def evaluate(self, x: str | Dict) -> EvaluationResult:
        def _extract(_res: Any) -> List[FunctionResult]:
            if isinstance(_res, FunctionResult):
                return [_res]
            if isinstance(_res, (list, tuple)):
                out: List[FunctionResult] = []
                for _r in _res:
                    out.extend(_extract(_r))
                return out
            raise NotImplementedError(f"Cannot extract function results from type `{type(_res)}`")

        try:
            results = _extract(await self.execute(x))
            op, value, op_args = self._config.op, self._config.value, self._config.op_args
            for r in results:
                if op and is_remote_op(op, self._context):
                    passed, reason = await remote_compare(op, r.result, value, op_args, self._context)
                elif op:
                    passed, reason = await COMPARISON_FUNCTIONS[op](r.result, value, op_args, context=self._context)
                else:
                    passed, reason = True, ""
                if not passed:
                    return EvaluationResult(config=self._config, response=x, passed=passed, reason=reason)
            return EvaluationResult(config=self._config, response=x, passed=True)
        except json.JSONDecodeError as e:
            return EvaluationResult(config=self._config, response=x, passed=False, reason="JSON decoding error", error=str(e))
        except Exception as e:  # noqa: BLE001 - 与上游一致:取值链或比较函数出错就是这条没通过
            return EvaluationResult(config=self._config, response=x, passed=False, reason="Execution error",
                                    error=str(e) + str(traceback.format_exc()))
