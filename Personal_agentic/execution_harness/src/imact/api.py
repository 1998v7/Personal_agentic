import hmac
import os

from fastapi import Depends, FastAPI, Header
from fastapi.responses import JSONResponse

from .context import ContextAssembler
from .errors import HarnessError
from .harness import ExecutionHarness
from .model import ChatModel
from .protocol import AdvanceInput, AdvanceOutput
from .settings import Settings, resolve_decision_timeout
from .store import SQLiteStore


def create_app(
    harness: ExecutionHarness | None = None, token: str | None = None, config: str | None = None
) -> FastAPI:
    model_ready = True
    if harness is None:
        config = config or os.getenv("IMACT_CONFIG")
        settings = Settings.load(config) if config else None
        runtime = settings.values.get("runtime", {}) if settings else {}
        mode = "chat" if settings else os.getenv("IMACT_MODEL_MODE", "chat")
        model_timeout = (
            settings.values["model"].get("timeout_seconds", 90)
            if settings
            else os.getenv("IMACT_MODEL_TIMEOUT", "90")
        )
        decision_timeout = resolve_decision_timeout(
            model_timeout,
            runtime.get("decision_timeout_seconds")
            if settings
            else os.getenv("IMACT_DECISION_TIMEOUT"),
        )
        if mode == "demo":
            from .demo import DemoModel

            model = DemoModel()
        elif mode == "chat":
            try:
                model = settings.model() if settings else ChatModel.from_env()
            except HarnessError as exc:
                if exc.code != "model_not_configured":
                    raise
                model_ready = False

                class UnconfiguredModel:
                    async def decide(self, context):
                        raise HarnessError(
                            "model_not_configured", "请填写模型配置文件或模型环境变量", 503
                        )

                model = UnconfiguredModel()
        else:
            raise ValueError("IMACT_MODEL_MODE must be chat or demo")
        harness = ExecutionHarness(
            SQLiteStore(
                settings.resolve("runtime", "output_dir") / "service.sqlite3"
                if settings
                else os.getenv("IMACT_DB_PATH", "data/workspace.sqlite3")
            ),
            model,
            decision_timeout=decision_timeout,
            context=ContextAssembler(
                runtime.get(
                    "context_max_bytes", int(os.getenv("IMACT_CONTEXT_MAX_BYTES", "240000"))
                )
            ),
            max_decision_attempts=runtime.get(
                "max_decision_attempts", int(os.getenv("IMACT_MAX_DECISION_ATTEMPTS", "3"))
            ),
        )
    else:
        mode = "injected"
    secret = os.getenv("IMACT_SERVICE_TOKEN", "") if token is None else token
    app = FastAPI(
        title="IMAct Execution Harness",
        version="0.2.0",
        description="内部事件 API。业务工具和 User Memory 由外层执行。",
    )
    app.state.harness = harness

    async def authorize(authorization: str | None = Header(default=None)):
        if secret and not hmac.compare_digest(authorization or "", "Bearer " + secret):
            raise HarnessError("unauthorized", "需要有效的服务令牌", 401)

    @app.exception_handler(HarnessError)
    async def handle_error(request, error: HarnessError):
        return JSONResponse(status_code=error.status, content={"error": error.as_dict()})

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model_mode": mode,
            "model_configured": model_ready,
            "protocol_version": "0.2.0",
            "decision_schema_version": "0.3",
            "decision_timeout_seconds": harness.decision_timeout,
            "recommended_client_timeout_seconds": harness.decision_timeout + 10,
        }

    @app.post(
        "/v1/execution/advance",
        response_model=AdvanceOutput,
        response_model_exclude_none=True,
        dependencies=[Depends(authorize)],
    )
    async def advance(request: AdvanceInput):
        return await harness.advance(request)

    @app.get("/v1/execution/tasks/{task_id}", dependencies=[Depends(authorize)])
    async def task(task_id: str):
        return harness.store.inspect(task_id)

    @app.get("/v1/execution/tasks/{task_id}/trace", dependencies=[Depends(authorize)])
    async def trace(task_id: str):
        return harness.store.trace(task_id)

    @app.get("/v1/execution/tasks/{task_id}/evidence", dependencies=[Depends(authorize)])
    async def evidence(task_id: str, ref: str):
        return harness.store.evidence(task_id, ref)

    return app
