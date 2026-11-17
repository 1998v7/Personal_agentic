import argparse
import asyncio
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="IMAct 执行层 Harness")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="运行固定决策的离线联调样例")
    demo.add_argument("--output", default="data/demo")
    serve = commands.add_parser("serve", help="启动内部事件 HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8010, type=int)
    serve.add_argument("--config", help="模型 TOML 配置，例如 config/model.toml")
    export = commands.add_parser("export-schema", help="导出内部输入输出 JSON Schema")
    export.add_argument("--output", default="docs/schemas")
    recover = commands.add_parser("recover", help="旧进程停止后将中断事件标为可重试")
    recover.add_argument("--db", required=True)
    recover.add_argument("--task-id", required=True)
    recover.add_argument("--event-id", required=True)
    recover.add_argument("--confirm-worker-stopped", action="store_true", required=True)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        if args.config:
            from .api import create_app

            uvicorn.run(create_app(config=args.config), host=args.host, port=args.port)
        else:
            uvicorn.run("imact.api:create_app", factory=True, host=args.host, port=args.port)
    elif args.command == "demo":
        from .demo import run_demo

        print(json.dumps(asyncio.run(run_demo(args.output)), ensure_ascii=False, indent=2))
    elif args.command == "export-schema":
        from .protocol import AdvanceInput, AdvanceOutput

        path = Path(args.output)
        path.mkdir(parents=True, exist_ok=True)
        for model in [AdvanceInput, AdvanceOutput]:
            (path / f"{model.__name__}.json").write_text(
                json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n"
            )
    else:
        from .store import SQLiteStore

        SQLiteStore(args.db).recover_interrupted(args.task_id, args.event_id)
        print("已标记可重试，请重新投递完全相同的请求。")


if __name__ == "__main__":
    main()
