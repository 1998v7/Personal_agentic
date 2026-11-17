# IMAct Execution Harness

Action Agent 的内部执行层：接收上层事件，解析工具回执，维护 Task State 与 Working Memory，调用 Action 模型，再校验并返回下一步动作。

**上层 Harness 负责业务工具执行、User Memory 服务调用、用户交互和 benchmark 评分。** 本项目提供内部事件 API、Python 接口、回执解析器和消息转换函数；解析与测试无需安装 VitaBench。

## 安装与离线验证

要求 Python 3.11 或更新版本。在项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python -m imact.cli demo
.venv/bin/python scripts/verify_inprocess.py
```

[requirements.lock.txt](requirements.lock.txt) 记录 Python 3.13 环境的依赖版本。需要使用这些版本时，先安装该文件，再运行 `pip install --no-deps -e .`。

离线 demo 和冒烟脚本使用固定模型决策。搜索回执来自留存样本，建单和支付回执是合成数据，用于验证协议、状态更新和事件去重；它们不证明真实模型或业务工具已完成任务。输出保存在被 Git 忽略的 `data/` 中。

`scripts/verify_inprocess.py` 在进程内验证 ASGI 请求，不监听端口。需要验证独立 HTTP 服务时，运行 `scripts/verify_http.py`；它会临时启动 localhost 服务，完成七轮交互并关闭服务。两个脚本均隔离调用者的 `IMACT_` 配置，避免误用真实模型。

## 配置与启动模型服务

编辑 [config/model.toml](config/model.toml)，填写 `[model]` 下的 `base_url` 和 `name`，并在环境中设置 `IMACT_MODEL_API_KEY`。密钥由 `api_key_env` 指定的环境变量读取。个人配置可另存为 `config/model.local.toml`，该文件不会被 Git 收录。

```bash
.venv/bin/python -m imact.cli serve --config config/model.toml
```

网关需支持非流式 Chat Completions。`base_url` 使用 API 前缀，例如 `https://your-model-gateway.example/v1`；适配器追加 `/chat/completions`。当前使用 `max_tokens` 参数，默认通过 prompt 要求 JSON；网关支持时可将 `response_format` 设为 `json_object`。

模型链路为：**上层提交事件 → 解析事实和组装上下文 → ChatModel → 校验联合 JSON → 返回动作**。`call_tools` 和 `query_memory` 由上层执行并回传结果。模型配置缺失时，`/health` 返回 `model_configured: false`，不会自动使用固定模型；该字段也不代表真实 API 已连通。

配置中的相对路径以 TOML 所在目录为基准，状态默认保存到 `data/service.sqlite3`。也可通过 `IMACT_CONFIG` 指定配置文件；不使用 TOML 时，按 [.env.example](.env.example) 设置进程环境变量后执行 `serve`。程序不自动加载 `.env`。TOML 中留空的 `base_url`、`name` 会回退到对应环境变量。

模型 HTTP 超时默认 90 秒，执行层单次模型决策总时限默认 120 秒。`runtime.decision_timeout_seconds` 必须大于 `model.timeout_seconds`；省略前者时，自动取 `max(120, 模型超时 + 30)`。上层客户端也需配置更长的超时，`/health` 的 `recommended_client_timeout_seconds` 提供建议值。默认 `ExecutionClient(timeout=130)`；例如模型超时改为 300 秒、决策时限自动取 330 秒时，客户端至少设为 340 秒，并按网络和存储耗时增加余量。

服务默认监听 `127.0.0.1:8010`，接口文档位于 `/docs`。设置 `IMACT_SERVICE_TOKEN` 后，业务接口要求 `Authorization: Bearer ...`。仅做固定样例联调时，可不传 TOML，显式设置 `IMACT_MODEL_MODE=demo`。

## 上层接入

上层通过 `POST /v1/execution/advance` 提交事件，收到三类动作：

| 动作 | 上层处理 |
| --- | --- |
| `call_tools` | 按稳定 `call_id` 去重执行工具，提交 `tool_results` |
| `query_memory` | 调用 Memory 服务，按 `request_id` 提交 `memory_result` |
| `reply_user` | 交给用户或用户模拟器，保留内部完成状态 |

重试必须保持请求完全相同，包括 `event_id`、`base_rev` 和 `metadata`。已提交事件会返回原动作；模型决策失败可由上层原样重试，默认累计三次失败后返回 `blocked`。未知写操作会保留为未决状态，等待上层核实。

接入前请阅读 [执行层设计与联调](docs/执行层设计与联调.md)，其中包含事件示例、错误恢复、超时约定、完成条件和已知限制。输入输出定义见 [AdvanceInput](docs/schemas/AdvanceInput.json) 与 [AdvanceOutput](docs/schemas/AdvanceOutput.json)。[外层示例](examples/outer_harness_example.py) 展示固定回执下的协议往返，接入时需替换为上层自己的执行与 Memory 路由。

## 代码入口

| 文件 | 作用 |
| --- | --- |
| `src/imact/protocol.py` | 六种输入事件与三种输出动作 |
| `src/imact/harness.py` | 事件处理、模型决策、校验与提交 |
| `src/imact/context.py`、`src/imact/store.py` | 上下文预算、SQLite 工作区和轨迹 |
| `src/imact/domain.py` | 读写规则、交易状态及完成条件 |
| `src/imact/vitabench.py` | VitaBench 回执解析与 72 个工具的来源目录 |
| `src/imact/model.py`、`src/imact/settings.py` | ActionModel 接口、ChatModel 和 TOML 配置 |
| `src/imact/prompts/action.txt` | 联合决策 prompt |
| `src/imact/schemas/action-turn-v0.3.json` | 模型联合输出格式 |
| `src/imact/api.py`、`src/imact/adapters.py` | HTTP API、上层客户端及 Chat 消息转换 |
| `tests/fixtures/vitabench/` | 静态回执、工具定义与上游许可证 |

`.run/` 提供 API、离线 demo 和测试的 PyCharm 配置，使用项目内 `.venv`。`.idea/`、运行数据、个人配置和虚拟环境均不进入交付文件。
