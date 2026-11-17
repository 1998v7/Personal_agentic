# IMAct

IMAct 的**外层**:把一个 Action 模型接入执行层(`execution_harness/`,别的团队交付),再接到 benchmark 上跑分。

```
benchmark 数据与环境 ──▶ IMAct harness(本项目) ──▶ 执行层 imact ──▶ 你的 Action 模型(GPT-5.5 / OpenRouter / Azure / 自建)
  任务、DB、工具、评分标准     用户模拟器、对话循环、评分     Task State、上下文预算、校验、完成条件
  (benchmarks/<bench>/)       事件映射、重试、Memory 路由
                               └──▶ 本地 MemoryClient / Memory 网关(M 团队)
```

分工:**benchmark 只提供任务、环境与工具、prompt 和评分标准;交互逻辑(用户模拟器、对话循环、模型调用、评分流程)全部由 IMAct 自己的 harness 实现**,不依赖任何 benchmark 的运行时包。换一个 benchmark,只需要在 `IMAct/benchmarks/` 下新增它的数据与环境,在 `IMAct/evaluation/` 下新增它的评估器。

目前接入三个 benchmark:**VitaBench**(personalization,被测对象是「Action 模型 + 执行层 + Memory」)、**τ²-bench**(airline / retail / telecom,被测对象是 agent 模型本身,不涉及 Memory,见下文「跑 τ²-bench」)、**MCP-Universe**(agent 经 MCP 网关调用真实服务,ReAct 协议,见下文「跑 MCP-Universe」)和 **PersonaMem-v3**(官方 llm_memory 模式,记忆来自外部 Memory 服务,被测对象是「Memory + action 模型」,见下文「跑 PersonaMem-v3」)。

执行层通过 `IMACT_EXEC_HARNESS_SRC` 加载,缺省指向仓库内的 `execution_harness/src`。本仓库的执行层包含上下文去重修复:任务策略、工具定义和 Memory 各保留一个输入位置,完整覆盖的候选回执使用已有结构化字段;原始事件仍完整保存。

## 目录

| 路径 | 作用 |
| --- | --- |
| `IMAct/exec_layer/` | **当前使用的主体**。`ProviderActionModel`(模型适配)、`ExecutionOuter`(事件重试、Memory 路由与幂等)、`LocalMemoryClient`(本地同步 Memory 的异步适配)、`MpsMemoryClient`(HTTP 网关)、`AsyncBridge`、配置与装配 |
| `IMAct/harness/` | **通用交互层,与 benchmark 无关**。`UserSimulator`(LLM 扮演用户,可带自己的工具)、`HarnessAgent`(消息 ↔ 执行层事件)、`NativeAgent`(原生 function-calling,不经过执行层)、`Conversation`(用户⇄助手⇄环境的对话循环)、`run_episode`(一段对话的完整生命周期)、`ReActAgent`(文本 JSON 协议的 ReAct 循环)、`mcp`(通用 MCP 客户端:经网关的 SSE、同步句柄、工具集)、`LLMCaller`(用户/裁判角色调用 `models/` 层的同步封装)、`batch`(任务级重试 / 断点续跑 / 并发 / 汇总) |
| `IMAct/evaluation/` | 评估:`base.py` 放通用类型(`RewardInfo`、`JudgeError`),每个 benchmark 一个评估模块(`vitabench.py`:滑动窗口 rubric 评分;`tau2.py`:环境终态 / 动作 / 必须告知的信息 / NL 断言;`mcp_universe.py`:`func op value` 评分引擎与远程评估) |
| `IMAct/benchmarks/vitabench/` ([README_VITABENCH.md](README_VITABENCH.md)) | **VitaBench 专属,后期可整体删除**:任务数据模型与读取、业务域环境(delivery / instore / ota 的 DB 与工具)、prompts、`personalization.py`(把 subtask 变成 harness 能跑的东西)、入口 `run.py` / `diagnose.py`。上游来源与许可证见该目录的 `ORIGIN`、`LICENSE` |
| `IMAct/benchmarks/tau2/` ([README_TAU2.md](README_TAU2.md)) | **τ²-bench 专属,后期可整体删除**:airline / retail / telecom 三个 domain 的 DB、工具、任务、人设与 prompt(模板文本从上游原样导出)、入口 `run.py`。数据放 `data/tau2/`(同 VitaBench,不入库) |
| `IMAct/benchmarks/mcp_universe/` ([README_MCP_UNIVERSE.md](README_MCP_UNIVERSE.md)) | **MCP-Universe 专属,后期可整体删除**:212 个任务 JSON 与各领域设置、ReAct prompt 模板、各领域评分函数(`checks/`)、清理逻辑、入口 `run.py`。数据随源码(逐字节取自上游);服务不在本地运行,经 MCP 网关访问 |
| `IMAct/benchmarks/personamem_v3/` ([README_PERSONAMEM_V3.md](README_PERSONAMEM_V3.md)) | **PersonaMem-v3 专属,后期可整体删除**:移植的官方评测代码 `pm3/`(题型 runner、prompt、评分、裁判、汇总,只改了导入与编码)、记忆来源适配(每题向 Memory 服务 query)、数据下载、入口 `run.py`。数据放 `data/personamem_v3/`(不入库) |
| `IMAct/models/` | 统一模型清单、检查/探测命令,以及 Chat Completions / Responses / Anthropic Messages 等 provider |
| `IMAct/retry.py`、`http_util.py` | 重试策略与 HTTP 错误映射,模型层与 Memory 客户端共用 |
| `tests/` | 离线回归测试,标准库 `unittest` |
| `models.yaml` | 所有角色的模型名、协议、URL、鉴权变量和请求参数 |
| `.env.example` → `.env` | 密钥模板 → 本地密钥(不提交) |
| `config.vitabench.example.yaml` | 执行层、Memory 和重试配置模板 |

## 环境

> 下面的命令按 **Windows + Git Bash** 写(`.venv/Scripts/python -X utf8`)。macOS / Linux 把它换成 `.venv/bin/python`,不需要 `-X utf8`。

Python ≥ 3.11(本机 3.14 验证过)。项目内 `.venv` 装了两边的依赖:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-server.txt
# 只含执行层、IMAct 和 VitaBench 环境/工具实际 import 的包;不再需要 VitaBench 自己的 pandas / plotly / scikit-learn / openai 等
```

**Windows 请始终加 `-X utf8`**(或设 `PYTHONUTF8=1`):执行层是在 macOS 上写的,默认的 GBK 编码会让它读取 UTF-8 文件时失败。

## 测试

```bash
.venv/Scripts/python -X utf8 -m unittest discover -s tests -t .
```

测试不需要数据集,也不需要任何真实 LLM:用执行层自带的 `DemoModel` 当脚本化模型。

## 模型配置:只改根目录 models.yaml

三个角色现在共用一份 [models.yaml](models.yaml)。`config.vitabench.yaml` 通过 `models_file: models.yaml` 引用它,不再填写模型 URL 或采样参数。路径相对配置文件解析。

```yaml
roles:
  action: qwen3.8-27b       # 被测 Action 基模
  user: gpt-5.6-sol        # 用户模拟器
  judge: claude-opus-4-6   # 裁判;当前接口排查中

models:
  qwen3.8-27b:             # 本地别名,供 roles 和命令行选择
    provider: openai_compatible
    model: Qwen3.8-27B     # 请求中发给服务端的精确模型名
    base_url: https://antchat.alipay.com/v1/chat/completions
    api_key_env: ANTCHAT_API_KEY
    timeout_s: 120
    options: {merge_system_messages: true}
    params: {stream: false, max_tokens: 4096}
    extra_body: {}
```

Qwen 只要求 `.env` 中的 `ANTCHAT_API_KEY=你的密钥`,不要求 SOFA-TraceId / SOFA-RpcId。入口自动加载 `models.yaml` 同目录的 `.env`;已经导出的 shell 变量优先。已有 `.env` 时不要用模板覆盖。

`merge_system_messages` 合并开头连续的 system 文本,保留执行层协议和任务策略。AntChat 的实际 Action 请求验证表明,两条 system 消息会触发 HTTP 400;这个选项仅为 Qwen 开启。

| 字段 | 用途 |
| --- | --- |
| `provider` | 明确选择协议,不按模型名猜测;`openai_compatible` 为 Chat Completions,`openai_responses` 为 Responses,`anthropic_messages` 为 Anthropic Messages |
| `base_url` | 可填完整请求 URL,也可填 `/v1` 根地址;`list` 会展示最终 `endpoint` |
| `api_key_env` | Bearer key 所在环境变量名;不在 YAML 写真实密钥 |
| `headers` / `headers_env` | 固定 header / header 名到环境变量名的映射,例如 `headers_env: {X-LY-TOKEN: X_LY_TOKEN_GPT}` |
| `params` | `stream`、`temperature`、`max_tokens`、`max_output_tokens` 等请求参数 |
| `extra_body` | 服务专有请求字段;合并顺序为默认 `params` → 本次调用参数 → `extra_body` |
| `timeout_s` / `ca_bundle` | 请求超时 / 可选的受信 PEM CA 文件路径;相对路径也相对模型配置文件 |
| `options` | provider 选项,如 Qwen 的 `merge_system_messages`、OpenRouter 路由等 |

当前 Chat Completions provider 支持 `stream: false`;Responses 和 Anthropic Messages provider 支持流式与非流式。Claude 在这份配置中显式设置 `stream: true`。模型名、消息和鉴权不能被请求参数覆盖。拼错字段、角色指向不存在的模型会直接报错;只检查所选角色的密钥,闲置模型不要求配置凭据。

## 查看、检查和单模型探测

```bash
# 显示所有模型、角色和最终接口;不显示密钥
.venv/Scripts/python -X utf8 -m IMAct.models list

# 只检查 Action 的本地配置和环境变量,不访问远端
.venv/Scripts/python -X utf8 -m IMAct.models check --role action

# 只真实调用 Action;不会调用用户模拟器或裁判
.venv/Scripts/python -X utf8 -m IMAct.models probe --role action
```

`--role user` / `--role judge` 选择其他角色;`--model 模型别名` 只覆盖这一次选择。`--models /path/to/models.yaml` 选择其他清单。`probe` 要求返回非空正文并正常结束,保存响应、模型名、request ID、usage 和脱敏配置到 `results/model-probe-*.json`。这证明接口可调用,不等于 benchmark 得分。

## 跑 VitaBench

被测对象是「Action 模型 + 执行层 + Memory」。**完整的准备、命令、参数、结果解读与常见问题见 [README_VITABENCH.md](README_VITABENCH.md)。** 最短路径:

```bash
# 不发请求的检查
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.vitabench.run --config config.vitabench.yaml --no-memory --num-tasks 1 --max-subtasks 1 --check
```

Memory 用的是同级 `IMAct_memory` 的本地 MemoryClient(见 [本地 Memory 联调](LOCAL_MEMORY.md)),也可以走 HTTP 网关或用 `--no-memory` 关闭。

运行时打印三个角色的脱敏配置,并保存 `models.resolved.<时间戳>.json`(配置文件 SHA256、角色别名、实际模型名、协议、URL 和参数)。`config.vitabench.yaml` 只管理执行层、Memory、重试;模型调用参数放 `models.yaml`,`models_file` 与旧的内联 `model` / `model_call.params` 不允许混用。

运行入口**必须**使用统一模型配置。旧的内联 `model` 运行 YAML 和 VitaBench 自己的模型 YAML(`VITA_MODEL_CONFIG_PATH`、`vita_models.openrouter.yaml`)已不再被读取,可以删除。旧脚本 `scripts/smoke_matrixllm.sh` 仍可用,是对入口的薄封装。

本项目不再调用 VitaBench 的 orchestrator、用户模拟器和评测代码,这些由 `IMAct/harness/` 与 `IMAct/evaluation/` 实现;对话循环、两个 system prompt 和滑动窗口评分都与上游逐项对拍过(同输入下消息、终止原因、prompt 文本、评分一致)。

## 跑 τ²-bench(airline / retail / telecom,6 语种)

跑官方 **text-v1**:6 种语言 × 3 个 domain = 18 个 cell,每个 cell 50 个冻结 task,共 900 个。被测对象是 agent 模型本身(原生 function-calling,prompt 与上游一致),**不涉及 Memory,不经过执行层**,分数可与 τ² 榜单对照。**完整的准备、命令、参数、结果解读、评分口径与费用估算见 [README_TAU2.md](README_TAU2.md)。** 最短路径:

```bash
# 不发请求的检查(会列出 18 个 cell 与 900 个 task)
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.tau2.run --models models.yaml --domain all --language all --check
# 冒烟:每个 cell 跑 1 个 task
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.tau2.run --models models.yaml --domain all --language all --num-tasks 1 --max-concurrency 3 --out results/tau2-smoke
```

上游没有 hotel domain。上游代码与数据以 MIT 许可证引入,来源见 `IMAct/benchmarks/tau2/ORIGIN`。

## 跑 MCP-Universe(5 个领域,212 个 task)

agent 通过 MCP 网关调用**真实服务**(搜索、地图、行情、GitHub、浏览器、Notion、天气)回答问题,评估器再逐条核验答案。被测对象是 agent 模型本身(ReAct 文本协议,prompt 与上游一致),**不涉及 Memory,不经过执行层**。需要在 `.env` 里配置 `MCP_GATEWAY_ADDRESS` / `MCP_GATEWAY_TOKEN`(以及可选的远程评估),**完整的准备、命令、参数、指标、评分口径与注意事项见 [README_MCP_UNIVERSE.md](README_MCP_UNIVERSE.md)。** 最短路径:

```bash
# 不发模型请求的检查(会连一下网关,列出每个服务的工具数)
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.mcp_universe.run --models models.yaml --domain all --check
# 冒烟:每个领域跑 1 个 task(每个领域的第一个 task 都是只读的)
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.mcp_universe.run --models models.yaml --domain all --num-tasks 1 --max-concurrency 3 --out results/mcpu-smoke
```

其中 43 个 task(`repository_management` 等)会在真实 GitHub / Notion 里创建并事后删除资源,默认也会跑,务必使用专用测试账号;只想跑只读的 169 个加 `--skip-side-effects`。不含 3d_design(Blender)、MCPMark、deep research。上游代码与数据以 Apache-2.0 许可证引入,来源见 `IMAct/benchmarks/mcp_universe/ORIGIN`。

## 跑 PersonaMem-v3(99 个用户,约 1.58 万道题)

只跑官方的 `llm_memory` 模式:每道题向 **Memory 服务**(统一调用接口,`bench=personamem_v3`,与 VitaBench 同一个网关)`query` 一次,返回的记忆接上日历后交给 action 模型作答,再按官方规则评分(裁判 = `judge` 角色)。题目、prompt、评分与官方逐行一致。**数据下载、网关配置、命令、指标口径与费用估算见 [README_PERSONAMEM_V3.md](README_PERSONAMEM_V3.md)。** 最短路径:

```bash
# 下载两个 persona 的数据
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.personamem_v3.download --users 25 36
# 检查(不调模型;向 Memory 网关发 1 次只读 query)。网关只在内网可达
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.personamem_v3.run --models models.yaml --users 25 --check
# 冒烟:一个 persona 的前 10 道题
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.personamem_v3.run --models models.yaml --users 25 --num-queries 10 --out results/pm3-smoke
```

上游代码以 MIT 许可证引入,数据为 CC-BY-NC-4.0,来源见 `IMAct/benchmarks/personamem_v3/ORIGIN`。

## 把结果打印成表

每个 bench 跑完都会把结果打印成表(分数统一是 0-100,保留两位小数):τ² 是「语言 × domain」的 pass^1 矩阵;MCP-Universe 第一张表与论文同列(各领域 SR + Overall SR / AE / AS);VitaBench 是总分与外卖 / 到店 / 酒旅的拆分;PersonaMem-v3 是 ALL / ALL-JUDGE / 各类别与各题型。

任意一个结果目录都可以事后重新打印,不重跑、不调用模型:

```bash
.venv/Scripts/python -X utf8 -m IMAct.report results/tau2-full results/mcpu-full results/pm3-25
```

它从目录里的每题结果文件重新汇总,所以断点续跑过的目录、旧版本跑出的目录也能得到完整、口径一致的表(`--from-summary` 则只读 `summary.json`)。

## 生成完整链路可视化

准备 VitaBench 源码、数据集和 `.env` 后，下面的诊断命令会真实调用 Action、用户模拟器和 Memory，并执行 VitaBench 模拟工具；跳过裁判，不产生正式评分。`MEMORY_GATEWAY_URL` 填当前可用的 Memory 网关。

```bash
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.vitabench.diagnose --config config.vitabench.yaml --subtask-ids sub_A891207_1 --memory-url "$MEMORY_GATEWAY_URL" --out results/chain-demo --check
# 配置与样本检查通过后，去掉 --check 执行任务
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.vitabench.diagnose --config config.vitabench.yaml --subtask-ids sub_A891207_1 --memory-url "$MEMORY_GATEWAY_URL" --out results/chain-demo

# 对已有 trace.jsonl / exec.sqlite3 所在目录重新生成页面，不调用远端模型
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.vitabench.episode_report results/chain-demo/sub_A891207_1
```

打开 `results/chain-demo/index.html` 查看任务入口，或打开样本目录的 `episode.html` 查看完整链路。页面展示用户、Memory 请求/输出、模型输入/输出、校验重试、工具回执、Task State 与 Working Memory 提交前后；`context-injection-verification.json` 核对提交状态与模型消息中的 `current_task` / `active_notes`。

## 真实验证记录

下列 `results/` 链接指向原调试工作区的本地产物，不随源码提交；新克隆的仓库需运行诊断后生成。完整链路导出入口为 `IMAct.benchmarks.vitabench.episode_report`，运行方法见下方。

完整任务与跨场景回放见 [可视化入口](results/full-task-20261110/index.html)。新版 `episode.html` 按实际时间展示全部采集节点:用户、Memory、完整 Action 输入/输出、校验重试、Task State / Working Memory 写入前后、工具请求/回执和最终证据;`rounds.html` 为辅助轮次对照。已验证外卖下单支付、酒店两晚分别下单支付、到店双人预约三类主链路。原始酒店提前 STOP 的未完成运行单独保留;复验使用明确标记的诊断用户结束提示,无正式 benchmark 分数。具体分母与边界见 [完整任务观察记录](results/full-task-20261110/OBSERVATIONS.md)。

Action 输入去重已修复,并使用原三轮事件完成离线回放:任务策略、工具定义和 Memory 每次请求各保留一份,完整覆盖的候选回执通过来源引用关联已有字段。三轮消息文本字节分别减少 38.19%、38.91%、17.47%,100 个候选及 Memory 全文保持完整。执行层 94 项、外层 102 项测试通过。修复后未新增远端模型调用;对比、完整新输入及验证边界见 [输入去重修复报告](results/context-compaction-20261110/comparison.md)。

Memory 首轮预取已接入默认运行配置。最新单轮联调使用真实任务 `sub_A891207_1`:先调用 Memory,将返回的 498 字符全文注入首次 Action 请求,再调用一次 Qwen。模型输出商品搜索和工作地址解析动作,通过 Harness 校验,停在动作输出处。完整 Memory 请求/返回、Action 输入/输出与核验见 [Memory + Action 首轮记录](results/memory-action-one-round-20261110-165440/action_agent_round.md)。网关出现过 3 次超时,成功前没有调用 Action;本次共 1 次 Action 调用,未执行业务工具或评分。

Qwen AntChat 文本接口和执行层首轮 Action 均已真实验证:前者见 `results/qwen-model-config-probe.json`,后者见 `results/qwen-action-probe-20261110-153912/result.json`。首轮生成搜索川菜店的动作并通过校验,未执行该业务工具。GPT/Memory 的历史运行、Claude 空响应及评分失败边界见 [真实运行记录](REAL_MODEL_RUN.md)。Claude 服务端仍在排查,此次不调用它们,也不据此生成 Qwen benchmark 分数。

## 部署到服务器

VitaBench 的环境与工具代码在 `IMAct/benchmarks/vitabench/`(不含 125MB 题目)。`bash scripts/setup_server.sh` 会建 `.venv`、装依赖、跑测试和 `--check`。数据集放在 `data/vitabench/tasks.json`。按 `models.yaml` 中所选角色填写 `.env`(参考 `.env.example`),不要提交真实密钥。
