# MCP-Universe 接入(5 个领域,212 个 task)

用 IMAct harness 跑 [MCP-Universe](https://github.com/SalesforceAIResearch/MCP-Universe) 的 **mcpuniverse** 评测集:agent 通过 MCP 协议调用**真实服务**(搜索、地图、行情、GitHub、浏览器、Notion、天气)回答一个问题,再由评估器对答案逐条打分。

被测对象是 **agent 模型本身**:ReAct 文本协议(模型每轮输出一段 JSON,`thought` + `action` 或 `answer`),prompt 与上游一致。**不涉及 Memory,也不经过执行层。**

```
任务 JSON ──▶ IMAct harness ──▶ ReAct agent ──▶ 模型(models.yaml 的 action 角色)
(问题、服务、评估)   │                 └─ 工具调用 ──▶ MCP 网关 ──▶ 真实服务
                    └─ 评估(逐条) ──▶ 评分引擎 ──▶ 查询真实服务 / 裁判(judge 角色)/ 远程评估
```

> 命令都在项目根目录(`Personal_agentic/`)的 Git Bash 里运行。`-X utf8` 的作用见文末「常见问题」。

## 1. 范围

| 领域 | task 数 | 用到的服务 | 说明 |
| --- | --- | --- | --- |
| `web_search` | 55 | google-search、fetch(部分含 notion) | 答案由**裁判模型**对照标准答案判定 |
| `location_navigation` | 45 | google-maps(部分含 playwright、weather) | 路线 / 地点 / 距离,评估再调一次地图 API 核验 |
| `financial_analysis` | 40 | yfinance、calculator | 评估现场用行情数据算出标准答案再比对 |
| `browser_automation` | 39 | playwright(部分含 date、notion) | 浏览器操作;比价类评估现场爬网页 |
| `repository_management` | 33 | github(部分含 playwright) | **会在 GitHub 里创建仓库 / 分支 / PR**,评估再查回来,事后删除 |

这是上游官方的划分(五个领域的 yaml 里各自引用的 task,其中 25 个 `multi_server` task 分散在各领域)。**不含**:3d_design(Blender)、MCPMark、deep research。

> 3d_design 需要在本机装 Blender,MCPMark / deep research 是上游里另外两套评测,本次接入都没做。

## 2. 准备

```bash
# 环境(只需一次)。Python >= 3.11
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-server.txt

# 密钥与网关:复制 .env.example 为 .env,填下面的变量(.env 不会被提交)
cp .env.example .env        # 已有 .env 就不要覆盖
```

### 2.1 MCP 网关(必需)

agent 的所有工具调用和评分里查询服务的步骤,都经**一个 MCP 网关**:`<网关地址>/<server 名>/sse`。服务(github / google-maps / yfinance / playwright / notion …)的密钥和运行环境都在网关那一侧,本机只需要:

| 变量 | 作用 |
| --- | --- |
| `MCP_GATEWAY_ADDRESS` | 网关地址,如 `https://www.example.com` |
| `MCP_GATEWAY_TOKEN` | 网关的 Bearer token(没有鉴权就留空) |

### 2.2 远程评估(强烈建议)

有两类评估**不经过 MCP**,而是评估器自己联网:`yfinance.*`(拉行情算标准答案,本机直连 Yahoo 会被限流)和 `playwright.check_*`(爬航班 / 酒店价格)。网关那一侧可以提供一个 `/eval/compare` 接口,让这些比较在那边执行:

| 变量 | 作用 |
| --- | --- |
| `EVAL_REMOTE_URL` | 远程评估服务地址(通常和网关同一个) |
| `EVAL_REMOTE_PREFIXES` | 哪些评估交给远端,逗号分隔的前缀,如 `yfinance.,playwright.check_` |

不配则在本机执行,这时 `--check` 会提示:yfinance 要能直接访问 Yahoo,playwright 要先 `playwright install chromium`。天气类评估(`weather.*`,5 条)总是在本机直接访问 `api.weather.gov`。

### 2.3 只有跑「会创建资源」的 task 才需要

`repository_management`(33 个)以及 `web_search` / `browser_automation` 里带 Notion 的各 5 个 task,共 **43 个**会在真实服务里创建东西,跑完后按 task 的 `cleanups` 删除。要跑它们需要:

| 变量 | 作用 |
| --- | --- |
| `GITHUB_PERSONAL_ACCOUNT_NAME` | **必填**。GitHub 账号名。评估据此查询仓库,清理时**只会删除它名下的仓库**。账号名可以用网关的 GitHub 服务 `get_me` 查到 |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | 清理时直接调 GitHub REST API 删仓库用。**网关的 GitHub 服务没有删除仓库的工具**,所以密钥只在网关那一侧时,仓库不会被自动删除:结果文件的 `cleanups` 里会写明,`--check` 也会提示,需要手动删(或在本机填这个令牌) |
| `NOTION_ROOT_PAGE` | **必填(Notion task)**。agent 在这个父页面下建页,写成「页面名-ID」;该页面必须已共享给 Notion 集成 |
| `NOTION_API_KEY` | 可选。没有时,Notion 页面经网关的 Notion 服务归档(`API-patch-page`) |

> ⚠ **务必使用专用的测试账号。** agent 会真的创建仓库;清理会删除与 agent 创建的仓库**同名**的仓库——如果账号里本来就有同名仓库,它也会被删。这些 task 默认**会跑**;只想跑只读的 169 个,加 `--skip-side-effects`。

> **Notion 集成要先有一个共享给它的页面**(页面右上角 ⋯ → 连接 → 选 Notion 集成),把它的「名字-ID」写进 `NOTION_ROOT_PAGE`,如 `MCP_universe_test-417e6a7b3754802e960fc9daca37d6ff`(页面链接末尾那一段)。
>
> 网关上的 Notion MCP server 比上游新,会拒绝上游评估器的搜索参数写法;`checks/notion.py` 的 `notion__search_page` 因此改成传纯字符串 `query`(见 `ORIGIN`),这是评分函数里与上游唯一的逻辑差别。

### 2.4 选模型

`models.yaml` 的 `roles` 里:

| 角色 | 作用 |
| --- | --- |
| `action` | **被测的 agent 模型**(同时负责工具回执摘要:`web_search` 领域会让模型先把每次工具返回摘要成 500 词以内再放进历史) |
| `judge` | 裁判。**只有 `web_search`**(50 个 task)用它对照标准答案判对错 |

`user` 角色在这个 benchmark 里用不到。也可以在命令行临时覆盖:`--agent-model <别名>`、`--judge-model <别名>`。

## 3. 运行

**先检查**(不发任何模型请求,不创建任何资源;会连一下网关,列出每个要用的服务的工具数):

```bash
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.mcp_universe.run --models models.yaml --domain all --check
```

检查会确认:模型密钥、网关配置与连通、每个服务可用、`{{变量}}` 是否都有值、哪些评估需要本机联网、有多少 task 会创建资源。

对 playwright 服务,检查还会真的打开一次空白页:服务在线但浏览器没装好时(网关容器里 `@playwright/mcp` 升级后常见),会在这里报出来,而不是让 49 个用到浏览器的 task 全部静悄悄地 0 分。

**冒烟**:每个领域只跑 1 个 task(只读的领域,不创建任何资源):

```bash
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.mcp_universe.run --models models.yaml \
  --domain all --num-tasks 1 --max-concurrency 3 --out results/mcpu-smoke
```

**完整跑**(默认:全部 212 个,含 43 个会创建资源的 task):

```bash
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.mcp_universe.run --models models.yaml \
  --domain all --max-concurrency 3 --out results/mcpu-full
```

**默认就是全部 212 个**(含会创建资源的 43 个;先确认 2.3 的变量与「专用测试账号」)。只想跑只读的 169 个:

```bash
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.mcp_universe.run --models models.yaml \
  --domain all --skip-side-effects --max-concurrency 3 --out results/mcpu-readonly
```

**常用选择**:

```bash
# 只跑金融和搜索
... --domain financial_analysis web_search --out results/mcpu-fin-search
# 指定 task(名字是文件名,只能配合单个 --domain)
... --domain financial_analysis --task-ids yfinance_task_0001 yfinance_task_0007
# 换被测模型
... --domain all --agent-model openrouter-gpt-4.1
# 同一批 task 多跑几次(算平均与方差)
... --num-trials 3 --seed 7
# 额外加载一个 .env(不覆盖已有的环境变量)
... --env-file other.env
```

### 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--models` | 必填 | `models.yaml` 路径 |
| `--domain` | 必填 | 五个领域之一或多个,`all` = 全部 |
| `--agent-model` / `--judge-model` | `roles` | 临时覆盖 |
| `--task-ids ...` | 全部 | 按 task 名选,只能配合单个 `--domain` |
| `--num-tasks N` | 全部 | 每个领域只取前 N 个(冒烟用) |
| `--num-trials` | 1 | 每个 task 跑几次 |
| `--max-concurrency` | 1 | **同一个领域内**同时跑几个 task(领域之间按顺序)。会创建资源的 task 始终一次一个 |
| `--skip-side-effects` | 否 | 不跑会创建并删除资源的 43 个 task(只跑只读的 169 个) |
| `--max-task-retries` | 3 | task 整体失败(连不上网关、模型请求持续失败)后的重试次数 |
| `--model-retries` | 3 | 单次模型请求(限流、超时、空回复)的重试次数 |
| `--call-timeout` | 60 | 单次工具调用的超时秒数 |
| `--eval-timeout` | 900 | 单条评估的超时秒数 |
| `--env-file` | 无 | 额外加载的 `.env`,可重复;不覆盖已有变量 |
| `--seed` / `--rerun` | 无 / 否 | 随机种子 / 忽略已有结果重跑 |
| `--check` | 否 | 只检查,不发请求 |
| `--out` | `results/mcpu-<时间戳>` | 结果目录 |

## 4. 断点续跑

每个 task 跑完立刻写一个结果文件。中途中断后,**用完全相同的命令重新运行**(同一个 `--out`),已完成的 task 会跳过。要全部重跑加 `--rerun`。

跑完会把结果打印成表;事后重新打印(不重跑):`.venv/Scripts/python -X utf8 -m IMAct.report <结果目录>`。断点续跑后的汇总包含以前跑完的部分,不只是这次新跑的。

## 5. 结果怎么看

```
results/mcpu-full/
├── summary.json                       # 总汇总:SR / AE / 平均轮数 + 每个领域的明细
├── financial_analysis/
│   ├── yfinance_task_0001.trial0.json # 每个 task 一份:完整过程与每条评估的判定
│   └── summary.json
├── trace.jsonl                        # 所有模型请求(agent / 裁判)的完整轨迹
├── run.log
└── models.resolved.*.json             # 本次实际用的模型配置(不含密钥)
```

**指标**(与上游论文一致):

| 指标 | 含义 |
| --- | --- |
| **SR** | 成功率:一个 task 的**全部**评估都通过才算成功,所有 task 取平均 |
| **AE** | 评估通过率:每个 task 里通过的评估占比,所有 task 取平均(比 SR 宽松,能看出"做对了一部分") |
| **AS** | 平均步数:**只在成功的 task 上**取平均(论文口径;没有成功的 task 时为空,不是 0)。`AS` 按 ReAct 轮数;`summary.json` 里另有 `AS_llm_calls` 按模型调用次数(含工具回执摘要),与上游 `report.py` 的 LLM Call Count 同口径 |
| 平均轮数 | 所有 task 的平均 ReAct 轮数(含失败的;失败的 task 常常用满轮数,所以它比 AS 大) |

**单个 task 的结果文件**:`evaluations` 是每条评估的判定(`passed`、`reason`、出错时的 `error`);`history` 是 agent 的完整过程(每一步的想法、调用的工具和参数、工具返回的全文、最终回答);`steps` 是每轮模型的原始输出;`tool_calls` 是实际执行过的工具调用;`cleanups` 是清理的结果。看一个 task 为什么失败,先看 `evaluations[*].reason`,再看 `history`。

- `失败`:task 在重试用尽后仍然出错(例如网关持续连不上),它**没有分数**,不会被当成 0 分算进平均。
- `评分错误`:对话跑完了,但评估过程中**网关或裁判不可用**。该 task 记 0 分并标 `evaluation_error`,`infra_errors` 里有具体原因。这个数不为 0 时,SR 偏低,应先查原因再看分数。

## 6. 评分口径

每个 task 有若干条评估,每条是 `取值链 比较函数 期望值`:

```json
{"func": "json -> get(routes) -> foreach -> get(cities_visited) -> len", "op": "=", "value": 4}
{"func": "json", "op": "yfinance.check_portfolio_task_output", "op_args": {"tickers": ["MSFT"], "start_date": "2023-01-09", ...}}
{"func": "raw",  "op": "github.check_file_content", "op_args": {"owner": "{{GITHUB_PERSONAL_ACCOUNT_NAME}}", "repo": "travel-planner-app", ...}, "value": "..."}
```

取值链把 agent 的最终回答(一个 JSON 字符串)变成值;比较函数有三类:纯比对(`=`、`<`、`in`、`contain` …)、**查询真实服务再比对**(`github.*`、`google_maps.*`、`notion.*` 经网关;`weather.*` 直连)、**现场算标准答案再比对**(`yfinance.*`、`playwright.check_*`,可交给远端)。一个 task 的所有评估都通过才算成功。

与上游一致的口径:
- agent 的最终回答是模型输出里 `answer` 字段:字符串原样;dict 序列化成 JSON;其它类型用 `str()`。用满轮数没有 `answer`,回答是一句固定的道歉文案,评估自然全部不通过。
- 取值链出错(回答不是合法 JSON、缺字段)记为这条没通过,不抛异常。
- `{{GITHUB_PERSONAL_ACCOUNT_NAME}}` 这样的变量在评估前用环境变量替换;没有定义的原样保留(`--check` 会报出来)。

与上游不同的两点,都是为了把「基础设施故障」和「agent 答错」分开:
- 上游的 github / notion 评估函数遇到网关异常时多半吞掉、当作"没找到"。这里评估照旧判为没通过(分数与上游一致),但把这次故障记进 `infra_errors` 并标 `evaluation_error`。
- 裁判(`web_search`)不再直连 OpenAI,而走 `models.yaml` 的 `judge` 角色;裁判不可用同样记为评分错误。

## 7. 费用与耗时

每个 task 是 N 轮模型调用(`web_search` 上限 50,其余领域 20;各领域的设置取自上游,在 `domains.yaml`),**每一轮都把到目前为止的完整历史作为一条消息重发**,所以 prompt 长度随轮数线性增长;工具返回的内容(如两年的日线行情)不做截断。实测(Qwen3.6-35B-A3B,经 OpenRouter):

| task | 轮数 | prompt 总 tokens | 耗时 |
| --- | --- | --- | --- |
| `financial_analysis` 的 `yfinance_task_0001` | 4 | 约 19 万 | 47 秒 |

`web_search` 每轮还有一次工具回执摘要调用,最慢的 task 会用满 50 轮。**费用取决于你选的模型的单价**;先用 `--num-tasks 1` 估一下再放大。并发受模型服务和网关的限流影响,出现大量 429 / 超时时调小 `--max-concurrency`。

## 8. 常见问题

**`-X utf8` 是什么?** 让 Python 一律用 UTF-8 读写文本。中文 Windows 的系统默认是 GBK。也可以设 `export PYTHONUTF8=1`,之后省略 `-X utf8`。

**报 `MCP_GATEWAY_ADDRESS 未设置`**:`.env` 里没填网关,见 2.1。

**报 `连接 MCP server 失败`**:网关不通或 token 不对。`--check` 会逐个服务测连通。

**`--check` 提示 `N 条评估需要在本机联网执行`**:没配远程评估,见 2.2。

**重复出现 `Error searching repositories: ...` 这类打印**:上游评估函数用 `print` 报告查询失败,同时这个 task 会被标 `evaluation_error`,看 `infra_errors`。

**某个 task 的评估一直是 `Execution error`**:看该条评估的 `error`:回答不是合法 JSON / 缺字段是 agent 的问题;连接类错误是基础设施问题。

**为什么没有 hotel / 3D 设计 / MCPMark?** 本次接入的范围是上面五个领域,见第 1 节。

## 9. 代码结构

| 路径 | 作用 |
| --- | --- |
| `IMAct/benchmarks/mcp_universe/run.py` | 命令行入口、领域与 task 选择、`--check`、汇总 |
| `IMAct/benchmarks/mcp_universe/runner.py` | 单个 task 的流程:连服务 → ReAct 作答 → 逐条评估 → 清理 |
| `IMAct/benchmarks/mcp_universe/tasks.py` + `IMAct/benchmarks/mcp_universe/task_data/` + `IMAct/benchmarks/mcp_universe/domains.yaml` | 任务读取;212 个任务 JSON(逐字节取自上游)与各领域的 agent 设置、task 顺序 |
| `IMAct/benchmarks/mcp_universe/prompts.py` + `IMAct/benchmarks/mcp_universe/prompt_templates/` | ReAct 系统 prompt、工具说明、输出格式、回执摘要(模板逐字取自上游) |
| `IMAct/benchmarks/mcp_universe/checks/` | 各领域的评分函数(github / google_maps / google_search / notion / weather / yfinance / playwright),从上游移植,仅改导入与裁判调用 |
| `IMAct/benchmarks/mcp_universe/gateway.py` | 评分函数查询真实服务的入口(上游 `MCPManager.execute` 的形状),并记录基础设施故障 |
| `IMAct/benchmarks/mcp_universe/cleanups.py` | 任务结束后删除 agent 创建的 GitHub 仓库 / Notion 页面 |
| `IMAct/harness/mcp.py` | **通用**:MCP 客户端(经网关的 SSE)、同步句柄、工具集 |
| `IMAct/harness/react.py` | **通用**:ReAct 循环 |
| `IMAct/evaluation/mcp_universe.py` | 评分引擎(取值链、比较函数注册、远程评估) |
| `IMAct/benchmarks/mcp_universe/ORIGIN`、`IMAct/benchmarks/mcp_universe/LICENSE` | 上游来源与 Apache-2.0 许可证 |

`IMAct/benchmarks/mcp_universe/` 是 MCP-Universe 专属的,放弃它时整个删除即可;`harness/mcp.py`、`harness/react.py` 与 `evaluation/mcp_universe.py` 是通用组件,不影响其它 benchmark。

## 10. 测试

```bash
.venv/Scripts/python -X utf8 -m unittest tests.test_mcp tests.test_mcp_universe
```

离线运行,不连网关、不调模型。`test_mcp` 在本机起一个真实的 MCP server 验证客户端;`test_mcp_universe` 用**上游 MCP-Universe 在同样输入上的输出**做基线(`tests/fixtures/mcp_universe_*`),逐项比对:ReAct 的每一轮 prompt / 历史 / 最终回答(24 个用例,含各种出错路径)、729 条评估 × 6 种回答共 3654 个判定、212 个任务渲染后的问题与评估配置、清理参数的解析。
