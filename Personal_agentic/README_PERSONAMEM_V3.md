# PersonaMem-v3 接入(llm_memory 模式 + 外部 Memory 服务)

用 IMAct harness 跑 [PersonaMem-v3](https://github.com/bowen-upenn/PersonaMem-v3):99 个匿名真实用户、每人约两周横跨 5 个 app(Instagram / Facebook / Threads / Chatbot / AI Studio)的行为记录与日历,约 1.58 万道题、30 个题型,测"记得用户并恰当地个性化"。

只做官方五种 mode 中的 **`llm_memory`**:每道题一次模型调用,prompt 里注入一段记忆。与官方的唯一区别是记忆从哪来:

| | 官方 `llm_memory` | 本接入 |
| --- | --- | --- |
| 记忆块 | 本地 LLM 按天把历史压缩成 ≤4096 token 的文字 | 向 **Memory 服务** `query` 一次(统一调用接口),返回的 `context` 同样截到 4096 token |
| 日历 | 截至该时刻的日历状态,附在记忆后 | 同官方(日历是实时状态,不进 memory) |
| 题目、prompt、评分、裁判、汇总 | — | 与官方逐行一致(移植的上游代码,见第 7 节) |

被测对象是 **「Memory 服务 + action 模型」**:同一个 action 模型下,分数差异来自记忆的质量。

```
test.json 的一道题 ──▶ Memory 服务 query(query_id, user_query, ts) ──▶ context(≤4096 token)
                        └─▶ 官方题型 prompt(context + 日历 + 题目)──▶ action 模型 ──▶ 官方评分(规则 / judge 模型)
```

> 命令都在项目根目录(`Personal_agentic/`)的 Git Bash 里运行。`-X utf8` 的作用见「常见问题」。

## 1. 准备

```bash
# 环境(只需一次)。Python >= 3.11
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-server.txt

# 数据:从 HuggingFace 下载到 data/personamem_v3/backend/(全部约 1.1 GB;可随时中断后重跑)
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.personamem_v3.download
# 只下几个 persona(冒烟用)
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.personamem_v3.download --users 25 36
```

每个 persona 下 8 个文件:五个 app 的历史、`calendar.json`、`profile.json`(部分题型的 prompt 与评分要用)、`test.json`(题目)。`persona.html`(人工浏览页)默认不下。数据放在别处:设 `IMACT_PERSONAMEM_V3_DATA` 指向含 `backend/` 的目录。数据许可证 CC-BY-NC-4.0(非商用)。

### 1.1 Memory 服务

按《统一调用接口》调用,与 VitaBench 用的是同一个网关和同一套 MPS 请求方式(`IMAct/exec_layer/memory_mps.py`):

```json
{"bench": "personamem_v3", "user_id": "25",
 "task": {"query_id": "25:0045:geo_shift_25_1_after_shift_sports", "user_query": "any rec for a place to play pickup or hit a class?", "timestamp": 1775525523}}
```

- `user_id` = persona 目录名;`query_id` / `user_query` / `timestamp` 直接取 `test.json` 的 `query_id` / `user_query` / `ts`(Unix 秒)。
- 没有用户正文的题(排序推荐、主动行为等,约 1/3),`test.json` 的 `user_query` 是 `[system prompt] …` 占位文本,按接口约定给 Memory 传**空字符串**(`--memory-query-placeholder verbatim` 改为原样传)。
- 每个用户的历史由 Memory 服务那边预先 `build` 好,IMAct 不调用 `build`。
- 网关地址缺省是统一调用接口里的那个,可用 `--memory-url` 或环境变量 `IMACT_MEMORY_URL` 覆盖。单次请求超时 20 秒(`--memory-timeout`),失败重试 2 次(`--memory-retries`)。
- **网关只在内网可达**,外网机器上 `--check` 会报 `SSL: WRONG_VERSION_NUMBER` 之类的连接错误。

### 1.2 选模型

`models.yaml` 的 `roles`:

| 角色 | 作用 |
| --- | --- |
| `action` | **作答模型**(读记忆 + 题目,产出回答) |
| `judge` | 裁判。大部分题型的分数由它打(个性化程度、克制、隐私泄露等);少数题型是规则评分(排序 NDCG、正则) |

官方的作答与裁判都是 gpt-5.5;换了裁判,绝对分数就不能与论文直接比,只能在同一裁判下横向比较。命令行可临时覆盖:`--agent-model`、`--judge-model`。

## 2. 运行

**先检查**(不调用模型;会向 Memory 网关发 **1 次只读 query** 试连):

```bash
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.personamem_v3.run --models models.yaml --users 25 --check
```

**冒烟**:一个 persona 的前 10 道题

```bash
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.personamem_v3.run --models models.yaml \
  --users 25 --num-queries 10 --out results/pm3-smoke
```

**一个 persona 全部题(约 140-230 道)**

```bash
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.personamem_v3.run --models models.yaml \
  --users 25 --max-concurrency 4 --out results/pm3-25
```

**全部 99 个 persona**(先看第 6 节的费用估算)

```bash
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.personamem_v3.run --models models.yaml \
  --max-concurrency 4 --out results/pm3-full
```

**常用选择**:

```bash
# 只跑某几类题
... --users 25 --task-types personalized_recommendation chatbot_personalized_response
# 前 5 个 persona
... --num-users 5
# 不调用裁判(只看作答与费用;依赖裁判的题型没有分数)
... --no-judge
```

### 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--models` | 必填 | `models.yaml` 路径 |
| `--agent-model` / `--judge-model` | `roles` | 临时覆盖作答 / 裁判模型 |
| `--users ...` | 已下载的全部 | persona 目录名 |
| `--num-users N` | 全部 | 只取前 N 个 persona |
| `--task-types ...` | 全部 | 只跑这些题型(官方 `task_type` 名) |
| `--num-queries N` | 全部 | 每个 persona 只跑前 N 道题(按 `test.json` 顺序) |
| `--max-concurrency` | 4 | **同一个 persona 内**同时跑几道题(persona 之间按顺序) |
| `--max-task-retries` | 2 | 一道题整体失败(Memory / 模型请求重试后仍失败)后的重试次数 |
| `--model-retries` | 3 | 单次模型请求的重试次数 |
| `--no-judge` | 否 | 不调用裁判 |
| `--memory-url` | 接口文档里的网关 | 也可用环境变量 `IMACT_MEMORY_URL` |
| `--memory-timeout` / `--memory-retries` | 20 秒 / 2 | 单次 Memory 请求的超时与重试 |
| `--memory-concurrency` | 16 | 同时在途的 Memory 请求上限 |
| `--memory-token-cap` | 4096 | 注入 prompt 的记忆上限(官方 llm_memory 的预算);超出按行截断 |
| `--memory-query-placeholder` | `empty` | 没有用户正文的题给 Memory 传空串(`empty`)还是原样传占位文本(`verbatim`) |
| `--rerun` | 否 | 忽略已有结果,重新跑 |
| `--check` | 否 | 只检查,不调用模型 |
| `--out` | `results/pm3-<时间戳>` | 结果目录 |

## 3. 断点续跑

每道题跑完立刻写一个结果文件。中断后**用完全相同的命令重新运行**(同一个 `--out`),已完成的题跳过;最后的汇总会把以前跑完的题一起算进去。要全部重跑加 `--rerun`。

跑完会把结果打印成表;事后重新打印(不重跑):`.venv/Scripts/python -X utf8 -m IMAct.report <结果目录>`。断点续跑后的汇总包含以前跑完的部分,不只是这次新跑的。

## 4. 结果怎么看

```
results/pm3-full/
├── summary.json               # 总汇总:官方口径的 micro 分、各类别 / 题型 / 能力轴的分数、每个 persona 的头条
├── 25/
│   ├── 25_0045_geo_shift_…-1a2b3c4d.trial0.json   # 每道题一份(文件名里的 : 换成了 _)
│   └── summary.json
├── trace.jsonl                # 所有模型请求、Memory 请求与返回的完整轨迹
├── run.log
└── models.resolved.*.json     # 本次实际用的模型配置(不含密钥)
```

**头条指标**(与官方 `scripts/aggregate_eval.py` 同一套代码):

| 指标 | 含义 |
| --- | --- |
| `accuracy_pct_micro` | **ALL(micro)**:所有计分题的平均分(每题先折成 0-100) |
| `accuracy_pct_micro_judge` | **ALL-JUDGE**:只含由裁判打分的题型 |
| `by_class` | 按类别:ranking(排序推荐)、chatbot(个性化回复)、over_personalization(过度个性化)、proactive(主动行为)、agentic、other |
| `by_axis` | 按能力轴:显式检索 / 混合 / 隐式推断 |
| `e6_paired` | 主动纠错题的 warn / foil 配对 F1 |

**单道题的结果文件**:`metrics` 是官方 runner 给出的全部指标(含裁判的各维度打分),`accuracy_pct` 是这道题折算后的 0-100 分,`agent_response` 是模型的完整回答,`memory` 是这道题拿到的记忆(`status`、`context` 全文、是否截断)。

- `失败`:重试用尽仍出错(Memory 网关不通、模型接口持续报错),**没有分数**,不算 0 分。
- `status=error`:题型代码在处理这道题时出错(例如模型给出了不合格式、评分代码无法处理的输出)。与官方一致,**按 0 分计**。
- 不计分的题(`accuracy_pct` 为空):官方的主指标缺失时整行不计入,例如裁判没给出该维度。另见第 5 节的 `short_vs_long_term_lifecycle`。
- `memory.status` 为 `empty`:Memory 服务说没有相关记忆,照常作答;为 `error` 的题不会作答,直接按失败处理。

## 5. 评分口径(官方)

每个题型有一个主指标(`task_registry.PRIMARY_METRIC`),折成 0-100:

- **大多数题型由裁判打分**,用一套统一的个性化评分细则:主维度 0-10,外加扣分项。硬规则(泄露用户当天讨厌的东西、泄露隐私、用了已过时的偏好)任何一条违反,整题 0 分。
- **排序题**(feed 推荐、@ai 后续)是分级 NDCG@5;**主动行为题**是 0.7 × 该不该动(规则判定)+ 0.3 × 理由(裁判);**主动纠错**按 warn / foil 是否做对;**换城市推荐**用正则判断有没有说对当前城市。

官方代码里有几处已知问题,**这里照原样保留,不修**,以保证分数和官方可比:

- `short_vs_long_term_lifecycle`(约 4 道/人):主指标需要前后成对的两道题一起算,但官方一次只送一道题,所以这个题型**永远不计分**,不进入任何头条。
- 模型输出的字段类型不符合 prompt 要求时(例如该给字符串却给了列表),个别题型的评分代码会出错,记 `status=error`、0 分。
- 部分题型的 prompt 直接带了用户档案里的信息(主动行为题带大五人格与隐藏人格标签,口吻类 agentic 题带用户的说话风格),这是官方的设计,不是记忆提供的。

## 6. 费用与耗时

每道题平均约 1.1 次作答调用 + 1.1 次裁判调用(重复疲劳题一题连续问约 7 次)。作答 prompt 约 1-2.5k token 加记忆(最多 4096)与日历;裁判 prompt 中位约 2.5k token,个别到 25k。粗估**每道题约 1 万输入 token**,全部约 1.58 万道题约 **1.5 亿输入 token**,再乘以作答 / 裁判模型的单价。Memory 网关每题一次请求。先用 `--users 25 --num-queries 10` 看看单题的实际花费与耗时,再放大。

## 7. 代码结构与对拍

| 路径 | 作用 |
| --- | --- |
| `IMAct/benchmarks/personamem_v3/run.py` | 命令行入口、persona / 题目选择、`--check`、汇总 |
| `IMAct/benchmarks/personamem_v3/runner.py` | 一道题的流程:Memory query → 官方题型 runner(作答 + 评分);官方 LLM 客户端与记忆块来源的适配 |
| `IMAct/benchmarks/personamem_v3/scoring.py` | 汇总(调用官方 `aggregate_eval` 的函数) |
| `IMAct/benchmarks/personamem_v3/download.py`、`data.py` | 数据下载与读取 |
| `IMAct/benchmarks/personamem_v3/pm3/` | **移植的官方评测代码**:题型 runner、prompt、评分、裁判、汇总。与上游的差别只有导入路径、文本编码、去掉 Claude Code 子进程与一处 `sys.path`(见 `ORIGIN`) |
| `IMAct/exec_layer/memory_mps.py` | Memory 网关客户端(与 VitaBench 共用) |
| `IMAct/harness/llm.py`、`IMAct/harness/batch.py` | 通用的模型调用、批处理 / 断点续跑 |

**对拍**:同一批题用同一个确定性的假模型、假裁判、假记忆,分别在上游代码和这里的实现上跑。3 个 persona、490 道题、30 个题型,每道题发给模型的 prompt、裁判 prompt、全部 metrics 与 status 都**逐项相同**。结果存成 `tests/fixtures/personamem_v3_upstream.json.gz`,作为永久的回归测试。

## 8. 测试

```bash
.venv/Scripts/python -X utf8 -m unittest tests.test_personamem_v3
```

离线运行,不连真实网关、不调真实模型。对拍与端到端测试需要 `data/personamem_v3` 里有 persona 25 / 36 / 115,没有时自动跳过。端到端测试会在本机起一个假 MPS 网关和一个假模型服务,验证 Memory 的真实请求格式、结果落盘、汇总与断点续跑。

## 9. 常见问题

**`-X utf8` 是什么?** 让 Python 一律用 UTF-8 读写文本。中文 Windows 的默认编码是 GBK。也可以 `export PYTHONUTF8=1`。

**报 `找不到 PersonaMem-v3 数据`**:先运行下载命令(第 1 节)。

**`--check` 报 Memory 网关连接错误**:网关只在内网可达;换到能访问内网的机器上运行。

**报 `Memory 服务没有返回记忆`**:网关返回了 error(例如"快照尚未构建"),这道题不会作答,按失败处理(可重跑)。

**token 数是怎么算的?** 与官方相同:装了 `tiktoken` 就用它,没装就按字符数 / 4 估算。本项目默认不装,所以记忆上限 4096 token 实际约 1.6 万字符。
