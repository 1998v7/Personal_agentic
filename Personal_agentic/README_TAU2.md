# τ²-bench 接入(airline / retail / telecom,6 语种)

用 IMAct harness 跑 [τ²-bench](https://github.com/sierra-research/tau2-bench) 2.0 的 **text-v1**:6 种语言 × 3 个 domain = **18 个 cell**,每个 cell 用同一批冻结的 **50 个 task**,共 **900 个**。

被测对象是 **agent 模型本身**:原生 function-calling,system prompt 与上游 `LLMAgent` 一致。**不涉及 Memory,也不经过执行层**,所以分数可以和 τ² 公布的榜单对照。

> 命令都在项目根目录(`Personal_agentic/`)的 Git Bash 里运行。`-X utf8` 的作用见文末「常见问题」。

## 1. 准备

```bash
# 环境(只需一次)。Python >= 3.11
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-server.txt

# 密钥:复制 .env.example 为 .env,填你要用的模型的密钥(.env 不会被提交)
cp .env.example .env        # 已有 .env 就不要覆盖
```

**数据集**放在 `data/tau2/`(不随源码提交,约 70 MB)。从上游 `tau2-bench` 仓库的 `data/tau2/` 复制这几项:

```
data/tau2/
├── benchmarks/text.json                      # text-v1 的 manifest(18 个 cell)
├── task_subsets/{airline,retail,telecom}_50.json   # 冻结的 50 个 task
├── domains/{airline,retail,telecom}/         # 原始数据库、policy、任务
└── multilingual/{en,es,hi,ko,pt,zh}/         # 语言包(pack.yaml)、元数据;非英文还有 regional/ 本地化数据库
```

数据放在别处:设环境变量 `IMACT_TAU2_DATA` 指向包含 `tau2/` 的那个目录。查找顺序:环境变量 → 本仓库 `data/`。

## 2. 选模型

三个角色统一在根目录 `models.yaml` 的 `roles` 里选:

| 角色 | 作用 |
| --- | --- |
| `action` | **被测的 agent 模型** |
| `user` | 扮演顾客的用户模拟器 |
| `judge` | 裁判,**只在需要语义判断时才调用**(见「评分」) |

```yaml
roles:
  action: openrouter-deepseek-v4.1-flash
  user:   openrouter-deepseek-v4.1-flash
  judge:  openrouter-deepseek-v4.1-flash
```

也可以不改文件,用命令行临时覆盖:`--agent-model <别名>`、`--user-model <别名>`、`--judge-model <别名>`。别名见 `models.yaml` 的 `models:`,新模型在那里加一段即可。

## 3. 运行

**先检查**(不发任何模型请求,不花钱):确认密钥、数据、要跑的 cell 与 task 数。

```bash
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.tau2.run --models models.yaml --domain all --language all --check
```

**冒烟**:每个 cell 只跑 1 个 task(3 个 domain × 6 语种 = 18 个 task,几分钟)。

```bash
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.tau2.run --models models.yaml \
  --domain all --language all --num-tasks 1 --max-concurrency 3 --out results/tau2-smoke
```

**完整 900 个**(会产生较多模型调用,建议先看下面的「费用与耗时」):

```bash
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.tau2.run --models models.yaml \
  --domain all --language all --max-concurrency 3 --out results/tau2-full
```

**常用选择**:

```bash
# 只跑英文三个 domain(150 个)
... --domain all --language en --out results/tau2-en
# 只跑中文 + 韩文的 airline
... --domain airline --language zh ko --out results/tau2-airline-zh-ko
# 指定 task(canonical ID,各语言通用,只能配合单个 --domain)
... --domain retail --language zh --task-ids 6 8 11
# 换被测模型
... --domain all --language en --agent-model openrouter-gpt-4.1
# 同一批 task 多跑几次(报告平均与方差时用)
... --num-trials 3 --seed 7
```

### 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--models` | 必填 | `models.yaml` 路径 |
| `--domain` | 必填 | `airline` / `retail` / `telecom` / `all`,可多选 |
| `--language` | `en` | `en es pt hi zh ko` / `all`,可多选 |
| `--benchmark` | `text-v1` | `text-v1` 是官方 2.0 评测集;`base` 是 pre-2.0 的原始英文任务(airline 50 / retail 114 / telecom 114),**只支持 `en`,分数不能与 text-v1 榜单对照** |
| `--num-tasks N` | 全部 | 每个 cell 只取前 N 个 task |
| `--task-ids ...` | 全部 | 按 canonical ID 选 |
| `--num-trials` | 1 | 每个 task 跑几次 |
| `--max-concurrency` | 1 | **同一个 cell 内**同时跑几个 task(cell 之间按顺序) |
| `--max-steps` / `--max-errors` | 200 / 10 | 一段对话的步数上限 / 工具报错累计上限 |
| `--max-task-retries` | 3 | 任务整体失败后的重试次数 |
| `--model-retries` | 3 | 单次模型请求失败(限流、超时、空回复)的重试次数 |
| `--communicate-scorer` | `semantic` | 「必须告知用户的信息」怎么判:`semantic` 用裁判模型(上游默认);`substring` 用子串匹配,不调裁判 |
| `--no-judge` | 否 | 只跑对话,不评分 |
| `--rerun` | 否 | 忽略已有结果,全部重跑 |
| `--out` | `results/tau2-<时间戳>` | 结果目录 |

## 4. 断点续跑

每个 task 跑完立刻写一个结果文件。中途中断(Ctrl+C、断网、限流)后,**用完全相同的命令重新运行**(同一个 `--out`),已有结果的 task 会跳过,只补没跑完的。要全部重跑加 `--rerun`。

跑完会把结果打印成表;事后重新打印(不重跑):`.venv/Scripts/python -X utf8 -m IMAct.report <结果目录>`。断点续跑后的汇总包含以前跑完的部分,不只是这次新跑的。

## 5. 结果怎么看

```
results/tau2-full/
├── summary.json                  # 总汇总:总平均分 + 每个 cell 的完成/失败/评分错误数与平均分
├── zh-airline/                   # 每个 cell 一个目录
│   ├── 0_zh_identity.trial0.json # 每个 task 一份:完整对话、每个评分分量的明细
│   └── summary.json
├── trace.jsonl                   # 所有 agent / 裁判请求的完整轨迹
├── run.log
└── models.resolved.*.json        # 本次实际用的三个模型配置(不含密钥)
```

终端最后会打印每个 cell 的表格和总平均分。看单个 task 为什么失败,打开它的 json:`reward_info.db_check` / `env_assertions` / `communicate_checks` / `nl_assertions` 里是各分量的逐项判定,`messages` 是完整对话。

- `失败`:task 在重试用尽后仍然出错(例如接口持续不可用),它**没有分数**,不会被当成 0 分算进平均。
- `评分错误`:对话跑完了,但裁判回复不可用,该 task 记 0 分并标 `evaluation_error`。这个数不为 0 时,平均分偏低,应先查原因再看分数。

## 6. 评分口径

得分是各分量的**乘积**,每个 task 用哪些分量由它自己的 `reward_basis` 决定:

| 分量 | 含义 | 是否调裁判 |
| --- | --- | --- |
| `DB` | 重放对话中的有状态工具调用,数据库终态与参考终态一致 | 否 |
| `ENV_ASSERTION` | 在重放后的环境上跑任务给定的断言(telecom 全部是这类) | 否 |
| `ACTION` | 参考动作都出现在 agent 或用户的工具调用里 | 否 |
| `COMMUNICATE` | 必须告知用户的信息(airline 的部分 task) | 默认是(`semantic`) |
| `NL_ASSERTION` | 自然语言断言(retail 的部分 task) | 是 |

对话没有正常结束(超步数、报错过多、agent 输出无效)直接 0 分。裁判回复必须完整合法(逐项有布尔判定与理由),缺项或格式错误会抛错并标记为评分错误,**不会悄悄当成通过**。

## 7. 多语种说明

- 英文 cell 用原始数据库,并用上游审阅过的英文 retail / telecom 任务替换同 ID 的原任务;其他 5 种语言用各自发布的**本地化数据库包**(本地化的姓名、地址、币种、时区、机场、policy 与任务)。
- 同一个 task 在各语言里有两个 ID:本语言 ID(如 `6_zh_identity`)和跨语言一致的 **canonical ID**(`6`)。命令行与统计都以 canonical ID 为准。
- 用户模拟器每个 task 按性别匹配一个固定人设(每种语言每个性别只有一个,所以没有随机性,也没有 persona 种子参数)。非英文的用户会被要求全程使用该语言;agent 的 prompt 也带对应的语言说明。
- 非英文 cell 评分时,「告知信息」的语义裁判会被告知顾客说的是哪种语言(译文、转写、混用都算数)。

## 8. 费用与耗时

每个 task 大约是 agent 10–30 次请求 + 用户模拟器 10–20 次请求(+ 偶尔 1–3 次裁判)。用 DeepSeek 实测,单个 task 约 **25–120 秒**(airline ≈ 45 秒,retail ≈ 60 秒,telecom ≈ 55 秒)。

粗估完整 900 个:`900 × 约 55 秒 ÷ 并发数`。`--max-concurrency 3` 约 **4 小时**,5 约 2.5 小时。**费用取决于你选的三个模型的单价**,换成大模型会高出一到两个数量级——先用 `--num-tasks 1` 或 `--domain airline --language en` 估一下再放大。并发受 OpenRouter 等服务的限流影响,出现大量 429 时调小。

## 9. 常见问题

**`-X utf8` 是什么?** 让 Python 一律用 UTF-8 读写文本。中文 Windows 的系统默认是 GBK,会让读 UTF-8 文件时报错或乱码。也可以用环境变量代替:`export PYTHONUTF8=1`,之后可以省略 `-X utf8`。

**报 `找不到 τ² 的任务数据`**:数据没放对,见第 1 节。

**报 `environment variable 'XXX_API_KEY' is not set`**:`.env` 里缺这个角色用到的密钥。`--check` 会提前报出,不会跑到一半才发现。

**某个 task 反复失败**:看 `<out>/run.log`;接口限流时调小 `--max-concurrency`。

**为什么没有 hotel domain?** 上游 τ²-bench 只有 airline / retail / telecom(以及这里不涉及的 banking_knowledge、intake),没有 hotel。

## 10. 代码结构

| 路径 | 作用 |
| --- | --- |
| `IMAct/benchmarks/tau2/run.py` | 命令行入口、cell 选择、汇总 |
| `IMAct/benchmarks/tau2/multilingual.py` | 语言包、cell(语言 × domain)加载、人设分配、冻结子集校验 |
| `IMAct/benchmarks/tau2/runner.py` | 单个 task 的流程:建环境 → 用户 + agent 对话 → 评分 |
| `IMAct/benchmarks/tau2/prompts.py` + `IMAct/benchmarks/tau2/prompt_templates/` | agent 与用户模拟器的 prompt(模板文本从上游原样导出) |
| `IMAct/benchmarks/tau2/domains/`、`IMAct/benchmarks/tau2/environment/`、`IMAct/benchmarks/tau2/data_model/` | 三个 domain 的数据库、工具、任务模型(从上游裁剪而来,不依赖上游包) |
| `IMAct/benchmarks/tau2/harness_env.py` | 把 τ² 环境接到通用 harness 的环境协议 |
| `IMAct/harness/` | **通用**交互层:用户模拟器、对话循环、原生 agent、批处理(与具体 benchmark 无关) |
| `IMAct/evaluation/tau2.py` | τ² 的评分 |
| `IMAct/benchmarks/tau2/ORIGIN`、`IMAct/benchmarks/tau2/LICENSE` | 上游来源与 MIT 许可证 |

`IMAct/benchmarks/tau2/` 是 τ² 专属的,放弃 τ² 时整个删除即可,不影响 `harness/` 与 `evaluation/base.py`。

## 11. 测试

```bash
.venv/Scripts/python -X utf8 -m unittest tests.test_tau2 tests.test_tau2_multilingual
```

离线运行,不调用任何真实模型(缺数据集时依赖数据的用例会被跳过)。其中包括:在 18 个 cell 里各用参考动作重放,验证本地化环境、工具与评分标准一致。
