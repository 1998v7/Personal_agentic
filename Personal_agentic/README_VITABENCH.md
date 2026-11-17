# VitaBench 接入(personalization)

用 IMAct harness 跑 [VitaBench 2.0](https://github.com/meituan-longcat/VitaBench-2.0) 的 **personalization** 任务:一个用户带若干个按顺序执行的 subtask(外卖 / 到店 / 酒旅),助手需要结合该用户的历史偏好完成请求。

被测对象是 **「Action 模型 + IMAct 执行层 + Memory」**:

```
VitaBench 任务/环境/工具 ──▶ IMAct harness ──▶ 执行层 imact ──▶ Action 模型
                              用户模拟器、对话循环、评分      Task State、校验、完成条件
                                           └──▶ Memory(本地 或 HTTP 网关)
```

> 命令都在项目根目录(`Personal_agentic/`)的 Git Bash 里运行。`-X utf8` 的作用见文末「常见问题」。

## 1. 准备

```bash
# 环境(只需一次)。Python >= 3.11
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-server.txt

# 密钥:复制 .env.example 为 .env,填你要用的模型的密钥(.env 不会被提交)
cp .env.example .env        # 已有 .env 就不要覆盖
```

**数据集**:把 personalization 的 `tasks.json`(约 125 MB)放到 `data/vitabench/tasks.json`(不随源码提交)。也可以设环境变量 `IMACT_VITABENCH_TASKS` 指向它。共 56 个用户、771 个 subtask(外卖 404 / 到店 223 / 酒旅 144)。

## 2. 选模型

三个角色统一在根目录 `models.yaml` 的 `roles` 里选:

| 角色 | 作用 |
| --- | --- |
| `action` | **被测的 Action 模型**(经过执行层) |
| `user` | 扮演用户的模拟器 |
| `judge` | 裁判,对每个 subtask 的对话按评分点(rubric)逐窗口打分 |

也可以在命令行临时覆盖:`--action-model <别名>`、`--user-llm <别名>`、`--evaluator-llm <别名>`。

## 3. Memory:三选一

VitaBench 的被测系统包含 Memory,由运行配置 `config.vitabench.yaml` 的 `memory.backend` 决定:

| backend | 含义 | 命令行开关 |
| --- | --- | --- |
| `local` | 进程内调用本地 MemoryClient,需要同级目录下的 `../IMAct_memory`(见 `LOCAL_MEMORY.md`) | `--local-memory <harness_config.json>` |
| `mps` | 走 HTTP Memory 网关 | `--memory-url <地址>` |
| `none` | **不用 Memory**。只验证 harness 本身,或做"无记忆"对照 | `--no-memory` |

> 没有 Memory 时,依赖用户历史偏好的评分点会拿不到分(例如要"辛香口味",助手无从知道)。这时 0 分是预期,**不是 harness 的问题**。要看 Memory 的效果,对比有/无 Memory 的得分差异。

## 4. 运行

**先检查**(不发任何模型请求):确认密钥、数据、Memory 配置。

```bash
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.vitabench.run --config config.vitabench.yaml \
  --no-memory --num-tasks 1 --max-subtasks 1 --check
```

**单个 subtask,看完整链路**(`diagnose`,最适合调试;不评分或可选评分):

```bash
# 只跑对话、不评分
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.vitabench.diagnose --config config.vitabench.yaml \
  --subtask-ids sub_A891207_1 --no-memory --out results/vita-diag-1

# 跑完再用裁判评分
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.vitabench.diagnose --config config.vitabench.yaml \
  --subtask-ids sub_A891207_1 --no-memory --with-reward --out results/vita-diag-2
```

**正式运行**(`run`:一个用户的全部 subtask 顺序执行,每个 subtask 评分,用户得分是各 subtask 的平均):

```bash
# 1 个用户的前 3 个 subtask
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.vitabench.run --config config.vitabench.yaml \
  --no-memory --num-tasks 1 --max-subtasks 3 --out results/vita-smoke

# 指定用户,全部 subtask,带本地 Memory
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.vitabench.run --config config.vitabench.yaml \
  --task-ids A891207 --max-concurrency 1 --out results/vita-A891207

# 全部 56 个用户
.venv/Scripts/python -X utf8 -m IMAct.benchmarks.vitabench.run --config config.vitabench.yaml \
  --max-concurrency 3 --out results/vita-full
```

### `IMAct/benchmarks/vitabench/run.py` 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--config` | 必填 | 运行配置 YAML(执行层 / Memory / 重试);模型在 `models.yaml` |
| `--models` | 配置里的 `models_file` | 覆盖模型清单 |
| `--action-model` / `--user-llm` / `--evaluator-llm` | `roles` | 临时覆盖三个角色 |
| `--no-memory` / `--memory-url` / `--local-memory` | 配置里的 backend | 三选一覆盖 Memory |
| `--task-ids ...` | 全部 | 按**用户** ID 选(如 `A891207`) |
| `--num-tasks N` | 全部 | 只取前 N 个用户 |
| `--max-subtasks N` | 全部 | 每个用户只跑前 N 个 subtask(冒烟用;汇总只覆盖这些) |
| `--num-trials` | 1 | 每个用户跑几次 |
| `--max-steps` | 50 | 每个 subtask 的对话步数上限 |
| `--max-task-retries` | 3 | 一个用户整体失败后的重试次数 |
| `--max-concurrency` | 1 | 同时跑几个**用户**(一个用户内的 subtask 始终顺序执行) |
| `--language` | `chinese` | `chinese` / `english` |
| `--seed` | 无 | 随机种子 |
| `--rerun` | 否 | 忽略已有结果,重跑 |
| `--check` | 否 | 只检查,不发请求 |

### `IMAct/benchmarks/vitabench/diagnose.py` 参数

| 参数 | 说明 |
| --- | --- |
| `--subtask-ids ...` | 要跑的 subtask(按原始位置隔离运行,Memory 用它原本的快照) |
| `--with-reward` | 对话结束后调用裁判评分;不加则只跑对话 |
| `--judge-model` | 本次覆盖裁判模型 |
| `--grade-existing <case 目录>` | 不重跑对话,只对已保存的结果重新评分(保留原目录) |
| `--user-completion-guard` | 诊断模式:提示用户模拟器"同意推荐不等于已执行完",避免用户在助手还没下单时就结束 |
| `--resume` | 跳过已完成的样本,不覆盖任何轨迹 |
| `--max-steps` / `--seed` / `--out` | 步数上限 / 种子 / 输出目录(必须为空或不存在) |

## 5. 断点续跑

`run` 每个用户跑完写一个结果文件。中断后**用完全相同的命令重新运行**(同一个 `--out`),已完成的用户会跳过。要全部重跑加 `--rerun`。`diagnose` 用 `--resume`。

跑完会把结果打印成表;事后重新打印(不重跑):`.venv/Scripts/python -X utf8 -m IMAct.report <结果目录>`。断点续跑后的汇总包含以前跑完的部分,不只是这次新跑的。

## 6. 结果怎么看

**`run` 的结果目录**:

```
results/vita-smoke/
├── summary.json                  # 平均 reward、完成/失败/跳过的用户数
├── A891207.trial0.json           # 每个用户一份:全部 subtask 的完整对话、各 subtask 得分
├── exec.sqlite3                  # 执行层的完整轨迹(每次决策尝试、证据、状态)
├── run.log
└── models.resolved.*.json        # 本次实际用的模型配置(不含密钥)
```

用户 json 里 `reward_info.info.subtask_rewards` 是每个 subtask 的得分;`messages` 是全部对话。

**`diagnose` 的结果目录**:除了上面这些,还有可视化页面,用浏览器打开:

- `index.html`:所有样本的总览
- `<subtask>/episode.html`:**完整链路**——用户、Memory 请求/返回、模型输入输出、校验重试、工具回执、Task State 前后变化
- `<subtask>/grading.html`:裁判的逐条 rubric 判定与依据(加了 `--with-reward` 时)

## 7. 评分口径

每个 subtask 有若干**评分点(rubric)**。裁判模型把对话切成重叠的滑动窗口(10 条消息一窗,重叠 2 条),逐窗口更新每个评分点是否满足,状态跨窗口保留。

- **一个 subtask 的得分**:全部评分点满足 = 1,否则 = 0。逐项通过比例单独记在 `rubric_pass_rate`。
- **一个用户的得分**:各 subtask 得分的平均(没有评分标准的 subtask 不计入)。
- 对话没有正常结束(超步数、报错过多)直接 0 分。
- 裁判回复必须完整合法(每个窗口返回全部评分点,布尔判定 + 理由),缺项或格式错误会标记为 `evaluation_error`,**不会悄悄当成通过或普通 0 分**。裁判偶发空回复(推理模型把输出预算花光)会自动重试。

## 8. 费用与耗时

每个 subtask 的对话约 15–30 条消息,执行层每轮决策可能有几次校验重投,另加裁判 2–3 次请求。用 DeepSeek 实测,**1 个用户的 3 个 subtask 约 5–6 分钟**(含评分)。全部 771 个 subtask 按此估算约 20 小时(顺序),`--max-concurrency` 按用户并发。费用取决于所选模型,先用 `--max-subtasks 1` 估算再放大。

## 9. 常见问题

**`-X utf8` 是什么?** 让 Python 一律用 UTF-8 读写文本。中文 Windows 的系统默认是 GBK,会让执行层读 UTF-8 文件时失败。也可以设 `export PYTHONUTF8=1`,之后省略 `-X utf8`。

**报 `找不到 personalization 任务数据`**:`tasks.json` 没放到 `data/vitabench/`。

**报 `environment variable 'XXX_API_KEY' is not set`**:`.env` 里缺某个角色用到的密钥,`--check` 会提前报出。

**报 `首轮 Memory 预取未成功`**:`config.vitabench.yaml` 里 `require_prefetch: true` 时,Memory 不可用就会拒绝运行(防止把"无记忆"误当成联调成功)。要么修好 Memory,要么加 `--no-memory`。

**得分全是 0**:先看是不是没开 Memory(见第 3 节),再看 `grading.html` 里每个评分点的判定依据。

**执行层的 `unverified_resolution` 之类的日志**:是执行层自己的校验在要求模型重新决策,模型一般会自行修正,不是错误。

## 10. 代码结构

| 路径 | 作用 |
| --- | --- |
| `IMAct/benchmarks/vitabench/run.py` / `IMAct/benchmarks/vitabench/diagnose.py` | 命令行入口(正式运行 / 单 subtask 诊断) |
| `IMAct/benchmarks/vitabench/personalization.py` | 把一个 subtask 变成 harness 能跑的东西(助手 prompt、环境、用户模拟器),并按用户汇总 |
| `IMAct/benchmarks/vitabench/domains/`、`IMAct/benchmarks/vitabench/environment/`、`IMAct/benchmarks/vitabench/data_model/` | 外卖 / 到店 / 酒旅的数据库、工具、任务模型(从上游裁剪而来,不依赖上游包) |
| `IMAct/benchmarks/vitabench/prompts/` | 助手与用户模拟器的 prompt |
| `IMAct/benchmarks/vitabench/tasks.py` | 任务读取、Memory 需要的原始 `current_time` |
| `IMAct/benchmarks/vitabench/reward.py` | 诊断模式下的评分与评分页 |
| `IMAct/benchmarks/vitabench/episode_report.py`、`IMAct/benchmarks/vitabench/trace_report.py` | 生成完整链路可视化 |
| `IMAct/harness/` | **通用**交互层:用户模拟器、对话循环、对接执行层的 agent、批处理 |
| `IMAct/evaluation/vitabench.py` | 滑动窗口 rubric 评分 |
| `IMAct/exec_layer/` | 执行层接入与 Memory 客户端 |
| `IMAct/benchmarks/vitabench/ORIGIN`、`IMAct/benchmarks/vitabench/LICENSE` | 上游来源与 MIT 许可证 |

`IMAct/benchmarks/vitabench/` 是 VitaBench 专属的,放弃 VitaBench 时整个删除即可,不影响 `harness/` 与 `evaluation/base.py`。

## 11. 测试

```bash
.venv/Scripts/python -X utf8 -m unittest tests.test_vitabench_bridge tests.test_vitabench_run tests.test_personalization tests.test_reward tests.test_trace
```

离线运行,不调用任何真实模型,也不需要数据集。
