# 真实模型运行记录（2026-11-10）

> 本文保留 2026-11-10 的本机运行记录；其中仓库路径、提交号和服务状态为当时快照。`results/` 中的原始运行产物保留在原工作区，不随本次源码提交。

已使用 MatrixLLM 的 `gpt-5.6-sol` 运行项目。启用 Memory 的一轮完成了真实模型决策、真实 Memory 网关查询、VitaBench 工具执行和对话收尾；模拟订单最终为 `paid`。**裁判请求失败，尚无有效的 LLM 评分。**

## 运行范围与结果

两轮均运行用户 `A891207` 的第一个子任务 `sub_A891207_1`，各 1 次 trial、并发 1、最多 20 步，整任务重试 0。Action、用户模拟器和裁判均配置为 `gpt-5.6-sol`，通过 `/v1/responses` 调用。开场问候和初始用户请求由 benchmark 提供，后续用户回复来自真实模型。

| 项目 | 第一轮：Memory 关闭 | 第二轮：Memory 开启 |
| --- | --- | --- |
| 结果目录 | `results/matrixllm-smoke-gpt-20261110-140857/` | `results/matrixllm-smoke-gpt-20261110-141702/` |
| 运行耗时 | 415.6 秒 | 440.8 秒 |
| Action 模型输出 | 14 条，均有 request_id | 15 条，均有 request_id |
| 被校验拒绝的输出 | 2 条 | 7 条 |
| 工具调用 | 发出 6 次；轨迹收到 5 条回执 | 5 次，均收到回执 |
| 对话终止 | `max_steps` | `user_stop` |
| 订单状态 | `unpaid` | `paid` |
| 原始 summary reward | 0.0 | 0.0 |
| 评分解释 | 步数上限规则置零，没有调用裁判模型 | 裁判 API 返回 `invalid_prompt`，异常被框架转为零分 |
| 进程退出码 | 0 | 0 |

这是同一子任务的两次冒烟运行，不是全量评测，也不能据此推导 Memory 的平均收益。第二轮的 `failed=[]` 仅表示没有异常逃出任务入口，**不代表评分成功**。

## 第二轮的真实输出与执行证据

用户请求：

> 忙得脚不沾地，中午饭就帮我点个外卖来单位算了

真实模型最终回复：

> 已完成支付！辛香汇的椒香鸡排饭套餐共32元，预计今天12:01送到单位。还有其他需要吗？

实际工具调用顺序：

1. `delivery_store_search_recommand`
2. `delivery_product_search_recommand`
3. `create_delivery_order`
4. `pay_delivery_order`
5. `get_delivery_order_status`

订单 `OT489e2f7e62`：椒香鸡排饭套餐 1 份，32 元，预计配送 26 分钟。支付回执为 `Payment successful`，随后状态回查为 `paid`；支付工具仅调用一次。SQLite 最终 `progress.phase=payment_confirmed`、`lifecycle=stopped`、`stop_reason=user_stop`，没有待处理工具调用。

业务工具在 VitaBench 本地模拟环境中执行。模型和 Memory 网关为真实远程调用；订单支付是模拟状态变化，未产生真实商户订单或扣款，也没有实际送达证明。

本次 Memory 只调用已有快照的 `query`，没有调用 `build` 或写入接口。实际运行的预取返回了 352 字符的口味、30 分钟配送上限等记忆；它与独立连通探测的返回文本并不完全相同。实际下单使用了资料里的脱敏地址 `辽宁省锦州市凌河区******`，因此不能从支付成功推导准确单位地址已得到验证。

## 暴露的问题

**任务收尾出现 7 次校验拒绝。** 模型先把“尚未支付”记录成 `other_requirement`。支付成功后，模型在工具回执事件里尝试关闭这条笔记，被 `unverified_resolution` 拒绝。当前规则只允许在 `user_message` 事件里关闭 `other_requirement` / `unclear_authorization`，见 `execution_harness/src/imact/harness.py:433`。回查订单状态后仍触发相同限制，执行层一度返回 blocked 提示；用户模拟器确认继续后，模型才关闭笔记并正常结束。没有通过放宽校验或改写模型输出掩盖这一过程。

**裁判失败被汇总成普通零分。** 第二轮开始计划的 3 个评分窗口，在第 1 个窗口收到 HTTP 200 后，SSE 内部返回 `response.failed`，错误码 `invalid_prompt`，消息提示该 prompt 被模型服务策略拒绝。后续窗口未完成，有效 LLM 评分窗口为 0/3。`PersonalizationOrchestrator._evaluate_subtask` 捕获异常并返回 reward=0.0；最终聚合只保留子任务分数，导致 `summary.json` 看起来像正常完成的零分。具体证据见第二轮 `console.log:670-675`。本次保留原始 summary，并在 `verification.json` 显式标记 `evaluation_status=failed_invalid_prompt`、`valid_llm_score=false`。

第一轮则先反复确认地址，随后创建了未支付订单；用户指出地址不对。末尾虽然发出了取消工具调用，但达到步数上限，保存的轨迹没有取消回执，最终快照仍为 `unpaid`，不能声称订单已取消。

两轮都是有效的真实模型执行记录；第二轮仍没有有效的裁判成绩。当前价格配置也不足以给出费用：日志里的 `cost=0.0` 不能解释成免费。Action API 回报的 token 用量分别为输入 272,340 / 输出 4,555，以及输入 619,047 / 输出 3,492，包含校验重试，不包含用户模拟器和裁判调用。

## 可直接查看的结果

优先查看第二轮：

- [完整对话与工具调用](results/matrixllm-smoke-gpt-20261110-141702/conversation.md)
- [Action 模型原始输出](results/matrixllm-smoke-gpt-20261110-141702/model_outputs.md)
- [原始输出 JSONL，含 request_id、usage 和校验错误](results/matrixllm-smoke-gpt-20261110-141702/model_outputs.jsonl)
- [核验结果与评分异常标记](results/matrixllm-smoke-gpt-20261110-141702/verification.json)
- [原始 benchmark 轨迹](results/matrixllm-smoke-gpt-20261110-141702/A891207.trial0.json)
- [原始汇总](results/matrixllm-smoke-gpt-20261110-141702/summary.json)
- [完整控制台日志](results/matrixllm-smoke-gpt-20261110-141702/console.log)
- [执行层数据库](results/matrixllm-smoke-gpt-20261110-141702/exec.sqlite3)

第一轮同类文件位于 `results/matrixllm-smoke-gpt-20261110-140857/`。两轮的 SQLite `integrity_check` 均为 `ok`，所有记录的事件均为 `committed`。数据库表为 `tasks`、`events`、`attempts`、`notes`，例如查看真实模型响应信息：

```sql
SELECT id, event_id,
       json_extract(metadata, '$.model') AS model,
       json_extract(metadata, '$.request_id') AS request_id,
       json_extract(metadata, '$.finish_reason') AS finish_reason,
       decision, error
FROM attempts ORDER BY id;
```

## 复跑

在仓库目录、允许访问 MatrixLLM 和 Memory 网关的终端执行：

```bash
PYTHON=.venv/bin/python PYTHONUNBUFFERED=1 bash scripts/smoke_matrixllm.sh gpt --memory
```

要复现第一轮的 Memory 关闭配置,请显式使用 `--no-memory`。当前脚本默认遵循运行配置,该配置已启用 Memory 预取。入口读取根目录 `.env` 并使用新的结果目录。该命令复现的是运行入口；真实模型输出和评分服务状态不保证与本次相同，裁判错误仍需单独处理。

当前检查到的主项目提交为 `c120d3566fea4bc83f5dc78979f41ad908c5f1b4`，VitaBench 提交为 `f60169e89f30499cb7883f3dad76bd03facc908d`。本次没有修改项目执行逻辑、执行层或 VitaBench 源码，也没有修改数据集。实际运行配置已保存为各结果目录下的 `action.config.yaml`、`vita.models.yaml` 和 `run.manifest.json`，密钥仍仅引用环境变量。

本次验证依据为真实网络请求、模型原文、工具回执、数据库状态和完整进程退出结果。此前的离线测试记录保留在 [LOCAL_RUN.md](LOCAL_RUN.md)，本轮未重复运行离线测试。

## 后续：切换 Claude 裁判及流式验证

按用户要求，`scripts/smoke_matrixllm.sh` 的默认裁判已改为 `claude-opus-4-6`。用户模拟器和裁判现在分别选择接口及 `X-LY-TOKEN`：GPT 走 Responses，Claude 走 Anthropic Messages。模型探测复用正式适配器，检查正文及完成状态。相关 24 项测试、脚本语法检查、GPT 用户与 Claude 裁判的独立路由检查、`--check-only` 均通过。

2026-11-10 14:47 的再次实测中，两种请求都显式设置 `stream: true`、`max_tokens: 1024`：

| 接口 | 结果 |
| --- | --- |
| `/anthropic/v1/messages` | HTTP 200、`Content-Type: text/event-stream`，但响应正文为 0 字节 |
| `/v1/chat/completions` | HTTP 403，`invalid_model`，`Model resource is not available` |
| `/v1/models` | 列表包含 `claude-opus-4-6`；这不能单独证明该资源可调用 |

正式 Claude 适配器原本已设置 `stream: true`。此前备用 Chat Completions 探测曾使用 `false`，该次探测不足以单独判断资源不可用；本节记录的是补齐流式参数后的实测结果。当前仍需对照资源页面提供的请求地址、鉴权方式及请求示例，不能仅据这些返回断定 `.env` 填写错误。

最新诊断目录：[judge-claude-opus-4-6-20261110-144713](results/judge-claude-opus-4-6-20261110-144713/manifest.json)，其中保存原始空流、模型列表和 403 响应。诊断只发小请求，尚未进入裁判评分；没有重新运行 Action、Memory 查询或业务工具，没有覆盖原始轨迹，也没有生成新的评分。原轨迹 SHA-256 仍为 `d38674933a4dce0a028b54f2d32e4794ee4fcb60785241db21c830204de85ff4`。

### 对照用户提供的 curl 示例

用户随后提供的成功示例使用 `claude-opus-5`，与此前指定的裁判 `claude-opus-4-6` 不同。2026-11-10 14:53，使用该示例的相同 URL、headers、中文问题、`max_tokens: 512`、`stream: true`，从项目 `.env` 读取实际凭据，经系统 `/usr/bin/curl` 分别测试两个模型；两者均返回 HTTP 200、正文 0 字节。凭据通过 curl 标准输入传递，没有写入命令参数或结果文件。

检测到当前进程带有本机 `127.0.0.1:13659` 代理环境变量后，14:56 又仅对 MatrixLLM 做了一次直连对照；两个模型仍返回 0 字节，服务器响应头明确为 `content-length: 0`。本次结果没有支持“本机代理导致正文丢失”或“流解析器吞掉内容”的解释。

- [使用环境代理的两模型对照结果](results/claude-model-comparison-20261110-145357/results.json)
- [直连的两模型对照结果](results/claude-direct-comparison-20261110-145615/results.json)

截至本次记录，`claude-opus-5` 与 `claude-opus-4-6` 都未在当前项目凭据下返回有效正文，不能把用户所述的另一条成功调用当成本机验证成功。默认裁判仍为此前指定的 `claude-opus-4-6`。需要进一步核对成功 curl 使用的真实凭据与 `.env` 中 `MATRIXLLM_API_KEY`、`X_LY_TOKEN_CLAUDE` 是否一致，或在用户自己的终端运行同一对照脚本排查环境差异：

```bash
.venv/bin/python results/setup/probe_claude_models.py
# 仅对 MatrixLLM 直连：
.venv/bin/python results/setup/probe_claude_models.py --direct
```


## 2026-11-10: 统一模型配置与 Qwen3.8-27B 接入

用户确认 Claude 接口由服务提供方排查,本轮不再探测两个 Claude 模型。新增根目录 `models.yaml`: `roles.action=qwen3.8-27b`,用户模拟器保持 `gpt-5.6-sol`,裁判保留 `claude-opus-4-6` 配置。只有被选中的角色要求凭据;`.env` 中的 `ANTCHAT_API_KEY` 供 Qwen 使用,不要求两个 SOFA header。

Action、用户模拟器、裁判统一通过 `IMAct/models/` provider 发请求。旧 shell 中的 URL、参数和临时配置生成已移除;`config.vitabench.yaml` 仅引用 `models_file`,管理执行层、重试和 Memory。运行时展示并保存三个角色的脱敏配置快照。

真实验证分两层:

| 验证 | 证据与结果 |
| --- | --- |
| AntChat 文本接口 | [响应](results/qwen-model-config-probe.json): `Qwen3.8-27B`, `finish_reason=stop`, 63 input / 201 output tokens, request ID `219f89e617942958829041277e22ec` |
| 执行层首轮 Action | [结果](results/qwen-action-probe-20261110-153912/result.json)、[真实输出](results/qwen-action-probe-20261110-153912/model_outputs.json)、同目录 `exec.sqlite3`: 一次模型调用,6806 input / 1530 output tokens,request ID `219fdace17942963528678234ec1ed`;生成 `delivery_store_search_recommand(keywords=["附近","川菜"])`,通过执行层校验 |

首轮 Action 的初次接入发现 HTTP 400: `System message must be at the beginning.`。原因是执行层发送协议和任务策略两条连续 system 消息。Qwen 配置现在显式启用 `options.merge_system_messages: true`,合并开头系统文本,保留全部内容。失败轨迹保存在 `results/qwen-action-probe-20261110-153801/`,成功轨迹另存,未覆盖。

首轮 Action 使用执行层自带的工具 schema 和构造的午餐请求,只验证模型输出和执行层解析,没有执行业务工具、用户模拟器或裁判,不代表一个 VitaBench 任务完成,没有 Qwen benchmark 分数。此前沙箱网络被阻止的记录 `results/qwen-action-probe-20261110-153620/` 没有真实模型输出,也不计为模型失败。

离线回归:项目 99 项测试通过(含统一配置、URL/header/request body、角色别名到实际模型名、三种协议适配、凭据脱敏、连接关闭等检查)。脚本语法和新入口 `--check-only` 通过;后者只确认本地配置与数据,不确认 Claude 服务恢复。

## 2026-11-10 16:54：接入 Memory 预取后只运行一次 Action

此前 16:37 的单轮记录使用了 Memory 未启用的配置,首轮输入为 `user_memory.status=not_requested`。该记录仅验证 Action 调用、校验和状态提交,未验证方案要求的首轮 Memory 预取。

现已将 `memory接口.md` 的网关写入 `config.vitabench.yaml`,保留 `prefetch: true`。`scripts/smoke_matrixllm.sh` 默认遵循运行配置,移除隐式 `--no-memory`;显式传入 `--no-memory` 仍可关闭。单轮诊断脚本先执行 `begin_subtask()` 预取,Memory 失败则停止,不调用 Action;Action 自动重试为 0,只调用一次 `advance()`。`--check` 不访问任何模型或 Memory 服务。

本次仍选 `A891207/sub_A891207_1`,原始查询时间为 `2026-03-02`。Memory 请求只使用原始用户、任务 ID、用户问题和原始时间,查询已有快照,没有调用 build 或写入接口。

| 核验项 | 结果 |
| --- | --- |
| Memory 服务 | 共 4 次 HTTP 请求;3 次网关侧 8 秒超时,第 4 次返回有效记忆 |
| 首次预取运行 | `results/memory-action-one-round-20261110-165300/`;两次请求均超时,Action 调用 0 次 |
| 成功运行 | `results/memory-action-one-round-20261110-165440/`;首次请求超时,自动重试成功 |
| Memory 返回 | `status=ok`,498 字符;包括辛香川菜偏好、通常不超过 30 分钟的配送要求等 |
| 首轮注入 | 原始响应 context、`start.memory`、首次 Action HTTP 请求的 `user_memory`、SQLite 上下文及状态中的记忆全文一致;Memory 返回早于 Action 请求 |
| Action | `Qwen3.8-27B`,1 次请求、1 次输出,`finish_reason=stop`;8431 input / 1763 output tokens |
| Action 输出 | `delivery_product_search_recommand(keywords=["川菜","辛香","午餐"])` 与 `address_to_longitude_latitude` 工作地址解析 |
| Harness | 1 条模型尝试、1 个 committed 事件,无校验错误;`rev=1`,`turns=1`,SQLite 完整性 `ok` |
| 执行边界 | 输出动作后停止;业务工具、用户模型、裁判均未调用,无评分 |

Action 将配送上限和口味偏好写入了 `task_changes`。这证明本轮 Memory 材料已进入决策且被采用,不证明后续选品、支付或完整任务完成。

查看 [完整首轮输入输出](results/memory-action-one-round-20261110-165440/action_agent_round.md)、[结构化记录](results/memory-action-one-round-20261110-165440/action_agent_round.json)、[核验结果](results/memory-action-one-round-20261110-165440/verification.json)。同目录保留 `trace.jsonl`、Memory 请求与原始返回、Action HTTP 请求与原始返回、`exec.sqlite3`。Action request ID 为 `2182adb817943008997578698e38eb`。

本次相关 3 项离线回归通过,覆盖首轮 Memory 注入、外层桥接和完整 trace。默认 Memory 开启及显式关闭的命令行配置检查通过;真实调用结果以本节的网络、轨迹和数据库证据为准。

复跑首轮（使用新的输出目录）：

```bash
.venv/bin/python results/setup/probe_one_round.py --out results/memory-action-one-round-next
```

## 2026-11-10 17:18–17:23：续跑第二、第三次 Action，检查输入重复

从上述 Memory 成功首轮的 SQLite checkpoint 副本恢复,原 checkpoint 保持不变。保存状态为 `rev=1`,`turns=1`,两个工具调用尚未执行;复用已有 Memory,没有重做首轮预取或首轮 Action。先用 mock Action 检查硬停止边界,随后只新增 2 次真实 Action 请求,停止在第 3 次调用之后。

结果与完整输入输出见 [第二、第三轮重复核验](results/action-round2-3-20261110-171730/duplication_report.md)。统计依据为实际 HTTP 请求,同一请求内完整材料的出现次数,不是 trace 展示次数：

| Action 调用序号 | Harness 通用协议 | 任务策略/账户资料 | 工具定义 | 首次 Memory 全文 | API prompt tokens |
| --- | --- | --- | --- | --- | --- |
| 1（已有记录） | 1 | 2 | 2 × 19 个 | 2 | 8,431 |
| 2（本次新增） | 1 | 2 | 2 × 19 个 | 2 | 40,946 |
| 3（本次新增） | 1 | 2 | 2 × 19 个 | 2 | 28,366 |

根因：首轮完整 `start` 事件被存入 `recent_events[0].event`,其中 `system_prompt`、`tools`、`memory` 再次与 system 消息、`available_tools`、`user_memory` 一起发送。第二、三次调用依然各有两份。工具定义比较忽略了冗余的 null 字段,名称、描述和参数 schema 一致。

第二次输入还增加了商品搜索返回的 100 个候选及结构化事实,因此输入增长并非全部来自上述重复。仅静态删除历史 start 中三类重复字段,每次可减少 15,162 字节,对应第二、三次消息文本的 13.23% 和 17.99%;未修改真实输入或重跑模型验证 token 收益。

本次另暴露两个实际执行问题：第二次 Action 用满 4096 输出 token,`finish_reason=length`,content 为空,被 Harness 标记 `model_incomplete`。Harness 返回 blocked 后,用户模拟器真实回复“麻烦帮我点份中午送到单位的外卖，其他你看着办。”,触发第三次调用。第三次 `finish_reason=stop`,返回了工具动作,但被 `ungrounded_id` 校验拦截,错误为“store_id 不在本轮候选事实中”。这两个错误未被归因到上下文重复,需要独立分析。

轮次按模型调用次数计数,不是成功提交的任务推进次数。最终有 3 条 attempts、3 个 committed 事件,`rev=3`,`turns=1`;第二、三次事件提交的是 blocked 回退结果。第二次完成后诊断脚本最初误将 accepted turns 与模型调用序号等同而停止,随后从已保存的 rev=2 恢复,没有重复调用第二次模型。保存了恢复前的 `continuation.before_resume.json`。

实际执行的业务工具只有首轮的商品搜索和地址解析,均在 VitaBench 本地模拟环境返回回执。新增 Memory 请求 0、用户模型调用 1、裁判调用 0,没有第 4 次 Action。原首轮数据库 SHA-256 保持不变;实际请求与 SQLite attempts.context 一致,响应正文与 attempts.decision 一致,数据库完整性为 ok。原始文件与校验结果保存在同一结果目录。

## 2026-11-10：修复 Action 输入重复，三轮离线回放

已修改仓库内 `execution_harness/src/imact/context.py` 与 `prompts/action.txt`。上下文中的 start 和 memory_result 事件使用去重视图,任务策略、工具定义和 Memory 各由一个正式字段提供。候选回执只有在所有内容、来源、顺序及字段值都能由当前候选完整还原时才改为引用;错误、交易、未完整解析和非结构化回执保留原文。数据库中的原始事件和证据不变。

用前述三轮的原事件、原回执和原模型输出离线回放,没有调用任何远端模型、Memory 或业务工具。三轮消息文本 UTF-8 字节分别从 38,544 / 114,593 / 84,265 降至 23,823 / 70,002 / 69,544,减少 38.19% / 38.91% / 17.47%。这是字节统计,修复后的 API token 数尚未实测。

三轮内任务策略、工具定义和 Memory 全文都各出现一次;第二、三轮的 100 个候选及其全部原有字段、Memory 全文、任务状态和工作笔记均保持一致。离线回放固定使用原输出,所以第二轮 `model_incomplete` 和第三轮 `ungrounded_id` 仍如实保留,未据此宣称模型决策改善或任务成功。原始运行文件哈希保持不变。

执行层 94 项测试、外层 102 项测试通过,新增 17 项覆盖多轮去重、Memory、完整与不完整回执、来源、顺序、字段变化、状态不变性及预算边界。Ruff 和 diff 检查通过。详情、三轮完整新输入、原始请求对比及日志见 [输入去重修复报告](results/context-compaction-20261110/comparison.md)。

## 2026-11-10：完整链路可视化与三个领域复验

运行与回放统一保存在 [完整任务入口](results/full-task-20261110/index.html)。本次覆盖 3 个领域、4 个原始 subtask,共 6 次真实业务场景运行,其中 5 次有 Harness 完成证据。另有一次沙箱网络受限的启动失败单独留存,无有效 Action 输出,不混入业务成功分母。每次使用新的环境与数据库,不覆盖此前记录。

| 运行 | Action 通过 / 调用 | 完成证据 | 完整链路节点 |
| --- | --- | --- | --- |
| 外卖首次完整运行 `delivery-live/sub_A891207_1` | 7 / 11 | 下单与支付;首轮 Memory 超时后主动补查恢复 | 74 |
| 酒店原始规则 `other-domains/sub_A891207_2` | 3 / 3 | **未完成**:用户模拟器确认房型时提前附带 STOP,没有订单 | 29 |
| 到店推荐 `other-domains/sub_A891207_3` | 3 / 3 | 完成推荐并取得用户确认;仅查询目标 | 30 |
| 酒店复验 `guarded-validation/sub_A891207_2` | 9 / 13 | 4月3日、4月4日两晚大床房分别创建订单且均已支付 | 83 |
| 到店预约 `guarded-validation/sub_A891207_5` | 4 / 4 | 2 人、2026-07-15 20:00,预约状态 unconsumed | 36 |
| 外卖严格模式复验 `guarded-validation/sub_A891207_1` | 7 / 7 | 首轮 Memory 成功,订单已支付 | 55 |

Action 仍为 Qwen3.8-27B,用户模拟器为 GPT。本次结果目录中的独立模型配置将 Qwen 输出上限设为 16,384、HTTP 超时设为 240 秒;根目录模型配置未改。裁判未运行,没有正式奖励或榜单分数。业务工具运行于 VitaBench 模拟环境。

新增 `memory.require_prefetch` 可选严格开关,启用后首轮 Memory 未成功会在调用 Action 前停止。复验开启该选项,Memory 使用最多 5 次有界请求。为处理酒店用户模拟器提前 STOP,诊断入口新增显式 `--user-completion-guard`,在用户提示词中说明“确认方案不等于执行完成”;提示全文保存在 `case.json`,原模型返回仍原样保留,不删除 STOP、不改写回复、不修改 VitaBench 源码。这属于诊断配置,不宣称是原始 benchmark 协议下的分数。

新增 `workspace_committed` 记录实际数据库提交快照。可视化默认按完整时间线展示,原始记录节点数与页面一一对应,完整模型输入、输出、Memory 与工具正文均可展开。Working Memory 提案和实际写入分开显示,校验拒绝不会显示成已生效。可用 `.venv/bin/python -m IMAct.benchmarks.vitabench.episode_report <结果目录>` 重新导出;诊断入口也会自动生成。

三个复验主样本均确认首轮 Memory 注入、输入去重、全部提交快照覆盖、完成证据可回查、无在途工具或 Memory 请求,SQLite 完整性为 ok。执行层 94 项、外层 106 项测试通过。六个页面验证了全部节点、原文展开与折叠的数据和事件处理逻辑;浏览器预览会话被停止,未声称完成截图或像素级验收。代表样本成功不等于全部 OTA 子类、取消/改签和全部 benchmark 任务均已通过真实模型验证。
