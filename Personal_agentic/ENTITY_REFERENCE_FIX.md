# Action 输出预算与实体引用修复（2026-11-12）

> 本文记录原调试工作区的修复验证；`results/` 原始运行产物不随源码提交。

原外卖轨迹的 36 次 Action 尝试中，12 次在 4,096 token 上限处截断；另有 12 次使用工具已经返回的店铺 ID，却因店铺未作为独立候选登记而被 Harness 拒绝。本次修复针对这两类问题，保留真实工具回执和已有校验边界。

## 输出预算

`models.yaml` 中 `qwen3.8-27b` 的 `max_tokens` 从 4,096 调整为 16,384，模型请求超时从 120 秒调整为 300 秒。执行层按现有规则自动将决策超时设为 330 秒。模拟用户和 judge 配置没有变化；继续保留 thinking 原有行为，没有额外关闭推理。

提高的是生成上限，不是强制每轮生成该数量。输出实际仍需正常结束并通过联合 JSON 校验；`finish_reason=length` 继续视为不完整输出，不将截断正文强行当成有效动作。

## 实体引用

商品工具回执中的类型化父实体字段会登记为候选。例如 `StoreProduct(product_id=P1, store_id=S1, ...)` 在保留 P1 的同时登记 S1：

```json
{
  "item_id": "S1",
  "kind": "store",
  "store_id": "S1",
  "record_type": "EntityReference",
  "identity_only": true,
  "source_refs": ["事件ID/工具调用ID"]
}
```

这只证明 ID 及商品所属关系，不生成店铺评分、配送时间、地址等未观察字段。模型可用该 ID 查询店铺详情，也可在满足现有下单校验的条件下使用它与所属商品创建订单。后续 Store 回执补齐详情并移除 `identity_only`；再次查询商品不会覆盖已有店铺详情。

- 只接受已解析商品回执中的明确外键，不从用户文本、Memory、名字、ID 前缀或模型实参猜测实体。
- 对商家、酒店、航班、车次和车型采用同样的类型化关联处理。景点商品的 `attraction_id` 在上游实际代表票号，只有嵌套景点详情确认的父实体才能用于关联，不将票号冒充景点 ID。
- 订单内的商品明细不自动加入可购买候选；错误回执也不会登记引用。
- 原 `validate_call` 的可见性、实体类型、商品与店铺匹配、未知写入保护仍然生效。凭空生成 ID、错误类型、跨店下单仍会被拒绝。
- 上下文裁剪时，保留商品就同时保留已登记的父实体；原始回执仍完整归档。派生实体不计入回执原始记录数，不破坏已有的上下文去重。

代码位置：`execution_harness/src/imact/vitabench.py` 的 `product_parent`，`domain.py` 的 `_register_parent_reference` 与 `observe`，以及 `context.py` 的回执投影与候选裁剪。Action prompt 增加 `identity_only` 的含义。

## 验证

- 执行层：111 项 pytest 通过，包含新增的实体引用、错误类型、跨店商品、详情补齐和裁剪回归。
- 外层：121 项 unittest 通过。
- 修改文件的 Ruff 检查与格式检查通过。
- 冻结旧轨迹重放：原来 12 次 `ungrounded_id` 决策的工具调用，使用各自时点已可见的工具回执，12 次均通过新逻辑；额外的伪造 ID 仍被拒绝。该验证没有调用模型或工具，不代表重跑任务成功。

测试日志、离线重放脚本与结果存放在 [results/entity-reference-fix-20261112](results/entity-reference-fix-20261112/)。

## 真实复跑结果

原始用例 `sub_A891207_1` 已完成本地 Memory → Action → VitaBench 工具与模拟用户 → judge 的真实模型复跑。配置保留原始题目、seed=300、max_steps=32，未开启用户结束提示补丁；业务操作仅发生在模拟环境中。

| 指标 | 修复前轨迹 | 本次复跑 |
| --- | --- | --- |
| Action 调用 | 36 | 9 |
| 输出截断 | 12 | 0 |
| 店铺 ID 校验拒绝 | 12 | 0 |
| 全部失败尝试 | 27 | 2 |
| 模拟业务工具调用/回执 | 5/5 | 8/8 |
| 已支付订单 | 1 | 1 |
| 耗时 | 1530.85 秒 | 416.51 秒 |
| Judge reward | 0 | 1 |
| Rubric 通过 | 5/7 | 7/7 |

9 次模型输出均为 `finish_reason=stop`；其中 3 次超过旧 4,096 上限，最长 5,745 token。商品搜索后直接查询 4 家店铺详情，未再被店铺引用校验阻塞。订单 `OTd5af1b691e` 支付成功；judge 的 3 个窗口均有完整返回，SQLite 完整性为 `ok`。

剩余两次失败分别为 `unverified_resolution`（工具事件中关闭通用待确认事项）和 `invalid_decision`（issue 笔记遗漏必填 topic），均通过重试恢复；本次没有修改这些规则。两条轨迹的 Memory 内容、模型决策和餐品选择存在随机差异，reward 和耗时对比只记录本次观察，不作为受控性能实验结论。

最终补充的候选裁剪保护通过离线回归验证；真实进程启动后已加载此前代码，且本轮 9 次上下文均未触发候选裁剪，该新增预算分支未由此轮真实调用验证。

- [完整新轨迹](results/entity-reference-fix-20261112/live/sub_A891207_1/episode.html)
- [Judge 评分与依据](results/entity-reference-fix-20261112/live/sub_A891207_1/grading.html)
- [机器可读核验与代码哈希](results/entity-reference-fix-20261112/verification.json)
- [离线重放 12 次原始拒绝](results/entity-reference-fix-20261112/replay-verification.json)

## 复跑

在 Personal_IMAct 工作目录下使用项目解释器，每次选择新的输出目录：

```bash
.venv/bin/python -m IMAct.benchmarks.vitabench.diagnose \
  --config config.vitabench.yaml --subtask-ids sub_A891207_1 \
  --with-reward --max-steps 32 --out results/entity-reference-next
```

本次没有修改 rubric、数据集、Memory 检索策略、工作笔记关闭规则或任务终止逻辑。旧轨迹保留用于对照；提升输出上限和修复实体引用不保证所有用例都无截断，也不等于 reward 必然提高。
