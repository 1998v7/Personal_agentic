# 本机安装与验证记录（2026-11-10）

> 本文保留 2026-11-10 的本机运行记录；其中仓库路径、提交号和服务状态为当时快照。`results/` 中的原始运行产物保留在原工作区，不随本次源码提交。

完整仓库位置：`/Users/longjing/PycharmProjects/imact-execution-harness/Personal_IMAct`。

已完成本机环境准备、两层测试和进程内协议闭环。安装阶段的真实模型探测曾被执行沙箱网络权限阻断；随后已获准在沙箱外完成两轮真实模型运行，启用 Memory 的一轮完成模拟支付，但裁判请求失败，尚无有效 LLM 评分。最新结果、原始输出与复跑命令见 [真实模型运行记录](REAL_MODEL_RUN.md)。下文保留安装阶段的验证事实。

## 代码与环境

- 远端：`git@code.alipay.com:adrain.wq/Personal_IMAct.git`。
- 分支：`main`；本次拉取时远端只公布了此分支。
- 提交：`c120d3566fea4bc83f5dc78979f41ad908c5f1b4`。
- 原目录没有 `.git`，不能从 Git 元数据确认它原来的分支名。比对其执行层源码、测试、脚本、示例和配置，共 33 个文件一致，`src/imact/prompts/action.txt` 有差异；原目录保留。
- Python：3.13.5；完整仓库使用自己的 `.venv`。
- 在线 pip 安装受沙箱网络限制，因此将本机已安装的 Python 3.13 依赖复制到新 `.venv`；没有引入旧项目的 editable 路径。`pip check` 通过，依赖清单见 `results/setup/environment.freeze.txt`，复制记录见 `results/setup/local_dependencies.json`。
- VitaBench：`VitaBench-2.0/`，固定提交 `f60169e89f30499cb7883f3dad76bd03facc908d`。使用本机已有 Git 对象恢复此提交的独立工作区，没有复制旧工作区的源码修改；本次检查其受跟踪文件无变化。
- 数据集：从本机已有数据复制 `tasks.json`，共 56 个用户。路径为 `VitaBench-2.0/data/vita/domains/personalization/tasks.json`，大小 131234835 字节，SHA-256 为 `3a05f7e2742f204ae28b71db95b14e2daa94c9ff354f510a7d9c707564c7466e`。本次未重新核对 Hugging Face 远端版本，来源与校验值见 `results/setup/dataset.json`。

## 验证结果

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 外层 unittest | 93 项通过 | `results/setup/outer-tests.log` |
| 执行层 pytest | 77 项通过，7.51 秒 | 本次命令输出；汇总见 `results/setup/verification.json` |
| 进程内 ASGI 冒烟 | 通过；7 个事件、14 次请求、7 次模型决策 | `execution_harness/data/verification/asgi-smoke.json` |
| GPT 配置检查 | 通过；1 个用户、1 个子任务 | `results/setup/gpt-check.log` |
| Claude / Gemini 配置检查 | 均通过；各选 1 个用户、1 个子任务 | `results/setup/claude-check.log`、`results/setup/gemini-check.log` |
| 独立 HTTP 冒烟 | 未通过；沙箱禁止绑定 localhost 端口 | `sock.bind` 报 `PermissionError: [Errno 1] Operation not permitted` |
| GPT 真实模型探测 | 未通过；沙箱禁止网络连接 | `results/setup/gpt-live-smoke.log`，`httpx.ConnectError: [Errno 1] Operation not permitted` |

单元测试与 ASGI 冒烟使用固定模型、模拟 HTTP 或合成回执。配置检查只验证配置、密钥变量存在和数据可加载，不验证密钥有效性、服务可达性或模型可用性。真实模型探测在 benchmark 执行前失败，因此没有任务完成率或 reward。Memory 默认关闭，本次没有验证真实 Memory 网关。

## 复跑

在可访问 MatrixLLM 的本机终端执行以下命令。脚本会读取仓库根目录 `.env`，先检查配置，再进行一次小请求探测，通过后运行 1 个用户的 1 个子任务，最多 20 步、任务级重试为 0。

```bash
cd /Users/longjing/PycharmProjects/imact-execution-harness/Personal_IMAct
PYTHON=.venv/bin/python bash scripts/smoke_matrixllm.sh gpt
```

可将 `gpt` 替换成 `claude` 或 `gemini`。只检查配置时追加 `--check-only`。显式联调 Memory 时追加 `--memory`，并按实际服务设置 `MEMORY_GATEWAY_URL`。

成功启动后，结果写入 `results/matrixllm-smoke-<profile>-<时间戳>/`，包括 `summary.json`、用户轨迹 JSON、`run.log` 和执行层数据库 `exec.sqlite3`。只有读取这些结果并核对失败项、终止原因和评分，才能判断实际任务是否成功。

复跑离线验证：

```bash
cd /Users/longjing/PycharmProjects/imact-execution-harness/Personal_IMAct
.venv/bin/python -m unittest discover -s tests -t .
cd execution_harness
PYTHONPATH=src ../.venv/bin/python -m pytest -q
PYTHONPATH=src ../.venv/bin/python scripts/verify_inprocess.py
PYTHONPATH=src ../.venv/bin/python scripts/verify_http.py
```

执行层通过 `PYTHONPATH=src` 或外层自带 loader 导入；没有对它做 editable 安装。最后一条命令需要允许 localhost 监听，完成后会自动关闭临时服务。
