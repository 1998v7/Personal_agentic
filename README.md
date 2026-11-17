# Personal agentic:评测(IMAct)+ RL 训练(uni-agent)

一个仓库放三部分,clone 一次就能在服务器上跑起来:

| 目录 | 是什么 | 谁维护 |
| --- | --- | --- |
| `Personal_agentic/` | **IMAct**:通用 harness(对话循环、用户模拟、工具环境、执行层)+ 四个 bench 的接入与评测 | 我们 |
| `personal_agent_train/` | **训练项目**:把 IMAct 接进 uni-agent,在三类数据上做 agentic RL。训练入口 | 我们 |
| `uni-agent/` | RL 框架(verl 在内),**vendored 快照,不改**,见 `uni-agent/VENDORED.md` | 上游 |

四个 bench:VitaBench 2.0 / τ²-bench / MCP-Universe / PersonaMem-v3。各自的跑法见 `Personal_agentic/README_*.md`。
训练(τ² / deep research / VitaBench 三类数据格式)见 **`personal_agent_train/README.md`**,服务器部署在它的第 3 节。

## 不在仓库里的东西

| | 怎么拿 |
| --- | --- |
| `Personal_agentic/.env`(OpenRouter / MCP 网关 / GitHub / Notion 密钥) | 私有通道传;格式见 `.env.example` |
| `Personal_agentic/data/tau2`(68 MB)、`data/vitabench`(126 MB) | `rsync` 从已有机器传 |
| `Personal_agentic/data/personamem_v3`(1.1 GB) | `python -m IMAct.benchmarks.personamem_v3.download` |
| 模型权重 | `personal_agent_train/scripts/wsl/download_model.sh Qwen/Qwen3.5-2B` |

## 快速开始(Linux 服务器)

```bash
git clone <本仓库> work && cd work
export PAT_ROOT=$PWD
bash personal_agent_train/scripts/wsl/setup_uniagent.sh      # uv + uv sync(约 10 分钟)
bash personal_agent_train/scripts/wsl/install_imact_deps.sh  # IMAct 的依赖装进同一个 venv
bash personal_agent_train/scripts/wsl/test_pat.sh            # 18 个测试应全过
```

目录名不必叫 `Personal_agentic`:`PAT_IMACT_ROOT` 可以指到别处。脚本在 WSL 与原生 Linux 上自动区分
(代理、pinned memory、CUDA_HOME、权重同步方式),细节在 `personal_agent_train/README.md` 第 3 节。
