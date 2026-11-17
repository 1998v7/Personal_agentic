# personal_agent_train:用 uni-agent 对 IMAct harness 做 RL 训练

把 IMAct 的 harness(对话循环、用户模拟、工具环境、执行层、评分)接进 [uni-agent](../uni-agent) 的 Task 层,在三类数据上做 agentic RL:

| 数据格式 | Task 名 | 一条样本 = | reward |
| --- | --- | --- | --- |
| **τ²-bench**(多轮客服,原生 function calling) | `imact_tau2` | 一个 τ² task 的一段对话 | τ² 官方 reward(0/1) |
| **deep research**(MCP-Universe 的 web_search / financial_analysis / location_navigation,ReAct + 真实 MCP 工具) | `imact_mcp_universe` | 一道题 | 通过的评估器占比(AE,0~1);`reward: sr` 改成全对才 1 |
| **VitaBench**(执行层 JSON 决策 + 外部 Memory) | `imact_vitabench` | 一个 subtask | 执行层拒绝过任何一次决策 → 0;否则 rubric 分 |

被训练的策略模型是 verl 起的本地 vLLM(经 uni-agent 的 Gateway 采样、记录 token),用户模拟器与裁判走 OpenRouter(`configs/models.yaml`)。

```
uni-agent / verl(GRPO)
  └─ Gateway 会话 ──(base_url)──▶ pat_train.runner.run_task ─▶ Task.run()
                                     ├─ imact_tau2        → IMAct Tau2Runner + NativeAgent
                                     ├─ imact_mcp_universe → IMAct McpUniverseRunner(ReAct,经 MCP 网关)
                                     └─ imact_vitabench   → IMAct 执行层 + PersonalizationRunner(Memory 网关)
```

## 1. 训练口径(与评测的区别)

与评测用的是同一套 IMAct 代码,只改了"谁来兜底":

- **策略模型的请求不重试**(`pat_train/policy.py`):Gateway 上的每次请求都会变成一条训练样本,重试等于偷换模型的输出。传输错误直接抛出,这条样本被丢弃。
- **空回复 / 超长截断原样交给对话循环**:按无效消息结束对话、照常评分(通常 0 分),而不是让 harness 重新生成(`Conversation(max_agent_retries=1)`)。
- **工具调用参数不是合法 JSON** → 当作无效消息,不丢样本。
- **VitaBench 的执行层每轮只给一次机会**(`retry.model_call.max_retries=0`),被拒绝即整段 0 分(不调用裁判)。
- **基础设施问题丢样本,不记 0 分**:裁判出错、MCP 评估的网关 / 远端错误、VitaBench 的模型请求传输错误或决策超时、Memory 预取失败。GRPO 组里其余样本照常训练。
- 步数用尽没作答也是模型的行为:`finished=True`、按 0 分训练(不会被 `mask_unfinished_episode` 屏蔽)。

## 2. 数据

```bash
bash scripts/wsl/py.sh -m pat_train.preprocess --out ~/pat/data                     # tau2 + mcp_universe
bash scripts/wsl/py.sh -m pat_train.preprocess --out ~/pat/data --name smoke_ --train-limit 3 --val-limit 2
bash scripts/wsl/py.sh -m pat_train.preprocess --out ~/pat/data --benches tau2 mcp_universe vitabench   # 内网
```

- **tau2**:训练集 = retail 的 base 划分 + telecom 的 full 划分,**去掉 text-v1 的冻结评测子集**(airline 的 base 50 题全在冻结子集里,不进训练集),共 64 + 2235 题;验证集 = 英文 text-v1 冻结子集。telecom 量大,用 `--train-limit N`(每个 data_source 最多 N 条)平衡。
- **mcp_universe**:三个只读领域里不写 notion / github 的 135 题。**它没有训练划分,这些就是评测题**:目前只用来打通 deep research 格式与流程,正式训练请换成同格式的新题(`task_data/` 的 JSON 格式)。
- **vitabench**:personalization 里每个有评分标准的 subtask。同样是评测集本身;且必须带 Memory。

每行的样本字段都在 `extra_info.tools_kwargs.task.metadata`(`domain / split / task_id / subtask`,三类数据同一组键);运行参数(步数、生成长度、reward 口径…)在 `configs/task_config.yaml`。

## 3. 部署:三个目录的关系

```
<父目录>/
├── Personal_agentic/      IMAct:harness、4 个 bench、评测、data/、.env、execution_harness
├── personal_agent_train/  本项目(训练入口)
└── uni-agent/            RL 框架(verl 子模块;只读,不改)
```

`pat_train` 运行时会 import `IMAct.*`,并从 IMAct 侧读 `.env`(密钥)、`data/`(任务数据)、`config.vitabench.yaml`。
目录名不必叫 `Personal_agentic`——用 `PAT_IMACT_ROOT` 指过去即可(`env.sh` 会把它传给所有脚本)。

### 在 Linux 服务器上从零拉起

```bash
cd ~/work                                   # 这就是上面的 <父目录>
git clone <IMAct 仓库> Personal_agentic
git clone <本项目仓库> personal_agent_train

# uni-agent 从上游拉,并钉在与本机验证过的同一个 commit
git clone https://github.com/verl-project/uni-agent.git
git -C uni-agent checkout 1fb99945acbe14dd870161bd55c4f9718f472ee3
git -C uni-agent submodule update --init --recursive     # verl 必须 checkout

# 环境(把 uni-agent 复制到 $WORK 并 uv sync;详见 scripts/wsl/setup_uniagent.sh)
export PAT_ROOT=~/work
bash personal_agent_train/scripts/wsl/setup_uniagent.sh
bash personal_agent_train/scripts/wsl/install_imact_deps.sh
bash personal_agent_train/scripts/wsl/download_model.sh Qwen/Qwen3.5-2B
```

**数据与密钥不在 git 里**(`data/` 1.3 GB、`.env` 含密钥,都被 ignore),单独传:

```bash
rsync -av Personal_agentic/data/tau2      server:~/work/Personal_agentic/data/   # τ² 训练必需,68 MB
rsync -av Personal_agentic/data/vitabench server:~/work/Personal_agentic/data/   # VitaBench 必需,126 MB
rsync -av Personal_agentic/.env           server:~/work/Personal_agentic/        # 密钥,务必只走私有通道
# PersonaMem 的 1.1 GB 不用传,服务器上重新下:python -m IMAct.benchmarks.personamem_v3.download
```

`scripts/wsl/` 这一套脚本 **WSL 与原生 Linux 共用**,按 `uname -r` 自动区分:

| 项 | WSL | 原生 Linux |
| --- | --- | --- |
| 出网代理 | 自动用 `127.0.0.1:18080`(需 Windows 侧 `scripts/win_proxy.py`) | 不设;要用就 `export PAT_PROXY=http://host:port` |
| `VLLM_WSL2_ENABLE_PIN_MEMORY` / flashinfer 采样器 | 自动设 / 自动关 | 不动(用默认,更快) |
| `CUDA_HOME` | venv 里 pip 装的 toolkit | 有 `/usr/local/cuda` 就用系统的 |
| 权重同步 | 共享内存回退,须先打 `patch_verl_wsl.sh` | 原生 CUDA IPC,**不需要补丁** |

**`patch_verl_wsl.sh`**:WSL2 不实现 `cudaIpcGetMemHandle`,而 verl 的 CUDA 平台把 `is_ipc_supported()` 写死为 True,
colocate 权重同步必然报 `CUDA error: invalid resource handle`(表现为挂死)。该脚本只改本机 verl 克隆的 7 行,
让它在 WSL 下走 verl 自带的共享内存回退;幂等,`--check` 查状态,在非 WSL 机器上是空操作。

### 多卡服务器上要放开的单卡限制

`train.sh` 的默认值是为单张 16 GB 卡调的:`trainer.n_gpus_per_node=1`、`rollout.tensor_model_parallel_size=1`、
`GPU_MEM=0.60`、`CONCURRENCY=4`、`BUCKET_MB=1200`、LoRA rank 32、`PROMPT_LEN=RESPONSE_LEN=8192`。
机器大了按需放开(并发、长度、`rollout.n`、去掉 `lora_rank` 做全参)。VitaBench 必须在能访问内网 Memory 网关的机器上跑。

## 4. 环境(Windows + WSL2,一次性)

公司网络只放行 Windows 进程出网,WSL 借 Windows 上的转发代理:

```bash
# Windows(Git Bash),保持运行
python scripts/win_proxy.py --port 18080
```

| 步骤 | 脚本 |
| --- | --- |
| C 编译器(Triton 运行时要编译) | `wsl.exe -u root -- bash scripts/wsl/install_build_tools.sh` |
| uv + uni-agent(同一 commit)+ `uv sync --extra vllm --extra fsdp` | `scripts/wsl/setup_uniagent.sh` |
| IMAct 的依赖装进同一个 venv(不改 uni-agent 锁定的版本) | `scripts/wsl/install_imact_deps.sh` |
| **verl 的 WSL 补丁**(见上) | `scripts/wsl/patch_verl_wsl.sh` |
| 下载模型 | `scripts/wsl/download_model.sh Qwen/Qwen3.5-2B` |
| 测试 | `scripts/wsl/test_pat.sh`(pat_train)、`scripts/wsl/imact_tests.sh discover`(IMAct 全部测试) |
| 排查:MCP 网关是否可达 / vLLM 是否返回 token id | `scripts/wsl/check_gateway.sh`、`scripts/wsl/check_vllm.sh` |

之后若在 uni-agent 里再 `uv sync`,要加 `--inexact`,否则 IMAct 的依赖会被删掉。

## 5. 冒烟:不启动训练器,只看 Gateway 交出的样本

```bash
bash scripts/wsl/serve_vllm.sh &            # 单独起 vLLM(训练时不需要)
bash scripts/wsl/py.sh -m pat_train.smoke --data ~/pat/data/train.parquet --rows 0 74 67 82 2 --out ~/pat/smoke
```

每个 session 输出 `trajectories.jsonl`(训练器收到的 token / mask / logprob)、`readable.txt`(按 mask 分段解码:`[MODEL]` 段训练,`[ENV]` 段不训练)与摘要(轨迹数、可训练 token、rollback 丢掉的 token)。τ² 一段对话 = 一条轨迹(多轮在 Gateway 上前缀续接,rollback 0);MCP 的 ReAct 每步重新渲染整个 prompt,一道题 = 多条单轮轨迹,共享这道题的 reward。

## 6. 训练

```bash
bash scripts/wsl/train.sh                                   # smoke_train.parquet,每步 4 道题 × n=4,3 步
STEPS=2 BATCH=4 ROLLOUT_N=4 TRAIN_FILE=~/pat/data/train.parquet bash scripts/wsl/train.sh
```

单卡 16GB 的设置:FSDP2 + LoRA(rank 32,合并后同步给 vLLM,`lora.merge=True`),参数 / 优化器 offload,fused kernel 算 log-prob(词表 248k,不能展开整张 logits),`trainer_mode=sync`(采样 → 训练 → 同步权重,无 off-policy)。每条轨迹的预算 `PROMPT_LEN + RESPONSE_LEN = 16k` token,超了 Gateway 返回空回复、对话按无效消息结束。

单机上踩过、已固化成默认值的三个坑:

- `dispatch_mode=inline_async`(不是 `ray_task`):每个 episode 一个 Ray worker 进程会各吃约 1.1 GB 主机内存,16 并发直接把 45 GB 打满、Ray 开始杀 worker。inline 模式下 episode 跑在 framework 进程的线程里(Task 自己用 `asyncio.to_thread` 包住阻塞调用)。
- `BUCKET_MB` 既不能太大也不能太小:默认 2048 MB 在共享内存回退下吃掉十几 GB;但**必须装得下最大的单个张量**(shm 路径不能把一个张量拆到多个 bucket),Qwen3.5-2B 的 `embed_tokens` 是 248320×2048×bf16 = 970 MiB,所以不能低于 ~1024。
- `GPU_MEM` 太小时 vLLM 的 KV cache 装不下一条 16k 轨迹而拒绝启动(0.4 → 0.03 GiB 可用)。

结果在 `~/pat/runs/<实验名>/`:`train.log`、`agent_logs/`(每个 session 的轨迹)、tensorboard。

## 7. 代码

| 路径 | 作用 |
| --- | --- |
| `pat_train/runner.py` | uni-agent 的 runner 入口:登记 Task 后交给官方 `run_task` |
| `pat_train/agent.py` | 登记 `imact` agent 配置(模型地址 + 每轮生成长度);样本字段的读取 |
| `pat_train/policy.py` | Gateway 会话地址 → IMAct `LLMCaller`(不重试、截断原样返回) |
| `pat_train/shared.py` | 进程内共享的事件循环线程与用户模拟器 / 裁判 |
| `pat_train/tasks/{tau2,mcp_universe,vitabench}.py` | 三类 Task |
| `pat_train/preprocess.py` | 生成 parquet |
| `pat_train/smoke.py` | Gateway + vLLM 冒烟 |
| `configs/task_config.yaml`、`configs/models.yaml` | Task 运行参数;辅助角色模型 |
| `scripts/wsl/*.sh`、`scripts/win_proxy.py` | 环境、测试、训练脚本 |

IMAct 本身只加了两个可选参数(缺省行为不变):`Conversation(max_agent_retries=...)`、`Tau2Runner(conversation_options=...)`。
