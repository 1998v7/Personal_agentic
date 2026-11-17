# 这是 uni-agent 的 vendored 快照,不是 git 子模块

| | |
| --- | --- |
| 上游 | https://github.com/verl-project/uni-agent |
| 固定在 | `1fb99945acbe14dd870161bd55c4f9718f472ee3`(2026-11-12) |
| 内含 | `verl/` 原本是子模块(https://github.com/verl-project/verl),这里一并以普通文件收录 |
| 改动 | **无**。本仓库不改 uni-agent / verl 的代码 |

整份收进来的原因:`verl` 是嵌套子模块,只 clone uni-agent 不会带上它;而少了 verl 整个训练栈都跑不起来。
收成普通文件后,服务器上 `git clone` 一次就齐了,不用再 `submodule update --recursive`。
代价是没法直接 `git pull` 上游;要更新就重新 clone 上游对应 commit 并覆盖这个目录。

## 唯一需要在 WSL 上打的补丁(不在本目录里)

WSL2 不实现 `cudaIpcGetMemHandle`,而 verl 的 CUDA 平台把 `is_ipc_supported()` 写死为 True,于是 colocate 训练
在首次权重同步时必然挂死(`CUDA error: invalid resource handle`)。补丁与说明在
`personal_agent_train/scripts/wsl/patch_verl_wsl.sh`:只改 7 行,让 WSL 走 verl 自带的共享内存回退;
幂等、`--check` 可查、在非 WSL 机器上是空操作。**原生 Linux 服务器不需要它。**
