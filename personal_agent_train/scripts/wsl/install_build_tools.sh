#!/usr/bin/env bash
# 以 root 运行(wsl.exe -u root):装 C 编译器。Triton / vLLM 运行时要编译 C 启动器,flash-attn 等也可能需要。
set -euo pipefail
P=http://127.0.0.1:18080
apt-get -o Acquire::http::Proxy=$P -o Acquire::https::Proxy=$P update -q
DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::http::Proxy=$P -o Acquire::https::Proxy=$P install -y -q build-essential python3-dev
gcc --version | head -1
