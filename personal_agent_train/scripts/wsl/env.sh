# 公共环境:WSL 与原生 Linux 都用这一份,WSL 专属的几项按内核名自动判断。
# 可用环境变量覆盖:PAT_ROOT(两个仓库的父目录)、WORK(工作目录)、PAT_PROXY(出网代理)。

# ---- 路径 ----
# PAT_ROOT 下应有 Personal_agentic(IMAct + 数据 + .env)与 personal_agent_train(本项目)。
if [ -z "${PAT_ROOT:-}" ]; then
  _here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # .../personal_agent_train/scripts/wsl
  export PAT_ROOT=$(cd "$_here/../../.." && pwd)
fi
export WORK=${WORK:-$HOME/pat}
# IMAct(Personal_agentic)的位置:缺省是本项目的同级目录,服务器上目录名不同就 export PAT_IMACT_ROOT=...
export PAT_IMACT_ROOT=${PAT_IMACT_ROOT:-$PAT_ROOT/Personal_agentic}
export HF_HOME=${HF_HOME:-$WORK/hf} UV_CACHE_DIR=${UV_CACHE_DIR:-$WORK/uv-cache}
export PATH=$HOME/.local/bin:$PATH
mkdir -p "$WORK"

case "$(uname -r)" in
  *microsoft*|*WSL*) PAT_IN_WSL=1 ;;
  *) PAT_IN_WSL=0 ;;
esac
export PAT_IN_WSL

# ---- 出网 ----
# WSL:公司网络只放行 Windows 进程,借 Windows 上的 scripts/win_proxy.py 出网(缺省 127.0.0.1:18080)。
# 原生 Linux:默认不设代理;要用就 export PAT_PROXY=http://host:port。
if [ "$PAT_IN_WSL" = 1 ] && [ -z "${PAT_PROXY+x}" ]; then
  PAT_PROXY=http://127.0.0.1:18080
fi
if [ -n "${PAT_PROXY:-}" ]; then
  export http_proxy=$PAT_PROXY https_proxy=$PAT_PROXY HTTP_PROXY=$PAT_PROXY HTTPS_PROXY=$PAT_PROXY
  # 本机地址(含 Ray / Gateway 绑定的节点 IP)绝不能走代理
  _local_ips=$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,*$//')
  export no_proxy=localhost,127.0.0.1,::1${_local_ips:+,$_local_ips}
  export NO_PROXY=$no_proxy
fi

# ---- CUDA / vLLM ----
# 系统没装 CUDA toolkit 时,JIT 编译(flashinfer 等)用 venv 里 pip 装的 toolkit。
if [ -z "${CUDA_HOME:-}" ] && [ ! -d /usr/local/cuda ]; then
  for _cu in "$WORK"/uni-agent/.venv/lib/python3.*/site-packages/nvidia/cu1*; do
    [ -x "$_cu/bin/nvcc" ] && export CUDA_HOME=$_cu && export PATH=$CUDA_HOME/bin:$PATH && break
  done
fi
if [ "$PAT_IN_WSL" = 1 ]; then
  # WSL2(内核 >= 4.19.121)支持 CUDA pinned memory,但 vLLM 在 WSL 上默认关掉;新版 GPU model runner 需要它(UVA)
  export VLLM_WSL2_ENABLE_PIN_MEMORY=1
  # flashinfer 的采样器要现场编译 CUDA;用 PyTorch 实现的采样即可
  export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
fi
