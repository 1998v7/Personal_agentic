#!/usr/bin/env bash
# 单卡(RTX 5080 16GB)冒烟训练:Qwen3.5-2B + LoRA,GRPO,uni-agent 的 Gateway 托管 IMAct 的 τ² / MCP-Universe(/ VitaBench)episode。
#
#   bash scripts/wsl/train.sh                       # 缺省:smoke_train.parquet,4 道题 × n=4,跑 3 步
#   STEPS=2 TRAIN_FILE=~/pat/data/train.parquet bash scripts/wsl/train.sh
#   bash scripts/wsl/train.sh trainer.total_training_steps=5   # 其余参数原样追加给 verl
#
# 路径类的值都加了引号:Hydra 的覆盖语法不接受未加引号的非 ASCII 字符(本机路径里有「新建文件夹」)。
# 权重同步走 NCCL:缺省的 naive checkpoint engine 用 CUDA IPC 在进程间传权重,WSL2 不支持(invalid resource handle)。
# 用户模拟器 / 裁判走 OpenRouter(configs/models.yaml);策略模型是 verl 起的本地 vLLM,经 Gateway 采样。
set -euo pipefail
source "$(dirname "$0")/env.sh"
PAT=$PAT_ROOT/personal_agent_train
export PYTHONPATH=$PAT_IMACT_ROOT:$PAT:$WORK/uni-agent PYTHONUTF8=1
export RAY_DEDUP_LOGS=0 TOKENIZERS_PARALLELISM=false

MODEL_PATH=${MODEL_PATH:-$WORK/models/Qwen3.5-2B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}
TRAIN_FILE=${TRAIN_FILE:-$WORK/data/smoke_train.parquet}
VAL_FILE=${VAL_FILE:-$WORK/data/smoke_val.parquet}
TASK_CONFIG=${TASK_CONFIG:-$PAT/configs/task_config.yaml}
EXP=${EXP:-smoke-$(date +%Y%m%d-%H%M%S)}
OUT=${OUT:-$WORK/runs/$EXP}
STEPS=${STEPS:-3}
BATCH=${BATCH:-4}               # 每步几道题
ROLLOUT_N=${ROLLOUT_N:-4}       # 每道题采样几次(GRPO 组大小);4 × 4 = 每步 16 个 episode
PROMPT_LEN=${PROMPT_LEN:-8192}  # Gateway 的轨迹预算 = PROMPT_LEN + RESPONSE_LEN
RESPONSE_LEN=${RESPONSE_LEN:-8192}
MAX_LEN=$((PROMPT_LEN + RESPONSE_LEN))
LORA_RANK=${LORA_RANK:-32}
LR=${LR:-2e-5}
CONCURRENCY=${CONCURRENCY:-4}   # 同时在跑的 episode 数。注意是主机内存而不是显存的瓶颈:每个 episode 的 harness 都在
                                # framework actor 进程里(dispatch_mode=inline_async),用户模拟器 / 裁判 / MCP 网关的并发也随之
BUCKET_MB=${BUCKET_MB:-1200}    # 权重同步缓冲区。WSL 走共享内存回退(见 patch_verl_wsl.sh),缺省 2048MB 会吃掉十几 GB 主机内存;
                                # 但**必须装得下最大的单个张量**(shm 路径不能把一个张量拆到多个 bucket):
                                # Qwen3.5-2B 的 embed_tokens 是 248320×2048×bf16 = 970 MiB,所以不能低于 ~1024
GPU_MEM=${GPU_MEM:-0.60}        # vLLM 占的显存比例(按整卡 16GB 算;Windows 桌面自己占约 3GB)。采样时 actor 参数在 CPU,
                                # 训练时 vLLM 释放 KV cache;只放得下 0.4 时 KV 不够一条 16k 的轨迹
mkdir -p "$OUT"
echo "run dir: $OUT"

cd "$WORK/uni-agent"
"$WORK/uni-agent/.venv/bin/python" -m verl.trainer.main_ppo \
  --config-name=ppo_trainer \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=sync \
  transfer_queue.enable=True \
  data.train_files="['$TRAIN_FILE']" \
  data.val_files="['$VAL_FILE']" \
  data.prompt_key=prompt \
  data.return_raw_chat=True \
  ++data.apply_chat_template_kwargs.enable_thinking=False \
  data.shuffle=False \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.max_prompt_length=$PROMPT_LEN \
  data.max_response_length=$RESPONSE_LEN \
  data.train_batch_size=$BATCH \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.bypass_mode=False \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
  actor_rollout_ref.model.lora_rank=$LORA_RANK \
  actor_rollout_ref.model.lora_alpha=$((LORA_RANK * 2)) \
  actor_rollout_ref.model.target_modules=all-linear \
  "actor_rollout_ref.model.exclude_modules='.*visual.*'" \
  ++actor_rollout_ref.model.lora.merge=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_LEN \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_LEN \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.prompt_length=$PROMPT_LEN \
  actor_rollout_ref.rollout.response_length=$RESPONSE_LEN \
  actor_rollout_ref.rollout.max_model_len=$MAX_LEN \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=0 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.video=0 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_LEN \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=$BUCKET_MB \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  ++actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
  ++actor_rollout_ref.rollout.custom.agent_framework.allowed_request_sampling_param_keys="[temperature,top_p,top_k]" \
  ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=1 \
  "++actor_rollout_ref.rollout.custom.agent_framework.log_dir='$OUT/agent_logs'" \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=pat_train.runner.run_task \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=inline_async \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=$CONCURRENCY \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=all \
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path='$TASK_CONFIG'" \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="$SERVED_MODEL_NAME" \
  ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=False \
  reward.custom_reward_function.path=pkg://pat_train.runner \
  reward.custom_reward_function.name=score_from_runner_result \
  trainer.project_name=personal_agent_train \
  trainer.experiment_name="$EXP" \
  trainer.logger="['console','tensorboard']" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=$STEPS \
  trainer.default_local_dir="$OUT/ckpt" \
  "$@" 2>&1 | tee "$OUT/train.log"
