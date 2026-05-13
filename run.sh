#!/bin/bash
# Refer to https://verl.readthedocs.io/en/latest/examples/config.html for explanation

set -x
ulimit -n 65535

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/EnvFactory/configs"
CONFIG_NAME="env_factory_config"
TIMESTAMP="$(date +"%Y%m%d-%H%M")"
BACKEND=sglang

MODEL_NAME=EnvFactory-RL-30B-A3B-Thinking
MODEL_PATH=//mnt/public/dengheyuan/models/Qwen3-30B-A3B-Thinking-2507

# MODEL_NAME=EnvFactory-RL-35B-A3B
# MODEL_PATH=//mnt/public/dengheyuan/models/Qwen3.5-35B-A3B

EXP_NAME="$MODEL_NAME-n8-$BACKEND-no_kl-grpo-1e-6-0.7"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=/mnt/public/dengheyuan/project/EnvFactory/data/rl/env_factory_rl.json \
    data.val_files=/mnt/public/dengheyuan/project/EnvFactory/data/rl/env_factory_rl_val.json \
    data.train_batch_size=256 \
    data.max_prompt_length=16384 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    reward.custom_reward_function.path=EnvFactory/reward/tool_reward_fcn.py \
    reward.custom_reward_function.name=compute_tool_reward \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=$BACKEND \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.trace.backend=wandb \
    actor_rollout_ref.rollout.trace.token2text=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='MCPFactory' \
    trainer.experiment_name="${EXP_NAME}-${TIMESTAMP}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    trainer.validation_data_dir=$PROJECT_DIR/log_validation/$EXP_NAME \
    trainer.rollout_data_dir=$PROJECT_DIR/log_rollout/$EXP_NAME \
    $@