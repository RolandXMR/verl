# Refer to https://verl.readthedocs.io/en/latest/examples/config.html for explanation

set -x
ulimit -n 65535
module load gcc/11.5 # use `module avail gcc` to check availble gcc
# module load glibc/2.32 # use `module avail glibc` to check availble glibc

export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/MCPFactory/configs"
CONFIG_NAME="mcp_factory_grpo"
TIMESTAMP="$(date +"%Y%m%d-%H%M")"
MODEL_NAME=Qwen3-1.7B
EXP_NAME="$MODEL_NAME-n8-vllm-think-no_kl-grpo-0.5trace-1e-6"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$PROJECT_DIR/data/MCPFactory/mcp_factory_rl_train.json \
    data.val_files=$PROJECT_DIR/data/MCPFactory/mcp_factory_rl_val.json \
    data.train_batch_size=256 \
    data.max_prompt_length=16384 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=False \
    data.trace_reward_weight=0.5 \
    actor_rollout_ref.model.path=/data/user/mxubh/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.trace.backend=tensorboard \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$PROJECT_DIR/tools/mcp_configs/mcp_server.json \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='MCPFactory' \
    trainer.experiment_name="${EXP_NAME}-${TIMESTAMP}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=6 \
    trainer.validation_data_dir=$PROJECT_DIR/log_validation/$EXP_NAME \
    trainer.rollout_data_dir=$PROJECT_DIR/log_rollout/$EXP_NAME \
    $@

# gpu_memory_utilization: 8b: 0.6; 4b: 0.85 1.7b: 0.9