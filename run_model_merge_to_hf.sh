#!/bin/bash

STEPS=(60 80)
BASE_LOCAL="checkpoints/MCPFactory/Qwen3-4B-n8-sglang-no_kl-SFT-grpo-0.5-1e-6-20260316-2154/global_step_"
BASE_TARGET="models/MCPFactory/Qwen3-4B-n8-sglang-no_kl-SFT-grpo-0.5-1e-6/step_"

for STEP in "${STEPS[@]}"; do
    LOCAL_DIR="${BASE_LOCAL}${STEP}/actor"
    TARGET_DIR="${BASE_TARGET}${STEP}/"
    
    echo "Merging step ${STEP}..."
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$LOCAL_DIR" \
        --target_dir "$TARGET_DIR"
done

