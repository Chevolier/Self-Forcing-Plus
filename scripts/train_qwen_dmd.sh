#!/bin/bash

# Qwen-Image-Edit DMD Training Script
# Uses LoRA for memory-efficient training with 8-step denoising

export MASTER_ADDR=localhost
export MASTER_PORT=29501

# Number of GPUs (adjust based on your setup)
NUM_GPUS=${NUM_GPUS:-8}

# Config and output paths
CONFIG_PATH=${CONFIG_PATH:-configs/qwen_image_dmd.yaml}
LOG_DIR=${LOG_DIR:-logs/qwen_image_dmd}

echo "Training Qwen-Image-Edit DMD"
echo "Config: ${CONFIG_PATH}"
echo "Log dir: ${LOG_DIR}"
echo "GPUs: ${NUM_GPUS}"

torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    --rdzv_id=5236 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    train.py \
    --config_path ${CONFIG_PATH} \
    --logdir ${LOG_DIR} \
    --no_visualize \
    # --disable-wandb \
    "$@"
