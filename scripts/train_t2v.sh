
export MASTER_ADDR=localhost
export MASTER_PORT=29500

torchrun --nnodes=1 --nproc_per_node=8 \
--rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
train.py \
--config_path configs/self_forcing_14b_dmd.yaml \
--logdir logs/self_forcing_14b_dmd \
--no_visualize \
# --disable-wandb