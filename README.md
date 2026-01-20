<p align="center">
<h1 align="center">Self Forcing Plus</h1>

Self-Forcing-Plus focuses on step distillation and CFG distillation for bidirectional models. Building upon Self-Forcing, we support 4-step T2V-14B model training and higher quality 4-step I2V-14B model training.

## 🔥 News
- (2025/09) Support Wan2.2-Moe distillation! [wan22](https://github.com/GoatWu/Self-Forcing-Plus/tree/wan22)

| Model Type | Model Link |
|------------|---------------|
| T2V-14B | [Huggingface](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill) |
| I2V-14B-480P | [Huggingface](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v) |

## Installation
Create a conda environment and install dependencies:
```
conda create -p /home/ec2-user/SageMaker/efs/conda_envs/self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Quick Start
### Download checkpoints
```
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir /home/ec2-user/SageMaker/efs/Models/wan_models/Wan2.1-T2V-14B
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir /home/ec2-user/SageMaker/efs/Models/wan_models/Wan2.1-I2V-14B-480P
```

## T2V Training

DMD training for bidirectional models do not need ODE initialization.

### DataSet Preparation

We build the dataset in the following way, each file contains a single prompt:

```
data_folder
  |__1.txt
  |__2.txt
  ...
  |__xxx.txt
```

<!-- ### DMD Training
```
torchrun --nnodes=8 --nproc_per_node=8 \
--rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
train.py \
--config_path configs/self_forcing_14b_dmd.yaml \
--logdir logs/self_forcing_14b_dmd \
--no_visualize \
--disable-wandb
``` -->

Our training run uses 3000 iterations and completes in under 3 days using 64 H100 GPUs.

## Qwen-Image-Edit-2509 Training

<!-- ### DataSet Preparation

1. Generate a series of videos using the original Wan2.1 model.

2. Generate the VAE latents.
```bash
python scripts/compute_vae_latent.py \
--input_video_folder {video_folder} \
--output_latent_folder {latent_folder} \
--model_name Wan2.1-T2V-14B \
--prompt_folder {prompt_folder}
```

3. Separate the first frame of the videos and create an lmdb dataset.
```bash
python scripts/create_lmdb_14b_shards.py \
--data_path {latent_folder} \
--prompt_path {prompt_folder} \
--lmdb_path {lmdb_folder}
``` -->

### DMD Training

<!-- ```
torchrun --nnodes=8 --nproc_per_node=8 \
--rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
train.py \
--config_path configs/self_forcing_14b_i2v_dmd.yaml \
--logdir logs/self_forcing_14b_i2v_dmd \
--no_visualize \
--disable-wandb
``` -->


```bash
nohup bash scripts/train_qwen_dmd.sh > logs/dmd.out 2>&1 &
```

### Evaluation

```bash
# Evaluate DMD trained LoRA models
CUDA_VISIBLE_DEVICES=5 python benchmark.py \
      --model_path /home/ec2-user/SageMaker/efs/Projects/Qwen-Image-Edit-Acceleration/checkpoints/Qwen-Image-Edit-2509-step4000 \
      --test_path /home/ec2-user/SageMaker/efs/Projects/Qwen-Image-Edit-Acceleration/data/test_data.csv \
      --data_dir /home/ec2-user/SageMaker/efs/Projects/Qwen-Image-Edit-Acceleration/data \
      --output_dir /home/ec2-user/SageMaker/efs/Projects/Qwen-Image-Edit-Acceleration/outputs/H100_ckpt4k_dmd_istep40_cfg4 \
      --cfg_scale 4 \
      --num_inference_steps 40 \
      --prompt "让图2的模特换上图1的下装" \
      --max_samples 50 \
      --seed 0 \
      > /home/ec2-user/SageMaker/efs/Projects/Qwen-Image-Edit-Acceleration/logs/H100_ckpt4k_dmd_istep40_cfg4.out 2>&1 &

CUDA_VISIBLE_DEVICES=6 python benchmark.py \
      --model_path /home/ec2-user/SageMaker/efs/Projects/Qwen-Image-Edit-Acceleration/checkpoints/Qwen-Image-Edit-2509-step4000 \
      --test_path /home/ec2-user/SageMaker/efs/Projects/Qwen-Image-Edit-Acceleration/data/test_data.csv \
      --data_dir /home/ec2-user/SageMaker/efs/Projects/Qwen-Image-Edit-Acceleration/data \
      --output_dir /home/ec2-user/SageMaker/efs/Projects/Qwen-Image-Edit-Acceleration/outputs/H100_ckpt4k_dmd_istep8_cfg1 \
      --cfg_scale 1 \
      --num_inference_steps 8 \
      --prompt "让图2的模特换上图1的下装" \
      --max_samples 50 \
      --seed 0 \
      > /home/ec2-user/SageMaker/efs/Projects/Qwen-Image-Edit-Acceleration/logs/H100_ckpt4k_dmd_istep8_cfg1.out 2>&1 &


CUDA_VISIBLE_DEVICES=7 python benchmark.py \
      --model_path /home/ec2-user/SageMaker/efs/Projects/Qwen-Image-Edit-Acceleration/checkpoints/Qwen-Image-Edit-2509-step4000 \
      --lora_path /home/ec2-user/SageMaker/efs/Projects/Self-Forcing-Plus/checkpoints/checkpoint_000200/generator.safetensors \
      --test_path /home/ec2-user/SageMaker/efs/Projects/Qwen-Image-Edit-Acceleration/data/test_data.csv \
      --data_dir /home/ec2-user/SageMaker/efs/Projects/Qwen-Image-Edit-Acceleration/data \
      --output_dir /home/ec2-user/SageMaker/efs/Projects/Qwen-Image-Edit-Acceleration/outputs/H100_ckpt4k_dmd_sam3k_tstep200_istep8 \
      --cfg_scale 1 \
      --num_inference_steps 8 \
      --prompt "让图2的模特换上图1的下装" \
      --max_samples 50 \
      --seed 0 \
      --debug \
      > /home/ec2-user/SageMaker/efs/Projects/Qwen-Image-Edit-Acceleration/logs/H100_ckpt4k_dmd_sam3k_tstep200_istep8.out 2>&1 &

```

Our training run uses 1000 iterations and completes in under 12 hours using 64 H100 GPUs.

## Acknowledgements
This codebase is built on top of the open-source implementation of [CausVid](https://github.com/tianweiy/CausVid), [Self-Forcing](https://github.com/guandeh17/Self-Forcing) and the [Wan2.1](https://github.com/Wan-Video/Wan2.1) repo.
