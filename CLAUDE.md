# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-Forcing-Plus is a video generation model distillation framework for step distillation and CFG (Classifier-Free Guidance) distillation of bidirectional diffusion models. It trains 4-step T2V-14B and I2V-14B models for faster inference, building on CausVid, Self-Forcing, and Wan2.1.

**License**: CC-BY-NC-SA 4.0 (Non-commercial use only)

## Common Commands

### Installation
```bash
conda create -p /path/to/conda_env python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

### Training (T2V-14B DMD)
```bash
torchrun --nnodes=8 --nproc_per_node=8 \
  --rdzv_id=5235 --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py \
  --config_path configs/self_forcing_14b_dmd.yaml \
  --logdir logs/self_forcing_14b_dmd \
  --no_visualize --disable-wandb
```

### Inference
```bash
# T2V inference
python inference.py \
  --config_path configs/self_forcing_14b_dmd.yaml \
  --checkpoint_path /path/to/checkpoint.pt \
  --data_path prompts/MovieGenVideoBench.txt \
  --output_folder outputs/

# I2V inference (add --i2v flag)
python inference.py --i2v \
  --config_path configs/self_forcing_14b_i2v_dmd.yaml \
  --checkpoint_path /path/to/checkpoint.pt \
  --data_path /path/to/image_text_pairs/ \
  --output_folder outputs/
```

### Demo Web UI
```bash
python demo.py \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --config_path configs/self_forcing_dmd.yaml \
  --port 5001
```

### Dataset Preparation (I2V)
```bash
# 1. Compute VAE latents from videos
python scripts/compute_vae_latent.py \
  --input_video_folder {video_folder} \
  --output_latent_folder {latent_folder} \
  --model_name Wan2.1-T2V-14B \
  --prompt_folder {prompt_folder}

# 2. Create LMDB dataset
python scripts/create_lmdb_14b_shards.py \
  --data_path {latent_folder} \
  --prompt_path {prompt_folder} \
  --lmdb_path {lmdb_folder}
```

### Checkpoint Conversion
```bash
python convert_checkpoint.py --input_path /path/to/fsdp_ckpt --output_path /path/to/output.pt
```

## Architecture

```
train.py → Trainer → Model → Pipeline → WanModel (Wan2.1)
                ↓
         FSDP wrapping for distributed training
```

### Core Components

**Entry Points:**
- `train.py` - Training orchestrator (parses config, selects trainer, runs training loop)
- `inference.py` - Distributed inference for T2V/I2V
- `demo.py` - Flask web UI for interactive generation

**Trainers** (`trainer/`):
- `ScoreDistillationTrainer` (distillation.py) - Main trainer for DMD/SiD distillation
- `DiffusionTrainer`, `GANTrainer`, `ODETrainer` - Alternative training modes

**Models** (`model/`):
- `DMD` (dmd.py) - Distribution Matching Distillation model
- `SiD` (sid.py) - Score-based Implicit Distillation
- `CausVid` (causvid.py) - Causal video model

**Pipelines** (`pipeline/`):
- `SelfForcingTrainingPipeline` - Few-step training logic
- `CausalInferencePipeline` - 4-step inference
- `BidirectionalTrainingPipeline` / `BidirectionalInferencePipeline` - For 14B models

**Model Wrappers** (`utils/wan_wrapper.py`):
- `WanDiffusionWrapper` - Main diffusion model wrapper
- `WanTextEncoder` - T5-based text encoding
- `WanCLIPEncoder` - CLIP image encoding (for I2V)
- `WanVAEWrapper` - VAE encoder/decoder

### Configuration System

Configs use OmegaConf with layered merging:
1. `configs/default_config.yaml` - Base defaults
2. Specific config (e.g., `self_forcing_14b_dmd.yaml`) - Overrides
3. CLI arguments - Final overrides

Key config parameters:
- `trainer`: `score_distillation` | `diffusion` | `gan` | `ode`
- `distribution_loss`: `dmd` | `sid` | `causvid`
- `generator_type`: `bidirectional` (14B) | `causal` (1.3B)
- `denoising_step_list`: `[1000, 750, 500, 250]` for 4-step training
- `sharding_strategy`: `full` (14B) | `hybrid_full` (1.3B)

### Distributed Training

- Uses FSDP (Fully Sharded Data Parallel) via `utils/distributed.py`
- EMA maintained with FSDP-aware utilities (`EMA_FSDP`)
- Supports 64 H100 GPUs (8 nodes × 8 GPUs)
- Mixed precision with bfloat16

### Key Data Flow

**Training:**
1. Text prompts loaded from folder (each `.txt` file = one prompt) or LMDB (I2V)
2. Text encoded via T5 (`WanTextEncoder`)
3. Noise sampled, denoised through `denoising_step_list` timesteps
4. DMD loss computed between generator and teacher outputs
5. Generator and critic updated alternately

**Inference:**
1. Prompts loaded via `TextDataset` or `TextImagePairDataset`
2. Few-step denoising (4 steps) through pipeline
3. Latents decoded via VAE
4. Video saved as MP4 (16 fps)
