"""
Benchmark script for Qwen-Image-Edit models using this repo's implementation.
Supports LoRA loading from trained checkpoints.
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional, List

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from safetensors.torch import load_file

from utils.qwen_wrapper import (
    QwenDiffusionWrapper,
    QwenTextEncoderWrapper,
    QwenVAEWrapper,
)
from pipeline.qwen_image_training import QwenImageInferencePipeline


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    num_inference_steps: int = 8


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Qwen-Image-Edit models")
    parser.add_argument("--model_path", type=str, default="qwen_models/Qwen-Image-Edit-2509",
                        help="Path to the base model")
    parser.add_argument("--test_path", type=str, default="data/test_data.csv",
                        help="Path to the test data CSV file")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Base directory for test data")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Base output directory")
    parser.add_argument("--num_inference_steps", type=int, default=8,
                        help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="CFG scale for classifier-free guidance (1.0 = no guidance)")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt for image editing")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Negative prompt for CFG (used when cfg_scale > 1.0)")
    parser.add_argument("--cfg_norm_rescale", action="store_true",
                        help="Enable CFG norm rescaling (matches official diffusers behavior)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--config_name", type=str, default=None,
                        help="Configuration name for output directory")
    parser.add_argument("--warmup_runs", type=int, default=1,
                        help="Number of warmup runs before benchmarking")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--max_pixels", type=int, default=1024 * 1024,
                        help="Maximum number of pixels for output images")
    parser.add_argument("--height_division_factor", type=int, default=16,
                        help="Height division factor for VAE compatibility")
    parser.add_argument("--width_division_factor", type=int, default=16,
                        help="Width division factor for VAE compatibility")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA checkpoint (safetensors or directory)")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank (must match training)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (must match training)")
    parser.add_argument("--scheduler_mu", type=float, default=0.8,
                        help="Scheduler mu parameter")
    parser.add_argument("--custom_timesteps", type=str, default=None,
                        help="Custom timesteps as comma-separated values")
    parser.add_argument("--edit_image_column", type=str, default="model_image",
                        help="Column name for edit image in CSV (default: model_image)")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug information about shapes and processing")
    return parser.parse_args()


def get_target_size(image: Image.Image, max_pixels: int, height_div: int, width_div: int) -> tuple:
    """
    Compute target size based on max_pixels constraint.
    Matches DiffSynth-Studio's QwenImagePipeline sizing logic.

    The algorithm:
    1. Calculate dimensions that maintain aspect ratio with target area = max_pixels
    2. Round to nearest 32 (not floor) - this matches official diffusers behavior
    3. Floor to division factor (16) for VAE compatibility
    """
    import math

    width, height = image.size

    # Calculate dimensions with target area (matching DiffSynth official)
    target_area = max_pixels
    aspect_ratio = width / height

    # Calculate new dimensions maintaining aspect ratio
    calc_width = math.sqrt(target_area * aspect_ratio)
    calc_height = calc_width / aspect_ratio

    # Round to nearest 32 (matching DiffSynth/official diffusers)
    calc_width = round(calc_width / 32) * 32
    calc_height = round(calc_height / 32) * 32

    # Floor to division factor (16) for VAE compatibility
    # This is redundant when rounding to 32 (since 32 is multiple of 16)
    # but kept for safety with other division factors
    calc_height = int(calc_height) // height_div * height_div
    calc_width = int(calc_width) // width_div * width_div

    return int(calc_height), int(calc_width)


def preprocess_image(image: Image.Image, target_height: int, target_width: int) -> torch.Tensor:
    """
    Preprocess PIL image to tensor for VAE encoding.
    Returns tensor in [-1, 1] range with shape [1, 3, H, W].
    """
    # Resize to target dimensions
    image = image.resize((target_width, target_height), Image.LANCZOS)

    # Convert to tensor
    import torchvision.transforms.functional as TF
    tensor = TF.to_tensor(image)  # [0, 1] range

    # Scale to [-1, 1]
    tensor = tensor * 2 - 1

    # Add batch dimension
    tensor = tensor.unsqueeze(0)  # [1, 3, H, W]

    return tensor


def generate_config_name(args) -> str:
    """Generate a configuration name based on the parameters."""
    if args.config_name:
        return args.config_name
    lora_suffix = "_lora" if args.lora_path else ""
    custom_suffix = "_custom_ts" if args.custom_timesteps else ""
    return f"steps{args.num_inference_steps}_seed{args.seed}{lora_suffix}{custom_suffix}"


def setup_output_dirs(output_dir: str, config_name: str) -> tuple:
    """Create output directories for images and results."""
    config_dir = os.path.join(output_dir, config_name)
    images_dir = os.path.join(config_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    return config_dir, images_dir


def convert_peft_state_dict(state_dict: dict) -> dict:
    """Convert PEFT format LoRA state dict keys to match our model structure.

    Both checkpoint and model use the same PEFT format:
    model.base_model.model.transformer_blocks.0.attn.to_q.lora_A.default.weight

    No conversion needed - just pass through the keys as-is.
    The '.default' is the adapter name and must be kept.
    """
    # No conversion needed - keys already match
    return state_dict


def load_lora_weights(model: QwenDiffusionWrapper, lora_path: str):
    """Load LoRA weights from checkpoint."""
    if os.path.isdir(lora_path):
        # Directory format - look for generator.safetensors
        safetensors_path = os.path.join(lora_path, "generator.safetensors")
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        else:
            # Try model.pt for legacy format
            pt_path = os.path.join(lora_path, "model.pt")
            if os.path.exists(pt_path):
                checkpoint = torch.load(pt_path, map_location="cpu")
                state_dict = checkpoint.get("generator", checkpoint)
            else:
                raise FileNotFoundError(f"No LoRA weights found in {lora_path}")
    else:
        # Direct file path
        if lora_path.endswith(".safetensors"):
            state_dict = load_file(lora_path)
        else:
            checkpoint = torch.load(lora_path, map_location="cpu")
            state_dict = checkpoint.get("generator", checkpoint)

    # Convert PEFT format keys to our model format
    state_dict = convert_peft_state_dict(state_dict)
    print(f"  Converted {len(state_dict)} LoRA keys from PEFT format")
    print(f"  Sample converted keys: {list(state_dict.keys())[:3]}")

    # Debug: show what LoRA keys the model actually has
    model_lora_keys = [k for k in model.state_dict().keys() if 'lora' in k.lower()]
    print(f"  Model has {len(model_lora_keys)} LoRA keys")
    if model_lora_keys:
        print(f"  Sample model LoRA keys: {model_lora_keys[:3]}")

    # Load with strict=False for LoRA (only loads matching keys)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded LoRA weights from {lora_path}")
    print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if len(unexpected) > 0:
        print(f"  Sample unexpected keys: {unexpected[:5]}")

    return model


def load_models(args, device: torch.device, dtype: torch.dtype):
    """Load all models (DiT, text encoder, VAE)."""
    model_path = args.model_path

    print(f"Loading models from {model_path}...")

    # Load text encoder
    print("  Loading text encoder...")
    text_encoder = QwenTextEncoderWrapper(model_name=model_path, load_weights=True)
    text_encoder.eval().requires_grad_(False)

    # Load VAE
    print("  Loading VAE...")
    vae = QwenVAEWrapper(model_name=model_path, load_weights=True)

    # Load DiT with LoRA if needed
    print("  Loading DiT...")
    enable_lora = args.lora_path is not None
    generator = QwenDiffusionWrapper(
        model_name=model_path,
        mu=args.scheduler_mu,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        enable_lora=enable_lora,
        load_weights=True,
    )

    # Load LoRA weights if specified
    if args.lora_path:
        print(f"  Loading LoRA from {args.lora_path}...")
        generator = load_lora_weights(generator, args.lora_path)

    generator.eval().requires_grad_(False)

    # Move to device
    generator = generator.to(device, dtype=dtype)
    text_encoder = text_encoder.to(device)
    # VAE moves itself when called

    print("Models loaded successfully")

    return generator, text_encoder, vae


def run_inference(
    generator,
    vae,
    pipeline,
    conditional_dict: dict,
    edit_images: List[Image.Image],
    args,
    device: torch.device,
    dtype: torch.dtype,
    custom_timesteps: Optional[torch.Tensor] = None,
    unconditional_dict: Optional[dict] = None,
) -> tuple:
    """
    Run inference and return the output image, inference time, and peak memory.

    Args:
        edit_images: List of PIL images [cloth_image, model_image]. All images are
                    passed to DiT as conditioning. The last image (model_image) is
                    used for noise initialization.
        unconditional_dict: Unconditional text embeddings for CFG (optional).
    """
    # Get output size from the LAST image (model_image) - this is for the denoising target
    target_height, target_width = get_target_size(
        edit_images[-1],
        args.max_pixels,
        args.height_division_factor,
        args.width_division_factor,
    )

    # Preprocess and encode edit images - EACH image resized independently
    # This matches DiffSynth's QwenImageUnit_EditImageEmbedder behavior
    vae_encode_start = time.perf_counter()
    edit_latents = []
    with torch.no_grad():
        for img in edit_images:
            # Each image gets its own target size based on its aspect ratio
            img_height, img_width = get_target_size(
                img,
                args.max_pixels,
                args.height_division_factor,
                args.width_division_factor,
            )
            tensor = preprocess_image(img, img_height, img_width)
            tensor = tensor.to(device, dtype=dtype)
            latent = vae.encode(tensor)
            edit_latents.append(latent)

        # Pass list of latents - DiT will concatenate them in sequence dimension
        # Each latent can have DIFFERENT spatial dimensions
        # For initialization, the last latent (model_image) will be used
        if len(edit_latents) > 1:
            edit_latent = edit_latents  # Pass as list for multi-image
        else:
            edit_latent = edit_latents[0]  # Single tensor for single image

        if device.type == "cuda":
            torch.cuda.synchronize()
        vae_encode_time = time.perf_counter() - vae_encode_start

        # Debug: print latent shapes and VAE timing
        if hasattr(args, 'debug') and args.debug:
            print(f"  [DEBUG] VAE encoding: {vae_encode_time:.3f}s")
            print(f"  [DEBUG] Number of edit images: {len(edit_images)}")
            for i, lat in enumerate(edit_latents):
                print(f"  [DEBUG] edit_latent[{i}] shape: {lat.shape}")
            print(f"  [DEBUG] edit_latent type: {type(edit_latent)}, len: {len(edit_latent) if isinstance(edit_latent, list) else 'N/A'}")

    # Generate noise - use target size from LAST image (model_image)
    # IMPORTANT: DiffSynth generates noise on CPU then moves to GPU
    # This ensures reproducibility across different GPU types
    latent_height = target_height // 8
    latent_width = target_width // 8
    noise = torch.randn(
        1, 16, latent_height, latent_width,
        device="cpu", dtype=dtype,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    ).to(device=device)

    # Set custom timesteps if provided
    if custom_timesteps is not None:
        generator.scheduler.set_timesteps(
            num_inference_steps=len(custom_timesteps),
            device=device,
            custom_timesteps=custom_timesteps,
        )

    # Reset GPU memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    # Run inference
    with torch.inference_mode():
        images = pipeline.inference(
            generator=generator,
            vae=vae,
            noise=noise,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            edit_latent=edit_latent,
            height=target_height,
            width=target_width,
            return_latents=False,
            num_inference_steps=args.num_inference_steps if custom_timesteps is None else None,
            cfg_scale=args.cfg_scale,
            cfg_norm_rescale=args.cfg_norm_rescale,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    inference_time = end_time - start_time

    if device.type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        peak_memory_gb = 0.0

    # Convert to PIL Image
    image_tensor = images[0]  # [3, H, W] in [0, 1]
    image_np = (image_tensor.cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
    image_np = image_np.transpose(1, 2, 0)  # [H, W, 3]
    output_image = Image.fromarray(image_np)

    return output_image, inference_time, peak_memory_gb


def main():
    args = parse_args()

    # Setup device and dtype
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device = torch.device("cuda")
    else:
        dtype = torch.float32
        device = torch.device("cpu")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Generate configuration name and setup directories
    config_name = generate_config_name(args)
    config_dir, images_dir = setup_output_dirs(args.output_dir, config_name)

    print(f"Configuration: {config_name}")
    print(f"Output directory: {config_dir}")
    print(f"Images directory: {images_dir}")

    # Load models
    generator, text_encoder, vae = load_models(args, device, dtype)

    # Create inference pipeline
    config = InferenceConfig(num_inference_steps=args.num_inference_steps)
    pipeline = QwenImageInferencePipeline(config=config, device=device)

    # Parse custom timesteps if provided
    custom_timesteps = None
    if args.custom_timesteps:
        custom_timesteps_list = [float(t.strip()) for t in args.custom_timesteps.split(",")]
        custom_timesteps = torch.tensor(custom_timesteps_list, dtype=dtype, device=device)
        print(f"Using custom timesteps: {custom_timesteps.tolist()}")

    # Load test data
    test_df = pd.read_csv(args.test_path)
    print(f"Loaded {len(test_df)} test samples")

    # Limit samples if specified
    if args.max_samples is not None:
        test_df = test_df.head(args.max_samples)
        print(f"Limited to {len(test_df)} samples")

    # Prompt for editing
    prompt = args.prompt
    print(f"Using prompt: '{prompt}'")

    # CFG settings
    if args.cfg_scale != 1.0:
        print(f"Using CFG scale: {args.cfg_scale}")
        print(f"Negative prompt: '{args.negative_prompt}'")
        if args.cfg_norm_rescale:
            print("CFG norm rescaling: enabled")

    # Determine which columns to use for images
    edit_column = args.edit_image_column
    has_cloth_column = 'cloth_image' in test_df.columns
    print(f"Edit image column: {edit_column}")
    if has_cloth_column:
        print("Multi-image mode: concatenating cloth_image and model_image")

    def encode_prompt_with_images(text_encoder, prompt, edit_images, device, dtype):
        """Encode prompt with edit images for multimodal conditioning."""
        with torch.no_grad():
            # Text encoder now accepts edit_images for multimodal encoding
            # Images are resized to 384x384 area internally
            conditional_dict = text_encoder([prompt], edit_images=edit_images)
            # Move to device and cast embeddings to correct dtype
            conditional_dict = {
                k: v.to(device, dtype=dtype) if (isinstance(v, torch.Tensor) and v.is_floating_point())
                   else v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in conditional_dict.items()
            }
        return conditional_dict

    # Warmup runs
    if args.warmup_runs > 0 and len(test_df) > 0:
        print(f"Running {args.warmup_runs} warmup run(s)...")
        warmup_images = []
        if has_cloth_column:
            warmup_cloth = os.path.join(args.data_dir, test_df.loc[0, 'cloth_image'])
            warmup_images.append(Image.open(warmup_cloth).convert("RGB"))
        warmup_model = os.path.join(args.data_dir, test_df.loc[0, edit_column])
        warmup_images.append(Image.open(warmup_model).convert("RGB"))

        # Encode prompt with warmup images (multimodal)
        warmup_conditional_dict = encode_prompt_with_images(
            text_encoder, prompt, warmup_images, device, dtype
        )

        # Encode negative prompt for CFG if needed
        warmup_unconditional_dict = None
        if args.cfg_scale != 1.0:
            warmup_unconditional_dict = encode_prompt_with_images(
                text_encoder, args.negative_prompt, warmup_images, device, dtype
            )

        for i in range(args.warmup_runs):
            _, warmup_time, warmup_mem = run_inference(
                generator, vae, pipeline, warmup_conditional_dict,
                warmup_images, args, device, dtype, custom_timesteps,
                unconditional_dict=warmup_unconditional_dict
            )
            print(f"  Warmup {i+1}: {warmup_time:.3f}s, peak_memory={warmup_mem:.2f}GB")

    # Benchmark results storage
    results = []

    # Process each sample
    print(f"\nStarting benchmark for {len(test_df)} samples...")
    for idx in tqdm(range(len(test_df)), desc="Processing"):
        edit_images = []

        # Load images - always include cloth_image if available
        if has_cloth_column:
            cloth_image_path = os.path.join(args.data_dir, test_df.loc[idx, 'cloth_image'])
            edit_images.append(Image.open(cloth_image_path).convert("RGB"))

        model_image_path = os.path.join(args.data_dir, test_df.loc[idx, edit_column])
        edit_images.append(Image.open(model_image_path).convert("RGB"))

        # Encode prompt with edit images (multimodal)
        # Text encoder sees images at 384x384 area for understanding
        text_encode_start = time.perf_counter()
        conditional_dict = encode_prompt_with_images(
            text_encoder, prompt, edit_images, device, dtype
        )

        # Encode negative prompt for CFG if needed
        unconditional_dict = None
        if args.cfg_scale != 1.0:
            unconditional_dict = encode_prompt_with_images(
                text_encoder, args.negative_prompt, edit_images, device, dtype
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
        text_encode_time = time.perf_counter() - text_encode_start

        # Run inference
        # DiT sees edit images at 1024x1024 area as latents
        output_image, inference_time, peak_memory_gb = run_inference(
            generator, vae, pipeline, conditional_dict,
            edit_images, args, device, dtype, custom_timesteps,
            unconditional_dict=unconditional_dict
        )

        # Debug: print timing breakdown
        if args.debug:
            print(f"  [DEBUG] Text encoding: {text_encode_time:.3f}s, Inference: {inference_time:.3f}s")

        # Save output image
        output_filename = f"sample_{idx:04d}.png"
        output_path = os.path.join(images_dir, output_filename)
        output_image.save(output_path)

        # Record results
        result_entry = {
            'sample_id': idx,
            'edit_image': test_df.loc[idx, edit_column],
            'edit_image_width': edit_images[-1].size[0],
            'edit_image_height': edit_images[-1].size[1],
            'output_image': output_filename,
            'output_image_width': output_image.size[0],
            'output_image_height': output_image.size[1],
            'inference_time_sec': inference_time,
            'peak_gpu_memory_gb': peak_memory_gb,
        }

        # Add cloth image info if available
        if has_cloth_column:
            result_entry['cloth_image'] = test_df.loc[idx, 'cloth_image']
            result_entry['cloth_image_width'] = edit_images[0].size[0]
            result_entry['cloth_image_height'] = edit_images[0].size[1]

        results.append(result_entry)

        print(f"Sample {idx+1}/{len(test_df)}: {inference_time:.3f}s, memory={peak_memory_gb:.2f}GB - saved to {output_filename}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save detailed results CSV
    results_csv_path = os.path.join(config_dir, "inference_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nDetailed results saved to: {results_csv_path}")

    # Generate and save statistics
    if len(results_df) > 0:
        stats_df = results_df[['inference_time_sec', 'peak_gpu_memory_gb']].describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
        stats_dict = stats_df.to_dict()

        time_stats = {
            'count': stats_dict['inference_time_sec']['count'],
            'mean': stats_dict['inference_time_sec']['mean'],
            'std': stats_dict['inference_time_sec']['std'],
            'min': stats_dict['inference_time_sec']['min'],
            '25%': stats_dict['inference_time_sec']['25%'],
            '50%': stats_dict['inference_time_sec']['50%'],
            '75%': stats_dict['inference_time_sec']['75%'],
            'max': stats_dict['inference_time_sec']['max'],
            'total_time_sec': results_df['inference_time_sec'].sum(),
            'throughput_samples_per_sec': len(results_df) / results_df['inference_time_sec'].sum() if results_df['inference_time_sec'].sum() > 0 else 0,
            'mean_peak_gpu_memory_gb': results_df['peak_gpu_memory_gb'].mean(),
            'max_peak_gpu_memory_gb': results_df['peak_gpu_memory_gb'].max(),
        }

        stats_summary = pd.DataFrame([time_stats])
        stats_csv_path = os.path.join(config_dir, "inference_statistics.csv")
        stats_summary.to_csv(stats_csv_path, index=False)
        print(f"Statistics saved to: {stats_csv_path}")

        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Configuration: {config_name}")
        print(f"Total samples: {len(results_df)}")
        print(f"Total time: {time_stats['total_time_sec']:.3f}s")
        print(f"Throughput: {time_stats['throughput_samples_per_sec']:.4f} samples/sec")
        print("\nInference Time Statistics (seconds):")
        print(f"  Mean:   {time_stats['mean']:.3f}")
        print(f"  Std:    {time_stats['std']:.3f}")
        print(f"  Min:    {time_stats['min']:.3f}")
        print(f"  25%:    {time_stats['25%']:.3f}")
        print(f"  50%:    {time_stats['50%']:.3f}")
        print(f"  75%:    {time_stats['75%']:.3f}")
        print(f"  Max:    {time_stats['max']:.3f}")
        print("\nGPU Memory Statistics:")
        print(f"  Mean Peak: {time_stats['mean_peak_gpu_memory_gb']:.2f} GB")
        print(f"  Max Peak:  {time_stats['max_peak_gpu_memory_gb']:.2f} GB")
        print("="*60)

    # Save configuration
    config_dict = vars(args).copy()
    config_dict['config_name'] = config_name
    config_dict['num_samples'] = len(test_df)
    config_dict['device'] = str(device)
    config_dict['torch_dtype'] = str(dtype)
    if custom_timesteps is not None:
        config_dict['custom_timesteps_parsed'] = custom_timesteps.tolist()
    config_df = pd.DataFrame([config_dict])
    config_csv_path = os.path.join(config_dir, "configuration.csv")
    config_df.to_csv(config_csv_path, index=False)
    print(f"Configuration saved to: {config_csv_path}")


if __name__ == "__main__":
    main()
