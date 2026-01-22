"""
QwenDMD: Distribution Matching Distillation for Qwen-Image-Edit models.
Adapted from dmd.py for single image generation with LoRA training.
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from typing import Tuple, Optional, Dict, List, Union

from pipeline import QwenImageTrainingPipeline
from utils.loss import get_denoising_loss
from utils.qwen_wrapper import (
    QwenDiffusionWrapper,
    QwenTextEncoderWrapper,
    QwenVAEWrapper,
)


class QwenDMD(nn.Module):
    """
    DMD (Distribution Matching Distillation) for Qwen-Image-Edit models.

    This class implements:
    - Generator with LoRA (trainable)
    - Real score (frozen base model)
    - Fake score with LoRA (trainable)

    For single image generation/editing with 8-step denoising.
    """

    def __init__(self, args, device, fsdp_wrapper=None):
        super().__init__()
        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if getattr(args, "mixed_precision", True) else torch.float32
        self.fsdp_wrapper = fsdp_wrapper  # Optional FSDP wrapper for immediate wrapping

        # Model names
        self.generator_name = getattr(args, "generator_name", "Qwen-Image-Edit-2509")
        self.real_model_name = getattr(args, "real_name", self.generator_name)
        self.fake_model_name = getattr(args, "fake_name", self.generator_name)

        # LoRA configuration
        self.lora_rank = getattr(args, "lora_rank", 64)
        self.lora_alpha = getattr(args, "lora_alpha", 64)

        # Initialize models
        self._initialize_models(args, device)

        # Training pipeline (initialized lazily)
        self.inference_pipeline: QwenImageTrainingPipeline = None

        # DMD hyperparameters
        self.num_train_timestep = getattr(args, "num_train_timestep", 1000)
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)

        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = getattr(args, "guidance_scale", 4.0)
            self.fake_guidance_scale = 0.0

        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        # Number of denoising steps (scheduler computes timesteps automatically)
        self.num_inference_steps = getattr(args, "num_inference_steps", 8)

        # Denoising loss function
        self.denoising_loss_func = get_denoising_loss(
            getattr(args, "denoising_loss_type", "flow")
        )()

        # Setup alphas_cumprod for loss computation
        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

        # Image dimensions (None means dynamic based on edit image)
        self.height = getattr(args, "height", None)
        self.width = getattr(args, "width", None)

        # Dynamic sizing parameters (when height/width are None)
        self.max_pixels = getattr(args, "max_pixels", 1024 * 1024)
        self.height_division_factor = getattr(args, "height_division_factor", 16)
        self.width_division_factor = getattr(args, "width_division_factor", 16)

    def _initialize_models(self, args, device):
        """Initialize generator, real_score, fake_score, text_encoder, and VAE.

        Only rank 0 loads weights from disk. Other ranks initialize empty models
        and receive weights via FSDP sync_module_states=True.
        """
        import gc
        mu = getattr(args, "scheduler_mu", 0.8)

        # Check if we're rank 0 - only rank 0 loads weights to save CPU RAM
        is_rank_zero = True
        is_distributed = dist.is_initialized()
        if is_distributed:
            is_rank_zero = dist.get_rank() == 0

        if is_rank_zero:
            print("Rank 0: Loading model weights from disk...", flush=True)
        else:
            print(f"Rank {dist.get_rank()}: Initializing empty models (weights via FSDP sync)...", flush=True)

        # Parse LoRA target modules from config (comma-separated string or list)
        lora_target_modules = getattr(args, "lora_target_modules", None)
        if isinstance(lora_target_modules, str):
            lora_target_modules = [m.strip() for m in lora_target_modules.split(",")]

        # Helper to optionally apply FSDP wrapping immediately after model creation
        def maybe_fsdp_wrap(model, wrap_key, cpu_offload=False):
            if self.fsdp_wrapper is not None:
                wrapped = self.fsdp_wrapper(model, wrap_key, cpu_offload=cpu_offload)
                gc.collect()
                torch.cuda.empty_cache()
                return wrapped
            return model

        # Generator with LoRA (trainable)
        # Add barrier so other ranks wait for rank 0 to finish loading
        if is_rank_zero:
            print("  [1/3] Loading generator...", flush=True)
        self.generator = QwenDiffusionWrapper(
            model_name=self.generator_name,
            mu=mu,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_target_modules=lora_target_modules,
            enable_lora=True,
            load_weights=is_rank_zero,
        )
        # Only LoRA parameters are trainable
        self._freeze_base_enable_lora(self.generator)
        gc.collect()
        torch.cuda.empty_cache()
        if is_rank_zero:
            print("  [1/3] Generator loaded, wrapping with FSDP...", flush=True)
        self.generator = maybe_fsdp_wrap(self.generator, "generator")
        if is_distributed:
            dist.barrier()  # Sync after generator init

        # Real score (frozen, no LoRA)
        if is_rank_zero:
            print("  [2/3] Loading real_score...", flush=True)
        self.real_score = QwenDiffusionWrapper(
            model_name=self.real_model_name,
            mu=mu,
            enable_lora=False,
            load_weights=is_rank_zero,
        )
        self.real_score.requires_grad_(False)
        gc.collect()
        torch.cuda.empty_cache()
        if is_rank_zero:
            print("  [2/3] Real_score loaded, wrapping with FSDP...", flush=True)
        self.real_score = maybe_fsdp_wrap(
            self.real_score, "real_score",
            cpu_offload=getattr(args, "real_score_cpu_offload", False)
        )
        if is_distributed:
            dist.barrier()  # Sync after real_score init

        # Fake score with LoRA (trainable)
        if is_rank_zero:
            print("  [3/3] Loading fake_score...", flush=True)
        self.fake_score = QwenDiffusionWrapper(
            model_name=self.fake_model_name,
            mu=mu,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_target_modules=lora_target_modules,
            enable_lora=True,
            load_weights=is_rank_zero,
        )
        self._freeze_base_enable_lora(self.fake_score)
        gc.collect()
        torch.cuda.empty_cache()
        if is_rank_zero:
            print("  [3/3] Fake_score loaded, wrapping with FSDP...", flush=True)
        self.fake_score = maybe_fsdp_wrap(
            self.fake_score, "fake_score",
            cpu_offload=getattr(args, "fake_score_cpu_offload", False)
        )
        if is_distributed:
            dist.barrier()  # Sync after fake_score init

        # Text encoder (frozen)
        # NOTE: All ranks must load weights because size-based FSDP wrapping
        # is sensitive to model structure differences. Unlike transformer-based
        # wrapping (used for DiT), size-based wrapping can desync if parameters
        # differ between ranks.
        rank_id = dist.get_rank() if is_distributed else 0
        print(f"  [Rank {rank_id}] Loading text_encoder (ALL ranks load weights)...", flush=True)
        self.text_encoder = QwenTextEncoderWrapper(
            model_name=self.generator_name,
            load_weights=True,  # All ranks load (size-based FSDP is structure-sensitive)
        )
        self.text_encoder.requires_grad_(False)
        # Count parameters for sanity check
        num_params = sum(p.numel() for p in self.text_encoder.parameters())
        print(f"  [Rank {rank_id}] Text encoder loaded: {num_params:,} parameters", flush=True)
        if is_distributed:
            dist.barrier()  # Sync after weight loading, before FSDP wrap
        print(f"  [Rank {rank_id}] FSDP wrapping text_encoder...", flush=True)
        self.text_encoder = maybe_fsdp_wrap(
            self.text_encoder, "text_encoder",
            cpu_offload=getattr(args, "text_encoder_cpu_offload", False)
        )
        print(f"  [Rank {rank_id}] Text encoder FSDP wrap complete", flush=True)
        if is_distributed:
            dist.barrier()  # Sync after text_encoder FSDP wrap

        # VAE (frozen) - NOT FSDP wrapped, so ALL ranks must load weights
        print(f"  [Rank {rank_id}] Loading VAE (ALL ranks load weights)...", flush=True)
        self.vae = QwenVAEWrapper(
            model_name=self.generator_name,
            load_weights=True,  # All ranks load (VAE is not FSDP wrapped)
        )
        self.vae.requires_grad_(False)
        vae_params = sum(p.numel() for p in self.vae.parameters())
        print(f"  [Rank {rank_id}] VAE loaded: {vae_params:,} parameters", flush=True)

        # Get scheduler from generator
        self.scheduler = self.generator.scheduler

        # Enable gradient checkpointing if requested
        if getattr(args, "gradient_checkpointing", False):
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        if is_distributed:
            dist.barrier()  # Final sync before returning

        if is_rank_zero:
            print("Rank 0: Model initialization complete.")

    def _freeze_base_enable_lora(self, model: QwenDiffusionWrapper):
        """Freeze base model parameters and enable LoRA parameters."""
        # First freeze everything
        model.requires_grad_(False)

        # Then enable LoRA parameters
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

    def _get_image_size_from_latent(self, latent: torch.Tensor) -> Tuple[int, int]:
        """
        Compute image height and width from latent tensor shape.

        Args:
            latent: [B, C, H, W] latent tensor

        Returns:
            (height, width) tuple for the corresponding image
        """
        # Latent spatial dimensions are 1/8 of image dimensions (VAE downsampling)
        latent_height, latent_width = latent.shape[2], latent.shape[3]
        height = latent_height * 8
        width = latent_width * 8
        return height, width

    def _get_target_size(self, original_height: int, original_width: int) -> Tuple[int, int]:
        """
        Compute target size based on max_pixels constraint.

        If height and width are explicitly set, use them.
        Otherwise, scale based on max_pixels while preserving aspect ratio.

        Args:
            original_height: original image height
            original_width: original image width

        Returns:
            (height, width) tuple for target size
        """
        # If explicit dimensions are set, use them
        if self.height is not None and self.width is not None:
            return self.height, self.width

        height, width = original_height, original_width

        # Scale down if exceeds max_pixels
        if height * width > self.max_pixels:
            scale = (height * width / self.max_pixels) ** 0.5
            height = int(height / scale)
            width = int(width / scale)

        # Round to division factors (VAE compatibility)
        height = height // self.height_division_factor * self.height_division_factor
        width = width // self.width_division_factor * self.width_division_factor

        return height, width

    def _get_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Randomly sample a timestep from [min_timestep, max_timestep].
        For single images, returns shape [batch_size].
        """
        timestep = torch.randint(
            min_timestep,
            max_timestep,
            [batch_size],
            device=self.device,
            dtype=torch.long
        )
        return timestep

    def _compute_kl_grad(
        self,
        noisy_latent: torch.Tensor,
        estimated_clean_latent: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        unconditional_dict: Optional[Dict[str, torch.Tensor]] = None,
        normalization: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL gradient (eq 7 in https://arxiv.org/abs/2311.18828).

        Args:
            noisy_latent: [B, C, H, W] noisy latent
            estimated_clean_latent: [B, C, H, W] estimated clean latent
            timestep: [B] timesteps
            conditional_dict: text conditioning
            unconditional_dict: unconditional text for CFG

        Returns:
            grad: KL gradient tensor
            log_dict: logging info
        """
        # Compute image dimensions from latent shape
        height, width = self._get_image_size_from_latent(noisy_latent)

        # Step 1: Compute fake score prediction
        _, pred_fake_cond = self.fake_score(
            noisy_latent=noisy_latent,
            conditional_dict=conditional_dict,
            timestep=timestep,
            height=height,
            width=width,
        )

        if self.fake_guidance_scale != 0.0 and unconditional_dict is not None:
            _, pred_fake_uncond = self.fake_score(
                noisy_latent=noisy_latent,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                height=height,
                width=width,
            )
            pred_fake = pred_fake_cond + (
                pred_fake_cond - pred_fake_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake = pred_fake_cond

        # Step 2: Compute real score prediction with CFG
        _, pred_real_cond = self.real_score(
            noisy_latent=noisy_latent,
            conditional_dict=conditional_dict,
            timestep=timestep,
            height=height,
            width=width,
        )

        if unconditional_dict is not None:
            _, pred_real_uncond = self.real_score(
                noisy_latent=noisy_latent,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                height=height,
                width=width,
            )
            pred_real = pred_real_cond + (
                pred_real_cond - pred_real_uncond
            ) * self.real_guidance_scale
        else:
            pred_real = pred_real_cond

        # Step 3: Compute DMD gradient (pred_fake - pred_real)
        grad = pred_fake - pred_real

        # Step 4: Gradient normalization (DMD paper eq. 8)
        if normalization:
            p_real = estimated_clean_latent - pred_real
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
            grad = grad / normalizer

        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach(),
        }

    def compute_distribution_matching_loss(
        self,
        latent: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        unconditional_dict: Optional[Dict[str, torch.Tensor]] = None,
        denoised_timestep_from: float = 0,
        denoised_timestep_to: float = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss for image latents.

        Args:
            latent: [B, C, H, W] predicted clean latent from generator
            conditional_dict: text conditioning
            unconditional_dict: unconditional text for CFG
            denoised_timestep_from: timestep we exited at (normalized 0-1)
            denoised_timestep_to: next timestep (normalized 0-1)

        Returns:
            dmd_loss: scalar DMD loss
            log_dict: logging info
        """
        original_latent = latent
        batch_size = latent.shape[0]

        with torch.no_grad():
            # Step 1: Sample timestep based on schedule
            min_timestep = int(denoised_timestep_to * self.num_train_timestep) if self.ts_schedule else self.min_score_timestep
            max_timestep = int(denoised_timestep_from * self.num_train_timestep) if self.ts_schedule_max else self.num_train_timestep

            # Ensure valid range
            min_timestep = max(min_timestep, self.min_score_timestep)
            max_timestep = min(max_timestep, self.num_train_timestep)
            if min_timestep >= max_timestep:
                min_timestep = self.min_score_timestep
                max_timestep = self.num_train_timestep

            timestep = self._get_timestep(min_timestep, max_timestep, batch_size)

            # Apply timestep shift
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * (timestep / 1000) / (
                    1 + (self.timestep_shift - 1) * (timestep / 1000)
                ) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step).long()

            # Add noise to latent
            noise = torch.randn_like(latent)
            noisy_latent = self.scheduler.add_noise(latent, noise, timestep)

            # Step 2: Compute KL gradient
            grad, log_dict = self._compute_kl_grad(
                noisy_latent=noisy_latent,
                estimated_clean_latent=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )

        # Step 3: Compute DMD loss
        dmd_loss = 0.5 * F.mse_loss(
            original_latent.double(),
            (original_latent.double() - grad.double()).detach(),
            reduction="mean"
        )

        return dmd_loss, log_dict

    def _run_generator(
        self,
        batch_size: int,
        conditional_dict: Dict[str, torch.Tensor],
        edit_latent: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float, float]:
        """
        Run generator through denoising trajectory.

        Args:
            batch_size: number of samples to generate
            conditional_dict: text conditioning
            edit_latent: source image latent(s) for editing. Can be:
                        - Single tensor [B, 16, H/8, W/8]
                        - List of tensors for multi-image editing

        Returns:
            pred_latent: generated latent
            gradient_mask: None for images
            denoised_timestep_from: exit timestep (normalized)
            denoised_timestep_to: next timestep (normalized)
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        # Compute dimensions from edit_latent or use defaults
        # For multi-image, use the LAST latent (model_image) for output dimensions
        # This matches benchmark.py behavior
        if edit_latent is not None:
            if isinstance(edit_latent, list):
                # Use last latent for output dimensions (model_image)
                last_latent = edit_latent[-1]
                height, width = self._get_image_size_from_latent(last_latent)
            else:
                height, width = self._get_image_size_from_latent(edit_latent)
        elif self.height is not None and self.width is not None:
            # Use explicitly configured dimensions
            height, width = self.height, self.width
        else:
            # Default to max_pixels-constrained square
            height, width = self._get_target_size(1024, 1024)

        # Generate noise
        latent_height = height // 8
        latent_width = width // 8
        noise = torch.randn(
            batch_size, 16, latent_height, latent_width,
            device=self.device, dtype=self.dtype
        )

        # Run through pipeline
        pred_latent, gradient_mask, denoised_from, denoised_to = self.inference_pipeline.inference_with_trajectory(
            generator=self.generator,
            noise=noise,
            conditional_dict=conditional_dict,
            edit_latent=edit_latent,
            height=height,
            width=width,
        )

        return pred_latent, gradient_mask, denoised_from, denoised_to

    def _initialize_inference_pipeline(self):
        """Initialize the training pipeline for backward simulation."""
        self.inference_pipeline = QwenImageTrainingPipeline(
            config=self.args,
            device=self.device,
        )

    def generator_loss(
        self,
        batch_size: int,
        conditional_dict: Dict[str, torch.Tensor],
        unconditional_dict: Optional[Dict[str, torch.Tensor]] = None,
        edit_latent: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate images from noise and compute DMD loss.

        Args:
            batch_size: number of samples
            conditional_dict: text conditioning
            unconditional_dict: unconditional text for CFG
            edit_latent: source image latent(s) for editing. Can be:
                        - Single tensor [B, 16, H/8, W/8]
                        - List of tensors for multi-image editing

        Returns:
            loss: generator loss
            log_dict: logging info
        """
        # Step 1: Run generator to get fake images
        pred_latent, _, denoised_from, denoised_to = self._run_generator(
            batch_size=batch_size,
            conditional_dict=conditional_dict,
            edit_latent=edit_latent,
        )

        # Step 2: Compute DMD loss
        dmd_loss, log_dict = self.compute_distribution_matching_loss(
            latent=pred_latent,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            denoised_timestep_from=denoised_from,
            denoised_timestep_to=denoised_to,
        )

        del pred_latent

        return dmd_loss, log_dict

    def critic_loss(
        self,
        batch_size: int,
        conditional_dict: Dict[str, torch.Tensor],
        unconditional_dict: Optional[Dict[str, torch.Tensor]] = None,
        edit_latent: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate images and train the fake score (critic) on them.

        Args:
            batch_size: number of samples
            conditional_dict: text conditioning
            unconditional_dict: unconditional text for CFG
            edit_latent: source image latent(s) for editing. Can be:
                        - Single tensor [B, 16, H/8, W/8]
                        - List of tensors for multi-image editing

        Returns:
            loss: critic denoising loss
            log_dict: logging info
        """
        # Step 1: Run generator (no gradients)
        with torch.no_grad():
            generated_latent, _, denoised_from, denoised_to = self._run_generator(
                batch_size=batch_size,
                conditional_dict=conditional_dict,
                edit_latent=edit_latent,
            )

        # Step 2: Sample timestep for critic training
        min_timestep = int(denoised_to * self.num_train_timestep) if self.ts_schedule else self.min_score_timestep
        max_timestep = int(denoised_from * self.num_train_timestep) if self.ts_schedule_max else self.num_train_timestep

        min_timestep = max(min_timestep, self.min_score_timestep)
        max_timestep = min(max_timestep, self.num_train_timestep)
        if min_timestep >= max_timestep:
            min_timestep = self.min_score_timestep
            max_timestep = self.num_train_timestep

        critic_timestep = self._get_timestep(min_timestep, max_timestep, batch_size)

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * (critic_timestep / 1000) / (
                1 + (self.timestep_shift - 1) * (critic_timestep / 1000)
            ) * 1000
        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step).long()

        # Step 3: Add noise to generated latent
        critic_noise = torch.randn_like(generated_latent)
        noisy_generated = self.scheduler.add_noise(
            generated_latent, critic_noise, critic_timestep
        )

        # Compute dimensions from generated latent
        height, width = self._get_image_size_from_latent(generated_latent)

        # Step 4: Get fake score prediction
        _, pred_fake = self.fake_score(
            noisy_latent=noisy_generated,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
            height=height,
            width=width,
        )

        # Step 5: Compute denoising loss
        if self.args.denoising_loss_type == "flow":
            flow_pred = self.generator._convert_x0_to_flow_pred(
                x0=pred_fake,
                xt=noisy_generated,
                timestep=critic_timestep,
            )
            pred_noise = None
        else:
            flow_pred = None
            # Convert x0 to noise prediction
            sigma = self.scheduler.get_sigma(critic_timestep)
            while sigma.dim() < pred_fake.dim():
                sigma = sigma.unsqueeze(-1)
            pred_noise = (noisy_generated - (1 - sigma) * pred_fake) / sigma

        denoising_loss = self.denoising_loss_func(
            x=generated_latent,
            x_pred=pred_fake,
            noise=critic_noise,
            noise_pred=pred_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep,
            flow_pred=flow_pred,
        )

        log_dict = {
            "critic_timestep": critic_timestep.detach(),
        }

        return denoising_loss, log_dict

    def encode_text(
        self,
        prompts: list,
        edit_images: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text prompts to embeddings, optionally with edit images.

        Args:
            prompts: List of text prompts
            edit_images: Optional list of PIL images for multimodal encoding.
                        Images will be resized to 384x384 area for text encoder.

        Returns:
            Dict with 'prompt_embeds' and 'attention_mask'
        """
        return self.text_encoder(prompts, edit_images=edit_images)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space."""
        return self.vae.encode(images)

    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images."""
        return self.vae.decode(latents)

    def get_trainable_parameters(self):
        """Get all trainable parameters (LoRA params from generator and fake_score)."""
        params = []
        for model in [self.generator, self.fake_score]:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    params.append(param)
        return params

    def get_generator_parameters(self):
        """Get generator LoRA parameters."""
        return [p for p in self.generator.parameters() if p.requires_grad]

    def get_critic_parameters(self):
        """Get fake_score LoRA parameters."""
        return [p for p in self.fake_score.parameters() if p.requires_grad]
