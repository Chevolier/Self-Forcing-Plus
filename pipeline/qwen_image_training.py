"""
Qwen-Image DMD Training Pipeline.
Simplified from bidirectional_training.py for single image generation.
Uses num_inference_steps parameter like DiffSynth-Studio.
"""

import torch
import torch.distributed as dist
from typing import Dict, Optional, List, Tuple


class QwenImageTrainingPipeline:
    """
    Training pipeline for Qwen-Image DMD with N-step denoising.
    Similar to BidirectionalTrainingPipeline but for single images.
    Uses scheduler.set_timesteps() like DiffSynth-Studio.
    """

    def __init__(
        self,
        config,
        device: torch.device,
    ):
        self.config = config
        self.device = device

        # Number of denoising steps (default 8)
        self.num_inference_steps = getattr(config, "num_inference_steps", 8)

        # Training settings
        self.same_step_exit = getattr(config, "same_step_across_blocks", True)

    def inference_with_trajectory(
        self,
        generator,
        noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        unconditional_dict: Optional[Dict[str, torch.Tensor]] = None,
        edit_latent: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Run denoising trajectory and return prediction with gradient.

        Args:
            generator: QwenDiffusionWrapper model
            noise: [B, 16, H/8, W/8] initial noise
            conditional_dict: text conditioning
            unconditional_dict: unconditional text for CFG (optional)
            edit_latent: [B, 16, H/8, W/8] source image latent for editing
            height: image height
            width: image width

        Returns:
            pred_image: predicted clean latent
            gradient_mask: mask for loss computation (None for images)
            denoised_timestep_from: timestep we exited at (normalized)
            denoised_timestep_to: next timestep (normalized)
        """
        batch_size = noise.shape[0]

        # Set timesteps using scheduler (DiffSynth-Studio style)
        generator.scheduler.set_timesteps(
            num_inference_steps=self.num_inference_steps,
            device=self.device,
        )
        timesteps = generator.scheduler.get_inference_timesteps()
        num_steps = len(timesteps)

        # Initialize noisy input
        if edit_latent is not None:
            # For image editing: start from source + noise
            first_sigma = generator.scheduler.inference_sigmas[0].item()
            noisy_latent = (1 - first_sigma) * edit_latent + first_sigma * noise
        else:
            # For pure generation: start from noise
            noisy_latent = noise

        # Randomly sample exit step (synchronized across ranks)
        exit_step = self._sample_exit_step(num_steps)

        # Denoising loop
        pred_latent = None
        exit_timestep = None
        next_timestep = None

        for step_idx, timestep in enumerate(timesteps):
            timestep_val = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
            timestep_tensor = torch.full(
                (batch_size,), timestep_val, device=self.device, dtype=torch.long
            )

            should_exit = (step_idx == exit_step)

            if not should_exit:
                # Run without gradients
                with torch.no_grad():
                    _, denoised = generator(
                        noisy_latent=noisy_latent,
                        conditional_dict=conditional_dict,
                        timestep=timestep_tensor,
                        height=height,
                        width=width,
                        edit_latents=edit_latent,  # Pass edit latent as conditioning
                    )

                    # Add noise for next step
                    if step_idx + 1 < num_steps:
                        next_t = timesteps[step_idx + 1]
                        next_t_val = next_t.item() if isinstance(next_t, torch.Tensor) else next_t
                        noisy_latent = generator.scheduler.add_noise(
                            denoised,
                            noise,
                            torch.full((batch_size,), next_t_val, device=self.device),
                        )
            else:
                # Run WITH gradients at exit step
                _, pred_latent = generator(
                    noisy_latent=noisy_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep_tensor,
                    height=height,
                    width=width,
                    edit_latents=edit_latent,  # Pass edit latent as conditioning
                )

                exit_timestep = timestep_val
                if step_idx + 1 < num_steps:
                    next_t = timesteps[step_idx + 1]
                    next_timestep = next_t.item() if isinstance(next_t, torch.Tensor) else next_t
                else:
                    next_timestep = 0

                break

        # Compute normalized timestep values for loss scheduling
        denoised_timestep_from = exit_timestep / 1000.0
        denoised_timestep_to = next_timestep / 1000.0

        # No gradient mask needed for single images
        gradient_mask = None

        return pred_latent, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _sample_exit_step(self, num_steps: int) -> int:
        """Sample random exit step, synchronized across distributed ranks."""
        if dist.is_initialized():
            # Generate on rank 0 and broadcast
            if dist.get_rank() == 0:
                exit_step = torch.randint(0, num_steps, (1,), device=self.device)
            else:
                exit_step = torch.zeros(1, dtype=torch.long, device=self.device)
            dist.broadcast(exit_step, src=0)
            return exit_step.item()
        else:
            return torch.randint(0, num_steps, (1,)).item()


class QwenImageInferencePipeline:
    """
    Inference pipeline for Qwen-Image DMD.
    Runs the full denoising trajectory.
    Uses scheduler.set_timesteps() like DiffSynth-Studio.
    """

    def __init__(
        self,
        config,
        device: torch.device,
    ):
        self.config = config
        self.device = device

        # Number of denoising steps (default 8)
        self.num_inference_steps = getattr(config, "num_inference_steps", 8)

    @torch.no_grad()
    def inference(
        self,
        generator,
        vae,
        noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        unconditional_dict: Optional[Dict[str, torch.Tensor]] = None,
        edit_latent: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        return_latents: bool = False,
        num_inference_steps: int = None,
        cfg_scale: float = 1.0,
        cfg_norm_rescale: bool = False,
    ) -> torch.Tensor:
        """
        Run full denoising trajectory for inference.

        Args:
            generator: QwenDiffusionWrapper model
            vae: QwenVAEWrapper for decoding
            noise: [B, 16, H/8, W/8] initial noise
            conditional_dict: text conditioning (positive prompt)
            unconditional_dict: unconditional text conditioning (negative prompt) for CFG
            edit_latent: source image latent for editing
            height: image height
            width: image width
            return_latents: whether to return latents instead of pixels
            num_inference_steps: override number of steps (optional)
            cfg_scale: classifier-free guidance scale (1.0 = no guidance)
            cfg_norm_rescale: whether to apply norm rescaling (matches official diffusers)

        Returns:
            images: [B, C, H, W] generated images or latents
        """
        batch_size = noise.shape[0]
        steps = num_inference_steps or self.num_inference_steps

        # Set timesteps using scheduler with dynamic shift based on image size
        # This matches DiffSynth's behavior: dynamic_shift_len = (height // 16) * (width // 16)
        dynamic_shift_len = (height // 16) * (width // 16)
        generator.scheduler.set_timesteps(
            num_inference_steps=steps,
            device=self.device,
            dynamic_shift_len=dynamic_shift_len,
        )
        timesteps = generator.scheduler.get_inference_timesteps()

        # Handle edit_latent for DiT conditioning
        # IMPORTANT: In DiffSynth's edit_image mode, latents start from PURE NOISE.
        # Edit images provide context through:
        # 1. Text encoder multimodal conditioning (384x384 area)
        # 2. DiT attention conditioning (concatenated in sequence dimension)
        # They do NOT initialize the latents - this is different from img2img!
        noisy_latent = noise  # Always start from pure noise, matching DiffSynth
        context_latents = edit_latent  # Pass edit_latent for DiT conditioning only

        # Denoising loop
        for step_idx, timestep in enumerate(timesteps):
            timestep_val = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
            timestep_tensor = torch.full(
                (batch_size,), timestep_val, device=self.device, dtype=torch.long
            )

            # Get flow prediction (velocity) from model - positive prompt
            flow_pred_posi, _ = generator(
                noisy_latent=noisy_latent,
                conditional_dict=conditional_dict,
                timestep=timestep_tensor,
                height=height,
                width=width,
                edit_latents=context_latents,
            )

            # Apply CFG if scale != 1.0
            if cfg_scale != 1.0 and unconditional_dict is not None:
                # Get flow prediction with negative/unconditional prompt
                flow_pred_nega, _ = generator(
                    noisy_latent=noisy_latent,
                    conditional_dict=unconditional_dict,
                    timestep=timestep_tensor,
                    height=height,
                    width=width,
                    edit_latents=context_latents,
                )
                # CFG formula: pred = pred_nega + cfg_scale * (pred_posi - pred_nega)
                flow_pred = flow_pred_nega + cfg_scale * (flow_pred_posi - flow_pred_nega)

                # Apply norm rescaling to match official diffusers behavior
                # DiffSynth uses dim=-1 (last dimension) for norm calculation
                if cfg_norm_rescale:
                    cond_norm = torch.norm(flow_pred_posi, dim=-1, keepdim=True)
                    noise_norm = torch.norm(flow_pred, dim=-1, keepdim=True)
                    flow_pred = flow_pred * (cond_norm / (noise_norm + 1e-8))
            else:
                flow_pred = flow_pred_posi

            # Use flow matching step: x_{t-1} = x_t + v * (sigma_{t-1} - sigma_t)
            if step_idx + 1 < len(timesteps):
                next_t = timesteps[step_idx + 1]
                next_t_tensor = torch.full((batch_size,), next_t.item() if isinstance(next_t, torch.Tensor) else next_t, device=self.device)
                noisy_latent = generator.scheduler.step(
                    model_output=flow_pred,
                    timestep=timestep_tensor,
                    sample=noisy_latent,
                    next_timestep=next_t_tensor,
                )
            else:
                # Final step: step to sigma=0
                noisy_latent = generator.scheduler.step(
                    model_output=flow_pred,
                    timestep=timestep_tensor,
                    sample=noisy_latent,
                    next_timestep=None,  # sigma=0
                )

        if return_latents:
            return noisy_latent

        # Decode to pixel space
        images = vae.decode(noisy_latent)
        images = (images + 1) / 2  # Scale to [0, 1]
        images = images.clamp(0, 1)

        return images
