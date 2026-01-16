"""
Qwen-Image DMD Training Pipeline.
Simplified from bidirectional_training.py for single image generation.
"""

import torch
import torch.distributed as dist
from typing import Dict, Optional, List, Tuple


class QwenImageTrainingPipeline:
    """
    Training pipeline for Qwen-Image DMD with 8-step denoising.
    Similar to BidirectionalTrainingPipeline but for single images.
    """

    def __init__(
        self,
        config,
        device: torch.device,
    ):
        self.config = config
        self.device = device

        # 8-step denoising schedule
        self.denoising_step_list = getattr(
            config,
            "denoising_step_list",
            [1000, 875, 750, 625, 500, 375, 250, 125]
        )
        self.num_steps = len(self.denoising_step_list)

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

        # Initialize noisy input
        if edit_latent is not None:
            # For image editing: start from source + noise
            # Use partial noising based on first timestep
            first_sigma = generator.scheduler.get_sigma(
                torch.tensor([self.denoising_step_list[0]], device=self.device)
            ).item()
            noisy_latent = (1 - first_sigma) * edit_latent + first_sigma * noise
        else:
            # For pure generation: start from noise
            noisy_latent = noise

        # Randomly sample exit step (synchronized across ranks)
        if self.same_step_exit:
            exit_step = self._sample_exit_step(batch_size)
        else:
            exit_step = self._sample_exit_step(batch_size)

        # Denoising loop
        pred_latent = None
        exit_timestep = None
        next_timestep = None

        for step_idx, timestep in enumerate(self.denoising_step_list):
            timestep_tensor = torch.full(
                (batch_size,), timestep, device=self.device, dtype=torch.long
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
                    )

                    # Add noise for next step
                    if step_idx + 1 < len(self.denoising_step_list):
                        next_t = self.denoising_step_list[step_idx + 1]
                        noisy_latent = generator.scheduler.add_noise(
                            denoised,
                            noise,
                            torch.full((batch_size,), next_t, device=self.device),
                        )
            else:
                # Run WITH gradients at exit step
                _, pred_latent = generator(
                    noisy_latent=noisy_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep_tensor,
                    height=height,
                    width=width,
                )

                exit_timestep = timestep
                if step_idx + 1 < len(self.denoising_step_list):
                    next_timestep = self.denoising_step_list[step_idx + 1]
                else:
                    next_timestep = 0

                break

        # Compute normalized timestep values for loss scheduling
        denoised_timestep_from = exit_timestep / 1000.0
        denoised_timestep_to = next_timestep / 1000.0

        # No gradient mask needed for single images
        gradient_mask = None

        return pred_latent, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _sample_exit_step(self, batch_size: int) -> int:
        """Sample random exit step, synchronized across distributed ranks."""
        if dist.is_initialized():
            # Generate on rank 0 and broadcast
            if dist.get_rank() == 0:
                exit_step = torch.randint(0, self.num_steps, (1,), device=self.device)
            else:
                exit_step = torch.zeros(1, dtype=torch.long, device=self.device)
            dist.broadcast(exit_step, src=0)
            return exit_step.item()
        else:
            return torch.randint(0, self.num_steps, (1,)).item()


class QwenImageInferencePipeline:
    """
    Inference pipeline for Qwen-Image DMD.
    Runs the full denoising trajectory.
    """

    def __init__(
        self,
        config,
        device: torch.device,
    ):
        self.config = config
        self.device = device

        # 8-step denoising schedule
        self.denoising_step_list = getattr(
            config,
            "denoising_step_list",
            [1000, 875, 750, 625, 500, 375, 250, 125]
        )

    @torch.no_grad()
    def inference(
        self,
        generator,
        vae,
        noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        edit_latent: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        return_latents: bool = False,
    ) -> torch.Tensor:
        """
        Run full denoising trajectory for inference.

        Args:
            generator: QwenDiffusionWrapper model
            vae: QwenVAEWrapper for decoding
            noise: [B, 16, H/8, W/8] initial noise
            conditional_dict: text conditioning
            edit_latent: source image latent for editing
            height: image height
            width: image width
            return_latents: whether to return latents instead of pixels

        Returns:
            images: [B, C, H, W] generated images or latents
        """
        batch_size = noise.shape[0]

        # Initialize
        if edit_latent is not None:
            first_sigma = generator.scheduler.get_sigma(
                torch.tensor([self.denoising_step_list[0]], device=self.device)
            ).item()
            noisy_latent = (1 - first_sigma) * edit_latent + first_sigma * noise
        else:
            noisy_latent = noise

        # Denoising loop
        for step_idx, timestep in enumerate(self.denoising_step_list):
            timestep_tensor = torch.full(
                (batch_size,), timestep, device=self.device, dtype=torch.long
            )

            _, denoised = generator(
                noisy_latent=noisy_latent,
                conditional_dict=conditional_dict,
                timestep=timestep_tensor,
                height=height,
                width=width,
            )

            # Prepare for next step
            if step_idx + 1 < len(self.denoising_step_list):
                next_t = self.denoising_step_list[step_idx + 1]
                noisy_latent = generator.scheduler.add_noise(
                    denoised,
                    noise,
                    torch.full((batch_size,), next_t, device=self.device),
                )
            else:
                noisy_latent = denoised

        if return_latents:
            return noisy_latent

        # Decode to pixel space
        images = vae.decode(noisy_latent)
        images = (images + 1) / 2  # Scale to [0, 1]
        images = images.clamp(0, 1)

        return images
