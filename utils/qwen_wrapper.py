"""
Qwen-Image model wrappers for DMD training.
Similar to wan_wrapper.py but for Qwen-Image-Edit models.
"""

import os
import math
from typing import List, Optional, Dict, Any
import torch
from torch import nn

from qwen.models.qwen_image_dit import QwenImageDiT
from qwen.models.qwen_image_text_encoder import QwenImageTextEncoder
from qwen.models.qwen_image_vae import QwenImageVAE
from qwen.utils.flow_match import FlowMatchScheduler


def get_model_path(model_name: str) -> str:
    """Get the model path, handling both relative names and absolute paths."""
    if os.path.isabs(model_name):
        return model_name
    return f"qwen_models/{model_name}"


class QwenFlowMatchScheduler:
    """
    Flow matching scheduler adapted for Qwen-Image models.
    Uses Qwen-specific sigma scheduling.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        mu: float = 0.8,
        shift_terminal: float = 0.02,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.mu = mu
        self.shift_terminal = shift_terminal

        # Pre-compute sigmas for training
        sigmas = torch.linspace(1.0, 0.0, num_train_timesteps + 1)[:-1]
        # Apply exponential shift
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        # Apply terminal shift to prevent collapse
        one_minus_z = 1 - sigmas
        scale_factor = one_minus_z[-1] / (1 - shift_terminal)
        sigmas = 1 - (one_minus_z / scale_factor)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps

        # Compute alphas_cumprod for compatibility with loss functions
        # For flow matching: alpha_t = 1 - sigma_t
        self.alphas_cumprod = (1 - self.sigmas) ** 2

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples using flow matching formulation."""
        # Get sigma for given timestep
        timesteps = timesteps.to(self.timesteps.device)
        timestep_ids = torch.argmin((self.timesteps.unsqueeze(0) - timesteps.unsqueeze(1)).abs(), dim=1)
        sigmas = self.sigmas[timestep_ids].to(original_samples.device, original_samples.dtype)

        # Reshape sigma for broadcasting
        while sigmas.dim() < original_samples.dim():
            sigmas = sigmas.unsqueeze(-1)

        # Flow matching noise addition: x_t = (1 - sigma) * x_0 + sigma * noise
        noisy_samples = (1 - sigmas) * original_samples + sigmas * noise
        return noisy_samples

    def get_sigma(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get sigma values for given timesteps."""
        timesteps = timesteps.to(self.timesteps.device)
        timestep_ids = torch.argmin((self.timesteps.unsqueeze(0) - timesteps.unsqueeze(1)).abs(), dim=1)
        return self.sigmas[timestep_ids]

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        next_timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform one denoising step."""
        sigma = self.get_sigma(timestep).to(sample.device, sample.dtype)
        if next_timestep is not None:
            sigma_next = self.get_sigma(next_timestep).to(sample.device, sample.dtype)
        else:
            sigma_next = torch.zeros_like(sigma)

        while sigma.dim() < sample.dim():
            sigma = sigma.unsqueeze(-1)
            sigma_next = sigma_next.unsqueeze(-1)

        # Flow matching step: x_{t-1} = x_t + v * (sigma_{t-1} - sigma_t)
        prev_sample = sample + model_output * (sigma_next - sigma)
        return prev_sample


class QwenTextEncoderWrapper(nn.Module):
    """Wrapper for Qwen2.5-VL text encoder."""

    def __init__(self, model_name: str = "Qwen-Image-Edit-2509"):
        super().__init__()
        self.model_name = model_name
        model_path = get_model_path(model_name)

        # Initialize text encoder
        self.text_encoder = QwenImageTextEncoder()

        # Load weights if available
        text_encoder_path = os.path.join(model_path, "text_encoder")
        if os.path.exists(text_encoder_path):
            self._load_weights(text_encoder_path)

        self.text_encoder.eval().requires_grad_(False)

    def _load_weights(self, path: str):
        """Load text encoder weights from safetensors or pytorch files."""
        import glob
        from safetensors.torch import load_file

        # Find safetensors files
        safetensor_files = glob.glob(os.path.join(path, "*.safetensors"))
        if safetensor_files:
            state_dict = {}
            for f in safetensor_files:
                state_dict.update(load_file(f))
            self.text_encoder.load_state_dict(state_dict, strict=False)
            print(f"Loaded text encoder from {path}")

    @property
    def device(self):
        return next(self.text_encoder.parameters()).device

    def forward(self, text_prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode text prompts to embeddings."""
        from transformers import Qwen2Tokenizer

        # Get tokenizer
        model_path = get_model_path(self.model_name)
        tokenizer_path = os.path.join(model_path, "tokenizer")
        if os.path.exists(tokenizer_path):
            tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        # Tokenize
        inputs = tokenizer(
            text_prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        return {
            "prompt_embeds": outputs.last_hidden_state,
            "attention_mask": inputs["attention_mask"],
        }


class QwenVAEWrapper(nn.Module):
    """Wrapper for Qwen-Image VAE."""

    def __init__(self, model_name: str = "Qwen-Image-Edit-2509"):
        super().__init__()
        self.model_name = model_name
        model_path = get_model_path(model_name)

        # Initialize VAE
        self.model = QwenImageVAE(
            base_dim=96,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
        )

        # Load weights if available
        vae_path = os.path.join(model_path, "vae")
        if os.path.exists(vae_path):
            self._load_weights(vae_path)

        self.model.eval().requires_grad_(False)
        self.dtype = torch.bfloat16

    def _load_weights(self, path: str):
        """Load VAE weights."""
        import glob
        from safetensors.torch import load_file

        safetensor_files = glob.glob(os.path.join(path, "*.safetensors"))
        if safetensor_files:
            state_dict = {}
            for f in safetensor_files:
                state_dict.update(load_file(f))
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded VAE from {path}")

    def encode(self, pixel: torch.Tensor) -> torch.Tensor:
        """Encode pixel images to latent space.

        Args:
            pixel: [B, C, H, W] pixel images in range [-1, 1]

        Returns:
            latents: [B, 16, H/8, W/8] latent representations
        """
        return self.model.encode(pixel.to(self.dtype))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel images.

        Args:
            latents: [B, 16, H/8, W/8] latent representations

        Returns:
            pixel: [B, C, H, W] pixel images in range [-1, 1]
        """
        return self.model.decode(latents.to(self.dtype))

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        """Alias for encode, matching wan_wrapper interface."""
        return self.encode(pixel)

    def decode_to_pixel(self, latents: torch.Tensor) -> torch.Tensor:
        """Alias for decode, matching wan_wrapper interface."""
        return self.decode(latents)


class QwenDiffusionWrapper(nn.Module):
    """
    Wrapper for Qwen-Image DiT model with LoRA support.
    Similar to WanDiffusionWrapper but for image editing.
    """

    def __init__(
        self,
        model_name: str = "Qwen-Image-Edit-2509",
        mu: float = 0.8,
        lora_rank: int = 64,
        lora_alpha: int = 64,
        enable_lora: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.dim = 3072  # Qwen-Image hidden dim
        model_path = get_model_path(model_name)

        # Initialize DiT model
        self.model = QwenImageDiT(num_layers=60)

        # Load weights if available
        dit_path = os.path.join(model_path, "transformer")
        if os.path.exists(dit_path):
            self._load_weights(dit_path)

        self.model.eval()

        # Initialize scheduler
        self.scheduler = QwenFlowMatchScheduler(num_train_timesteps=1000, mu=mu)

        # LoRA configuration
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_enabled = False

        if enable_lora:
            self._add_lora_layers()

    def _load_weights(self, path: str):
        """Load DiT weights."""
        import glob
        from safetensors.torch import load_file

        safetensor_files = glob.glob(os.path.join(path, "*.safetensors"))
        if safetensor_files:
            state_dict = {}
            for f in sorted(safetensor_files):
                state_dict.update(load_file(f))
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded DiT from {path}")

    def _add_lora_layers(self):
        """Add LoRA adapters to attention layers."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=["to_q", "to_k", "to_v", "to_out"],
                lora_dropout=0.0,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.lora_enabled = True
            print(f"Added LoRA with rank={self.lora_rank}, alpha={self.lora_alpha}")
        except ImportError:
            print("Warning: peft not installed, LoRA disabled")
            self._add_manual_lora()

    def _add_manual_lora(self):
        """Manual LoRA implementation if peft is not available."""
        # Simple LoRA implementation for attention layers
        for name, module in self.model.named_modules():
            if hasattr(module, "to_q"):
                # Add LoRA to query projection
                in_features = module.to_q.in_features
                out_features = module.to_q.out_features
                module.lora_q_down = nn.Linear(in_features, self.lora_rank, bias=False)
                module.lora_q_up = nn.Linear(self.lora_rank, out_features, bias=False)
                nn.init.zeros_(module.lora_q_up.weight)

        self.lora_enabled = True

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.model.gradient_checkpointing = True

    def forward(
        self,
        noisy_latent: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        height: int = 1024,
        width: int = 1024,
        **kwargs,
    ) -> tuple:
        """
        Forward pass through the diffusion model.

        Args:
            noisy_latent: [B, 16, H/8, W/8] noisy latent
            conditional_dict: dict with 'prompt_embeds' and 'attention_mask'
            timestep: [B] timesteps
            height: image height
            width: image width

        Returns:
            (flow_pred, x0_pred): flow prediction and x0 prediction
        """
        prompt_embeds = conditional_dict["prompt_embeds"]
        attention_mask = conditional_dict.get("attention_mask", None)

        if attention_mask is None:
            attention_mask = torch.ones(
                prompt_embeds.shape[:2], device=prompt_embeds.device, dtype=torch.long
            )

        # Forward through DiT
        output = self.model(
            latents=noisy_latent,
            timestep=timestep,
            prompt_emb=prompt_embeds,
            prompt_emb_mask=attention_mask,
            height=height,
            width=width,
        )

        # Output is the flow prediction (velocity)
        flow_pred = output

        # Convert flow to x0 prediction
        x0_pred = self._convert_flow_pred_to_x0(flow_pred, noisy_latent, timestep)

        return flow_pred, x0_pred

    def _convert_flow_pred_to_x0(
        self,
        flow_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Convert flow prediction to x0 prediction.

        For flow matching:
        x_t = (1 - sigma_t) * x_0 + sigma_t * noise
        v = noise - x_0  (flow/velocity)
        Therefore: x_0 = x_t - sigma_t * v
        """
        sigma = self.scheduler.get_sigma(timestep).to(xt.device, xt.dtype)
        while sigma.dim() < xt.dim():
            sigma = sigma.unsqueeze(-1)

        x0 = xt - sigma * flow_pred
        return x0

    def _convert_x0_to_flow_pred(
        self,
        x0: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Convert x0 prediction to flow prediction.

        For flow matching: v = (x_t - x_0) / sigma_t
        """
        sigma = self.scheduler.get_sigma(timestep).to(xt.device, xt.dtype)
        while sigma.dim() < xt.dim():
            sigma = sigma.unsqueeze(-1)

        # Avoid division by zero
        sigma = sigma.clamp(min=1e-6)
        flow_pred = (xt - x0) / sigma
        return flow_pred
