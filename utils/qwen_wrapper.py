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
    Compatible with DiffSynth-Studio's FlowMatchScheduler API.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        mu: float = 0.8,
        shift_terminal: float = None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.mu = mu
        self.shift_terminal = shift_terminal

        # Initialize with default 1000 steps for training
        self._compute_sigmas_for_training()

    @staticmethod
    def _calculate_shift_qwen_image(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
        """
        Calculate dynamic shift (mu) based on image sequence length.
        Parameters matched with official diffusers QwenImageEditPlusPipeline.
        """
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def _compute_sigmas_for_training(self):
        """Pre-compute sigmas for full training timesteps."""
        sigmas = torch.linspace(1.0, 0.0, self.num_train_timesteps + 1)[:-1]
        sigmas = math.exp(self.mu) / (math.exp(self.mu) + (1 / sigmas - 1))

        # Apply terminal shift only if specified (default off to match official diffusers)
        if self.shift_terminal is not None:
            one_minus_z = 1 - sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            sigmas = 1 - (one_minus_z / scale_factor)

        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps

        # Compute alphas_cumprod for compatibility with loss functions
        self.alphas_cumprod = (1 - self.sigmas) ** 2

    def set_timesteps(
        self,
        num_inference_steps: int = 8,
        denoising_strength: float = 1.0,
        device: torch.device = None,
        exponential_shift_mu: float = None,
        dynamic_shift_len: int = None,
        custom_timesteps: torch.Tensor = None,
        shift_terminal: float = None,
    ):
        """
        Set timesteps for inference/training with specified number of steps.
        Compatible with DiffSynth-Studio's FlowMatchScheduler API.

        Args:
            num_inference_steps: Number of denoising steps (e.g., 8 for student)
            denoising_strength: Strength of denoising (1.0 = full)
            device: Device to place tensors on
            exponential_shift_mu: Override mu for exponential shift
            dynamic_shift_len: Image sequence length for dynamic mu calculation
            custom_timesteps: Use custom timesteps directly (list, tuple, or tensor)
            shift_terminal: Terminal shift value (None = off, matching official diffusers)
        """
        # Use custom timesteps if provided
        if custom_timesteps is not None:
            if isinstance(custom_timesteps, (list, tuple)):
                custom_timesteps = torch.tensor(custom_timesteps)
            timesteps = custom_timesteps.clone()
            sigmas = timesteps / self.num_train_timesteps

            if device is not None:
                sigmas = sigmas.to(device)
                timesteps = timesteps.to(device)

            self.inference_sigmas = sigmas
            self.inference_timesteps = timesteps
            self.num_inference_steps = len(timesteps)
            return sigmas, timesteps

        sigma_min = 0.0
        sigma_max = 1.0

        # Compute sigmas for the specified number of steps
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]

        # Determine mu value
        if exponential_shift_mu is not None:
            mu = exponential_shift_mu
        elif dynamic_shift_len is not None:
            mu = self._calculate_shift_qwen_image(dynamic_shift_len)
        else:
            mu = self.mu

        # Apply exponential shift (Qwen-specific)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))

        # Apply terminal shift only if specified (default off to match official diffusers)
        if shift_terminal is not None:
            one_minus_z = 1 - sigmas
            scale_factor = one_minus_z[-1] / (1 - shift_terminal)
            sigmas = 1 - (one_minus_z / scale_factor)

        # Compute timesteps
        timesteps = sigmas * self.num_train_timesteps

        if device is not None:
            sigmas = sigmas.to(device)
            timesteps = timesteps.to(device)

        self.inference_sigmas = sigmas
        self.inference_timesteps = timesteps
        self.num_inference_steps = num_inference_steps

        return sigmas, timesteps

    def get_inference_timesteps(self) -> torch.Tensor:
        """Get the timesteps for inference after calling set_timesteps."""
        if hasattr(self, 'inference_timesteps'):
            return self.inference_timesteps
        return self.timesteps

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples using flow matching formulation."""
        # Use inference schedule if available (matches DiffSynth behavior)
        if hasattr(self, 'inference_timesteps') and hasattr(self, 'inference_sigmas'):
            ref_timesteps = self.inference_timesteps
            ref_sigmas = self.inference_sigmas
        else:
            ref_timesteps = self.timesteps
            ref_sigmas = self.sigmas

        # Get sigma for given timestep
        timesteps = timesteps.to(ref_timesteps.device)
        timestep_ids = torch.argmin((ref_timesteps.unsqueeze(0) - timesteps.unsqueeze(1)).abs(), dim=1)
        sigmas = ref_sigmas[timestep_ids].to(original_samples.device, original_samples.dtype)

        # Reshape sigma for broadcasting
        while sigmas.dim() < original_samples.dim():
            sigmas = sigmas.unsqueeze(-1)

        # Flow matching noise addition: x_t = (1 - sigma) * x_0 + sigma * noise
        noisy_samples = (1 - sigmas) * original_samples + sigmas * noise
        return noisy_samples

    def get_sigma(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get sigma values for given timesteps.

        Uses inference schedule if available (set by set_timesteps()),
        otherwise falls back to training schedule.
        """
        # Use inference schedule if available (matches DiffSynth behavior)
        if hasattr(self, 'inference_timesteps') and hasattr(self, 'inference_sigmas'):
            ref_timesteps = self.inference_timesteps
            ref_sigmas = self.inference_sigmas
        else:
            ref_timesteps = self.timesteps
            ref_sigmas = self.sigmas

        timesteps = timesteps.to(ref_timesteps.device)
        timestep_ids = torch.argmin((ref_timesteps.unsqueeze(0) - timesteps.unsqueeze(1)).abs(), dim=1)
        return ref_sigmas[timestep_ids]

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        next_timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform one denoising step."""
        sigma = self.get_sigma(timestep).to(sample.device)
        if next_timestep is not None:
            sigma_next = self.get_sigma(next_timestep).to(sample.device)
        else:
            sigma_next = torch.zeros_like(sigma)

        while sigma.dim() < sample.dim():
            sigma = sigma.unsqueeze(-1)
            sigma_next = sigma_next.unsqueeze(-1)

        # Upcast to float32 to avoid precision issues (matching DiffSynth/official diffusers scheduler)
        original_dtype = sample.dtype
        sample = sample.to(torch.float32)
        # Flow matching step: x_{t-1} = x_t + v * (sigma_{t-1} - sigma_t)
        prev_sample = sample + model_output.to(torch.float32) * (sigma_next - sigma).to(torch.float32)
        # Cast back to original dtype
        prev_sample = prev_sample.to(original_dtype)
        return prev_sample


class QwenTextEncoderWrapper(nn.Module):
    """Wrapper for Qwen2.5-VL text encoder with multimodal support."""

    # Templates matching DiffSynth-Studio
    TEMPLATE_TEXT_ONLY = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    TEMPLATE_EDIT = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
    TEMPLATE_EDIT_MULTI = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    IMG_PROMPT_TEMPLATE = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"

    DROP_IDX_TEXT_ONLY = 34
    DROP_IDX_EDIT = 64

    def __init__(self, model_name: str = "Qwen-Image-Edit-2509", load_weights: bool = True):
        super().__init__()
        self.model_name = model_name
        model_path = get_model_path(model_name)

        # Initialize text encoder
        self.text_encoder = QwenImageTextEncoder()

        # Load weights only if requested (rank 0 loads, others get via FSDP sync)
        if load_weights:
            text_encoder_path = os.path.join(model_path, "text_encoder")
            if os.path.exists(text_encoder_path):
                self._load_weights(text_encoder_path)

        self.text_encoder.eval().requires_grad_(False)

    def _convert_state_dict(self, state_dict: dict) -> dict:
        """Convert state dict keys to match Qwen2_5_VLModel structure.

        This matches DiffSynth's QwenImageTextEncoderStateDictConverter:
        - visual.X → model.visual.X (vision encoder)
        - model.X → model.language_model.X (text encoder)
        """
        state_dict_ = {}
        for k, v in state_dict.items():
            if k.startswith("visual."):
                k = "model." + k
            elif k.startswith("model."):
                k = k.replace("model.", "model.language_model.")
            state_dict_[k] = v
        return state_dict_

    def _load_weights(self, path: str):
        """Load text encoder weights from safetensors or pytorch files."""
        import glob
        from safetensors.torch import load_file
        from tqdm import tqdm

        # Find safetensors files
        safetensor_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if safetensor_files:
            state_dict = {}
            for f in tqdm(safetensor_files, desc="Loading text encoder shards", unit="file"):
                state_dict.update(load_file(f))

            # Debug: show sample keys before conversion
            sample_keys_before = list(state_dict.keys())[:5]
            has_visual = any(k.startswith("visual.") for k in state_dict.keys())
            has_model = any(k.startswith("model.") for k in state_dict.keys())
            print(f"Before conversion: {len(state_dict)} keys, has_visual={has_visual}, has_model={has_model}")
            print(f"  Sample keys: {sample_keys_before}")

            # Convert keys to match model structure
            state_dict = self._convert_state_dict(state_dict)

            # Debug: show sample keys after conversion
            sample_keys_after = list(state_dict.keys())[:5]
            has_model_visual = any(k.startswith("model.visual.") for k in state_dict.keys())
            has_model_lm = any(k.startswith("model.language_model.") for k in state_dict.keys())
            print(f"After conversion: {len(state_dict)} keys, has_model_visual={has_model_visual}, has_model_lm={has_model_lm}")
            print(f"  Sample keys: {sample_keys_after}")

            print(f"Loading state dict ({len(state_dict)} keys)...")
            # Use assign=True to match DiffSynth's loading behavior
            # This directly assigns tensor references instead of copying data
            missing, unexpected = self.text_encoder.load_state_dict(state_dict, strict=False, assign=True)
            print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            if len(missing) > 0:
                # Show sample missing keys to diagnose
                print(f"  Sample missing keys: {missing[:5]}")
            if len(unexpected) > 0:
                print(f"  Sample unexpected keys: {unexpected[:5]}")
            print(f"Loaded text encoder from {path}")

    @property
    def device(self):
        # Use torch.cuda.current_device() for FSDP compatibility
        # FSDP shards parameters so next(parameters()).device may not work
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cpu")

    def _get_tokenizer(self):
        """Get or create tokenizer (cached for efficiency)."""
        if not hasattr(self, '_tokenizer'):
            from transformers import Qwen2Tokenizer
            model_path = get_model_path(self.model_name)
            tokenizer_path = os.path.join(model_path, "tokenizer")
            if os.path.exists(tokenizer_path):
                self._tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
            else:
                self._tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        return self._tokenizer

    def _get_processor(self):
        """Get or create Qwen2VL processor for multimodal encoding (cached)."""
        if not hasattr(self, '_processor'):
            from transformers import Qwen2VLProcessor
            model_path = get_model_path(self.model_name)
            # Processor is in 'processor' subfolder (not 'tokenizer')
            processor_path = os.path.join(model_path, "processor")
            if os.path.exists(processor_path):
                self._processor = Qwen2VLProcessor.from_pretrained(processor_path)
            else:
                self._processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        return self._processor

    def _resize_image_for_encoder(self, image, target_area: int = 384 * 384):
        """Resize image for text encoder (smaller than DiT input)."""
        width, height = image.size
        aspect_ratio = width / height
        calc_width = math.sqrt(target_area * aspect_ratio)
        calc_height = calc_width / aspect_ratio
        calc_width = round(calc_width / 32) * 32
        calc_height = round(calc_height / 32) * 32
        return image.resize((int(calc_width), int(calc_height)))

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        """Extract hidden states based on attention mask."""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def forward(
        self,
        text_prompts: List[str],
        edit_images: Optional[List] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text prompts to embeddings, optionally with edit images.

        Args:
            text_prompts: List of text prompts
            edit_images: Optional list of PIL images for multimodal encoding.
                        Images will be resized to 384x384 area for text encoder.

        Returns:
            Dict with 'prompt_embeds' and 'attention_mask'
        """
        device = self.device

        if edit_images is None:
            # Text-only encoding
            return self._encode_text_only(text_prompts, device)
        elif isinstance(edit_images, list) and len(edit_images) > 1:
            # Multi-image edit encoding
            return self._encode_edit_multi(text_prompts, edit_images, device)
        else:
            # Single image edit encoding
            if isinstance(edit_images, list):
                edit_images = edit_images[0]
            return self._encode_edit_single(text_prompts, edit_images, device)

    def _encode_text_only(self, text_prompts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        """Encode text-only prompts (no images)."""
        tokenizer = self._get_tokenizer()

        # Apply template
        txt = [self.TEMPLATE_TEXT_ONLY.format(p) for p in text_prompts]

        inputs = tokenizer(
            txt,
            padding=True,
            truncation=True,
            max_length=4096 + self.DROP_IDX_TEXT_ONLY,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            hidden_states = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        last_hidden_state = hidden_states[-1]

        # Extract and drop template prefix
        split_hidden_states = self._extract_masked_hidden(last_hidden_state, attention_mask)
        split_hidden_states = [e[self.DROP_IDX_TEXT_ONLY:] for e in split_hidden_states]

        # Pad to same length
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
            for u in split_hidden_states
        ])
        encoder_attention_mask = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
            for u in attn_mask_list
        ])

        return {
            "prompt_embeds": prompt_embeds,
            "attention_mask": encoder_attention_mask,
        }

    def _encode_edit_single(self, text_prompts: List[str], edit_image, device: torch.device) -> Dict[str, torch.Tensor]:
        """Encode text with single edit image (multimodal)."""
        processor = self._get_processor()

        # Resize image for text encoder (384x384 area)
        edit_image_resized = self._resize_image_for_encoder(edit_image)

        # Apply template
        txt = [self.TEMPLATE_EDIT.format(p) for p in text_prompts]

        model_inputs = processor(
            text=txt,
            images=edit_image_resized,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            hidden_states = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
            )

        last_hidden_state = hidden_states[-1]

        # Extract and drop template prefix
        split_hidden_states = self._extract_masked_hidden(last_hidden_state, model_inputs.attention_mask)
        split_hidden_states = [e[self.DROP_IDX_EDIT:] for e in split_hidden_states]

        # Pad to same length
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
            for u in split_hidden_states
        ])
        encoder_attention_mask = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
            for u in attn_mask_list
        ])

        return {
            "prompt_embeds": prompt_embeds,
            "attention_mask": encoder_attention_mask,
        }

    def _encode_edit_multi(self, text_prompts: List[str], edit_images: List, device: torch.device) -> Dict[str, torch.Tensor]:
        """Encode text with multiple edit images (multimodal)."""
        processor = self._get_processor()

        # Resize images for text encoder (384x384 area)
        edit_images_resized = [self._resize_image_for_encoder(img) for img in edit_images]

        # Build image prompt prefix (Picture 1: <image>, Picture 2: <image>, ...)
        base_img_prompt = "".join([
            self.IMG_PROMPT_TEMPLATE.format(i + 1)
            for i in range(len(edit_images))
        ])

        # Apply template
        txt = [self.TEMPLATE_EDIT_MULTI.format(base_img_prompt + p) for p in text_prompts]

        model_inputs = processor(
            text=txt,
            images=edit_images_resized,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            hidden_states = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
            )

        last_hidden_state = hidden_states[-1]

        # Extract and drop template prefix
        split_hidden_states = self._extract_masked_hidden(last_hidden_state, model_inputs.attention_mask)
        split_hidden_states = [e[self.DROP_IDX_EDIT:] for e in split_hidden_states]

        # Pad to same length
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
            for u in split_hidden_states
        ])
        encoder_attention_mask = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
            for u in attn_mask_list
        ])

        return {
            "prompt_embeds": prompt_embeds,
            "attention_mask": encoder_attention_mask,
        }


class QwenVAEWrapper(nn.Module):
    """Wrapper for Qwen-Image VAE."""

    def __init__(self, model_name: str = "Qwen-Image-Edit-2509", load_weights: bool = True):
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

        # Load weights only if requested (rank 0 loads, others get via FSDP sync)
        if load_weights:
            vae_path = os.path.join(model_path, "vae")
            if os.path.exists(vae_path):
                self._load_weights(vae_path)

        self.dtype = torch.bfloat16
        # Convert model to bfloat16 to match input dtype
        self.model = self.model.to(self.dtype)
        self.model.eval().requires_grad_(False)
        self._current_device = None

    def _load_weights(self, path: str):
        """Load VAE weights."""
        import glob
        from safetensors.torch import load_file
        from tqdm import tqdm

        safetensor_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if safetensor_files:
            state_dict = {}
            for f in tqdm(safetensor_files, desc="Loading VAE shards", unit="file"):
                state_dict.update(load_file(f))
            print(f"Loading state dict ({len(state_dict)} keys)...")
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded VAE from {path}")

    def _ensure_device(self, target_device: torch.device):
        """Move model to target device if needed."""
        if self._current_device != target_device:
            self.model = self.model.to(target_device)
            self._current_device = target_device

    def encode(self, pixel: torch.Tensor) -> torch.Tensor:
        """Encode pixel images to latent space.

        Args:
            pixel: [B, C, H, W] pixel images in range [-1, 1]

        Returns:
            latents: [B, 16, H/8, W/8] latent representations
        """
        # Ensure VAE is on the same device as input
        self._ensure_device(pixel.device)
        return self.model.encode(pixel.to(self.dtype))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel images.

        Args:
            latents: [B, 16, H/8, W/8] latent representations

        Returns:
            pixel: [B, C, H, W] pixel images in range [-1, 1]
        """
        # Ensure VAE is on the same device as input
        self._ensure_device(latents.device)
        return self.model.decode(latents.to(self.dtype))

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        """Alias for encode, matching wan_wrapper interface."""
        return self.encode(pixel)

    def decode_to_pixel(self, latents: torch.Tensor) -> torch.Tensor:
        """Alias for decode, matching wan_wrapper interface."""
        return self.decode(latents)


class QwenDiffusionWrapper(nn.Module):
    """
    Wrapper for Qwen-Image DiT model with LoRA and FSDP support.
    Similar to WanDiffusionWrapper but for image editing.
    """

    def __init__(
        self,
        model_name: str = "Qwen-Image-Edit-2509",
        mu: float = 0.8,
        lora_rank: int = 64,
        lora_alpha: int = 64,
        lora_target_modules: List[str] = None,
        enable_lora: bool = False,
        load_weights: bool = True,  # Set False on non-rank-0 for FSDP sync
    ):
        super().__init__()
        self.model_name = model_name
        self.dim = 3072  # Qwen-Image hidden dim
        model_path = get_model_path(model_name)

        # Initialize DiT model
        self.model = QwenImageDiT(num_layers=60)

        # Load weights only if requested (rank 0 loads, others get via FSDP sync)
        if load_weights:
            dit_path = os.path.join(model_path, "transformer")
            if os.path.exists(dit_path):
                self._load_weights(dit_path)

        self.model.eval()

        # Initialize scheduler
        self.scheduler = QwenFlowMatchScheduler(num_train_timesteps=1000, mu=mu)

        # LoRA configuration
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.lora_enabled = False

        if enable_lora:
            self._add_lora_layers(target_modules=lora_target_modules)

    @property
    def blocks(self):
        """Expose transformer blocks for FSDP wrapping (compatible with Wan API)."""
        return self.model.transformer_blocks

    @property
    def transformer_blocks(self):
        """Alias for blocks, matching the internal DiT attribute name."""
        return self.model.transformer_blocks

    def _load_weights(self, path: str):
        """Load DiT weights."""
        import glob
        from safetensors.torch import load_file
        from tqdm import tqdm

        safetensor_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if safetensor_files:
            state_dict = {}
            for f in tqdm(safetensor_files, desc="Loading DiT shards", unit="file"):
                state_dict.update(load_file(f))
            print(f"Loading state dict ({len(state_dict)} keys)...")
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded DiT from {path}")

    def _add_lora_layers(self, target_modules: List[str] = None):
        """Add LoRA adapters to attention layers.

        Args:
            target_modules: List of module names to apply LoRA to.
                For Qwen DiT, valid targets include:
                - "to_q", "to_k", "to_v" - query/key/value projections
                - "add_q_proj", "add_k_proj", "add_v_proj" - text cross-attention
                - "to_out.0" - output projection (inside Sequential)
                - "to_add_out" - text output projection
                - "img_mod.1", "txt_mod.1" - modulation layers (inside Sequential)
        """
        try:
            from peft import LoraConfig, get_peft_model

            # Default target modules - only Linear layers, not Sequential containers
            if target_modules is None:
                target_modules = [
                    "to_q", "to_k", "to_v",  # Image attention
                    "add_q_proj", "add_k_proj", "add_v_proj",  # Text cross-attention
                    "to_out.0",  # Output projection (Linear inside Sequential)
                    "to_add_out",  # Text output projection
                ]

            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.0,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.lora_enabled = True
            print(f"Added LoRA with rank={self.lora_rank}, alpha={self.lora_alpha}")
            print(f"LoRA target modules: {target_modules}")
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
        # When PEFT/LoRA is enabled, self.model is PeftModel, need to get base model
        if hasattr(self.model, 'get_base_model'):
            # PEFT wrapped model
            base_model = self.model.get_base_model()
            base_model.gradient_checkpointing = True
            print(f"Enabled gradient checkpointing on base model (PEFT wrapped)")
        else:
            # Direct model
            self.model.gradient_checkpointing = True
            print(f"Enabled gradient checkpointing on model")

    def forward(
        self,
        noisy_latent: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        height: int = 1024,
        width: int = 1024,
        edit_latents: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple:
        """
        Forward pass through the diffusion model.

        Args:
            noisy_latent: [B, 16, H/8, W/8] noisy latent
            conditional_dict: dict with 'prompt_embeds' and 'attention_mask'
            timestep: [B] timesteps in [0, 1000] range
            height: image height
            width: image width
            edit_latents: Optional [B, 16, H/8, W/8] or list of edit image latents for conditioning

        Returns:
            (flow_pred, x0_pred): flow prediction and x0 prediction
        """
        prompt_embeds = conditional_dict["prompt_embeds"]
        attention_mask = conditional_dict.get("attention_mask", None)

        if attention_mask is None:
            attention_mask = torch.ones(
                prompt_embeds.shape[:2], device=prompt_embeds.device, dtype=torch.long
            )

        # Normalize timestep to [0, 1] range (model expects this, has internal scale=1000)
        # DiffSynth does: timestep = timestep / 1000
        timestep_normalized = timestep / 1000.0

        # Forward through DiT
        output = self.model(
            latents=noisy_latent,
            timestep=timestep_normalized,
            prompt_emb=prompt_embeds,
            prompt_emb_mask=attention_mask,
            height=height,
            width=width,
            edit_latents=edit_latents,
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
