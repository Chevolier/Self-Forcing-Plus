import gc
import logging
from tqdm import tqdm
from safetensors.torch import save_file as safetensors_save_file
from safetensors.torch import load_file as safetensors_load_file

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset, TextFolderDataset, ImageEditDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD, QwenDMD
import torch
import wandb
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir
        self.checkpoint_dir = getattr(config, "checkpoint_dir", None) or config.logdir

        # Step 2: Initialize the model and optimizer
        self.is_qwen = config.distribution_loss == "qwen_dmd"
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        elif config.distribution_loss == "qwen_dmd":
            # Create FSDP wrapper function for immediate wrapping during model init
            # This allows each model to be FSDP-wrapped right after creation,
            # freeing CPU memory before the next model is loaded
            def qwen_fsdp_wrapper(model, wrap_key, cpu_offload=False):
                wrap_strategy_map = {
                    "generator": config.generator_fsdp_wrap_strategy,
                    "real_score": config.real_score_fsdp_wrap_strategy,
                    "fake_score": config.fake_score_fsdp_wrap_strategy,
                    "text_encoder": config.text_encoder_fsdp_wrap_strategy,
                }
                wrap_strategy = wrap_strategy_map.get(wrap_key, "size")
                if self.is_main_process:
                    print(f"    FSDP wrapping {wrap_key} (strategy={wrap_strategy}, cpu_offload={cpu_offload})...", flush=True)
                return fsdp_wrap(
                    model,
                    sharding_strategy=config.sharding_strategy,
                    mixed_precision=config.mixed_precision,
                    wrap_strategy=wrap_strategy,
                    cpu_offload=cpu_offload
                )

            self.model = QwenDMD(config, device=self.device, fsdp_wrapper=qwen_fsdp_wrapper)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU (after FSDP wrapping for Qwen)
        if not self.is_qwen:
            self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

            # FSDP wrapping with memory cleanup between models to prevent OOM
            self.model.generator = fsdp_wrap(
                self.model.generator,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.generator_fsdp_wrap_strategy
            )
            gc.collect()
            torch.cuda.empty_cache()

            self.model.real_score = fsdp_wrap(
                self.model.real_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.real_score_fsdp_wrap_strategy,
                cpu_offload=getattr(config, "real_score_cpu_offload", False)
            )
            gc.collect()
            torch.cuda.empty_cache()

            self.model.fake_score = fsdp_wrap(
                self.model.fake_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.fake_score_fsdp_wrap_strategy
            )
            gc.collect()
            torch.cuda.empty_cache()

            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
                cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
            )
            gc.collect()
            torch.cuda.empty_cache()
        else:
            # For Qwen, FSDP wrapping already done in QwenDMD init
            # Get state dict from FSDP-wrapped model
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            with FSDP.state_dict_type(self.model.fake_score, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
                self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        if self.config.i2v:
            self.model.image_encoder = fsdp_wrap(
                self.model.image_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.image_encoder_fsdp_wrap_strategy,
                min_num_params=int(5e6),
                cpu_offload=getattr(config, "image_encoder_cpu_offload", False)
            )
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16)

        elif not getattr(config, "no_visualize", True) or getattr(config, "load_raw_video", False):
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        elif self.is_qwen:
            # Qwen image edit dataset with CSV support
            data_max_count = config.get("data_max_count", 100000)
            dataset = ImageEditDataset(
                data_path=config.data_path,
                height=getattr(config, "height", 1024),
                width=getattr(config, "width", 1024),
                max_count=data_max_count,
                metadata_path=getattr(config, "metadata_path", None),
                data_file_keys=getattr(config, "data_file_keys", None),
                edit_image_keys=getattr(config, "edit_image_keys", None),
                label_image_key=getattr(config, "label_image_key", None),
                fixed_prompt=getattr(config, "fixed_prompt", None),
            )
        else:
            if self.config.data_type == "text_folder":
                data_max_count = config.get("data_max_count", 30000)
                dataset = TextFolderDataset(config.data_path, data_max_count)
            elif self.config.data_type == "text_file":
                dataset = TextDataset(config.data_path)
            else:
                raise ValueError("Invalid data type")
            
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        self.ema_weight = config.get("ema_weight", -1.0)
        self.ema_start_step = config.get("ema_start_step", 0)
        self.generator_ema = None
        if (self.ema_weight > 0.0) and (self.step >= self.ema_start_step):
            print(f"Setting up EMA with weight {self.ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "resume_ckpt", False):
            print(f"Resuming training from {config.resume_ckpt}")

            # Try safetensors format first (checkpoint.pt + *.safetensors)
            checkpoint_meta_path = os.path.join(config.resume_ckpt, "checkpoint.pt")
            generator_safetensors_path = os.path.join(config.resume_ckpt, "generator.safetensors")

            if os.path.exists(checkpoint_meta_path) and os.path.exists(generator_safetensors_path):
                print("Loading from safetensors format...")

                # Load checkpoint metadata
                checkpoint_meta = torch.load(checkpoint_meta_path, map_location="cpu")
                weights_type = checkpoint_meta.get("weights_type", "lora")
                print(f"Checkpoint weights type: {weights_type}")

                # Load step
                if "step" in checkpoint_meta:
                    self.step = checkpoint_meta["step"]
                    print(f"Resuming from step {self.step}")
                elif getattr(config, "resume_step", False):
                    self.step = config.resume_step
                    print(f"Resuming from step {self.step} (from config)")

                # Use strict=False for LoRA (partial state), strict=True for full weights
                strict = (weights_type == "full")

                # Load generator weights
                generator_weights = safetensors_load_file(generator_safetensors_path)
                print(f"Loading generator weights ({len(generator_weights)} params)")
                self.model.generator.load_state_dict(generator_weights, strict=strict)
                print("Generator weights loaded successfully")

                # Load critic weights
                critic_safetensors_path = os.path.join(config.resume_ckpt, "critic.safetensors")
                if os.path.exists(critic_safetensors_path):
                    critic_weights = safetensors_load_file(critic_safetensors_path)
                    print(f"Loading critic weights ({len(critic_weights)} params)")
                    self.model.fake_score.load_state_dict(critic_weights, strict=strict)
                    print("Critic weights loaded successfully")

                # Load EMA weights if applicable
                ema_safetensors_path = os.path.join(config.resume_ckpt, "generator_ema.safetensors")
                if os.path.exists(ema_safetensors_path) and self.generator_ema is not None:
                    ema_weights = safetensors_load_file(ema_safetensors_path)
                    print(f"Loading generator EMA weights ({len(ema_weights)} params)")
                    self.generator_ema.load_state_dict(ema_weights)
                    print("Generator EMA weights loaded successfully")

                # Load optimizer states if available
                optimizer_path = os.path.join(config.resume_ckpt, "optimizer.pt")
                if os.path.exists(optimizer_path):
                    print(f"Loading optimizer states from {optimizer_path}")
                    optimizer_state = torch.load(optimizer_path, map_location="cpu")
                    if "generator_optimizer" in optimizer_state:
                        self.generator_optimizer.load_state_dict(optimizer_state["generator_optimizer"])
                        print("Generator optimizer state loaded successfully")
                    if "critic_optimizer" in optimizer_state:
                        self.critic_optimizer.load_state_dict(optimizer_state["critic_optimizer"])
                        print("Critic optimizer state loaded successfully")
                else:
                    print(f"Info: Optimizer states not found at {optimizer_path}, starting fresh optimizers")

            # Try old model.pt format
            elif os.path.exists(os.path.join(config.resume_ckpt, "model.pt")):
                model_path = os.path.join(config.resume_ckpt, "model.pt")
                print(f"Loading from model.pt format: {model_path}")
                checkpoint = torch.load(model_path, map_location="cpu")

                weights_type = checkpoint.get("weights_type", "lora")
                print(f"Checkpoint weights type: {weights_type}")

                if "step" in checkpoint:
                    self.step = checkpoint["step"]
                    print(f"Resuming from step {self.step}")
                elif getattr(config, "resume_step", False):
                    self.step = config.resume_step
                    print(f"Resuming from step {self.step} (from config)")

                strict = (weights_type == "full")

                gen_key = "generator" if "generator" in checkpoint else "generator_lora"
                if gen_key in checkpoint:
                    print(f"Loading generator weights ({len(checkpoint[gen_key])} params)")
                    self.model.generator.load_state_dict(checkpoint[gen_key], strict=strict)
                    print("Generator weights loaded successfully")

                critic_key = "critic" if "critic" in checkpoint else "critic_lora"
                if critic_key in checkpoint:
                    print(f"Loading critic weights ({len(checkpoint[critic_key])} params)")
                    self.model.fake_score.load_state_dict(checkpoint[critic_key], strict=strict)
                    print("Critic weights loaded successfully")

                ema_key = "generator_ema" if "generator_ema" in checkpoint else "generator_ema_lora"
                if ema_key in checkpoint and self.generator_ema is not None:
                    print(f"Loading generator EMA weights ({len(checkpoint[ema_key])} params)")
                    self.generator_ema.load_state_dict(checkpoint[ema_key])
                    print("Generator EMA weights loaded successfully")

                optimizer_path = os.path.join(config.resume_ckpt, "optimizer.pt")
                if os.path.exists(optimizer_path):
                    print(f"Loading optimizer states from {optimizer_path}")
                    optimizer_state = torch.load(optimizer_path, map_location="cpu")
                    if "generator_optimizer" in optimizer_state:
                        self.generator_optimizer.load_state_dict(optimizer_state["generator_optimizer"])
                        print("Generator optimizer state loaded successfully")
                    if "critic_optimizer" in optimizer_state:
                        self.critic_optimizer.load_state_dict(optimizer_state["critic_optimizer"])
                        print("Critic optimizer state loaded successfully")

            else:
                # Fallback to legacy format (separate files)
                print("New checkpoint formats not found, trying legacy format...")

                # Set resume step from config
                if getattr(config, "resume_step", False):
                    self.step = config.resume_step
                    print(f"Resuming from step {self.step}")

                # Load generator_ema checkpoint (if exists)
                generator_ema_path = os.path.join(config.resume_ckpt, "generator_ema.pt")
                if os.path.exists(generator_ema_path):
                    if self.generator_ema is None and self.ema_weight > 0.0:
                        print("Initializing EMA for resume...")
                        generator_state_dict = torch.load(generator_ema_path, map_location="cpu")
                        self.model.generator.load_state_dict(generator_state_dict, strict=False)
                        self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)
                        print("Generator EMA checkpoint loaded successfully")

                # Load generator checkpoint
                generator_path = os.path.join(config.resume_ckpt, "generator.pt")
                if os.path.exists(generator_path):
                    print(f"Loading generator from {generator_path}")
                    generator_state_dict = torch.load(generator_path, map_location="cpu")
                    self.model.generator.load_state_dict(generator_state_dict, strict=False)
                    print("Generator checkpoint loaded successfully")

                # Load critic checkpoint
                critic_path = os.path.join(config.resume_ckpt, "critic.pt")
                if os.path.exists(critic_path):
                    print(f"Loading critic from {critic_path}")
                    critic_state_dict = torch.load(critic_path, map_location="cpu")
                    self.model.fake_score.load_state_dict(critic_state_dict, strict=False)
                    print("Critic checkpoint loaded successfully")
        

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        # if self.step < config.ema_start_step:
        #     self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.save_weights_only = getattr(config, "save_weights_only", True)
        self.use_lora = getattr(config, "lora_rank", 0) > 0
        self.log_every_n_steps = getattr(config, "log_every_n_steps", 10)
        self.total_steps = getattr(config, "total_steps", 100000)
        self.previous_time = None

    def _extract_lora_state_dict(self, full_state_dict):
        """Extract only LoRA weights from a full state dict."""
        lora_state_dict = {}
        for key, value in full_state_dict.items():
            # LoRA weights have "lora_" in their parameter names
            if "lora_" in key.lower():
                lora_state_dict[key] = value
        return lora_state_dict

    def _prepare_state_dict_for_safetensors(self, state_dict):
        """Convert state dict to be compatible with safetensors (contiguous tensors)."""
        return {k: v.contiguous() for k, v in state_dict.items()}

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(self.model.generator)
        critic_state_dict = fsdp_state_dict(self.model.fake_score)

        # Determine what weights to save based on training mode
        if self.use_lora:
            # LoRA training: save only LoRA weights
            generator_weights = self._extract_lora_state_dict(generator_state_dict)
            critic_weights = self._extract_lora_state_dict(critic_state_dict)
            print(f"Generator LoRA params: {len(generator_weights)} / {len(generator_state_dict)} total")
            print(f"Critic LoRA params: {len(critic_weights)} / {len(critic_state_dict)} total")
            weights_type = "lora"
        else:
            # Full finetuning: save full weights
            generator_weights = generator_state_dict
            critic_weights = critic_state_dict
            print(f"Generator params: {len(generator_weights)}")
            print(f"Critic params: {len(critic_weights)}")
            weights_type = "full"

        # Get EMA weights if applicable
        ema_weights = None
        if (self.ema_weight > 0.0) and (self.ema_start_step < self.step) and self.generator_ema is not None:
            ema_state_dict = self.generator_ema.state_dict()
            if self.use_lora:
                ema_weights = self._extract_lora_state_dict(ema_state_dict)
            else:
                ema_weights = ema_state_dict

        if self.is_main_process:
            ckpt_dir = os.path.join(self.checkpoint_dir, f"checkpoint_{self.step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)

            # Save weights as separate safetensors files
            generator_path = os.path.join(ckpt_dir, "generator.safetensors")
            safetensors_save_file(self._prepare_state_dict_for_safetensors(generator_weights), generator_path)
            print(f"Generator {weights_type} weights saved to {generator_path}")

            critic_path = os.path.join(ckpt_dir, "critic.safetensors")
            safetensors_save_file(self._prepare_state_dict_for_safetensors(critic_weights), critic_path)
            print(f"Critic {weights_type} weights saved to {critic_path}")

            if ema_weights is not None:
                ema_path = os.path.join(ckpt_dir, "generator_ema.safetensors")
                safetensors_save_file(self._prepare_state_dict_for_safetensors(ema_weights), ema_path)
                print(f"Generator EMA {weights_type} weights saved to {ema_path}")

            # Save checkpoint metadata as .pt
            checkpoint_meta = {
                "step": self.step,
                "weights_type": weights_type,
            }
            torch.save(checkpoint_meta, os.path.join(ckpt_dir, "checkpoint.pt"))
            print(f"Checkpoint metadata saved to {os.path.join(ckpt_dir, 'checkpoint.pt')}")

            # Save optimizer states only if save_weights_only is False
            if not self.save_weights_only:
                optimizer_state = {
                    "generator_optimizer": self.generator_optimizer.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict(),
                }
                torch.save(optimizer_state, os.path.join(ckpt_dir, "optimizer.pt"))
                print(f"Optimizer states saved to {os.path.join(ckpt_dir, 'optimizer.pt')}")

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]

        # Handle QwenDMD separately
        if self.is_qwen:
            return self._fwdbwd_one_step_qwen(batch, train_generator, text_prompts)

        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

            if self.config.i2v:
                img = batch["img"].to(self.device).squeeze(0)
                clip_fea = self.model.image_encoder(img)
                y = self.model.vae.run_vae_encoder(img)
            else:
                clip_fea = None
                y = None

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
                clip_fea=clip_fea,
                y=y
            )

            torch.cuda.empty_cache()

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None,
            clip_fea=clip_fea,
            y=y
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def _fwdbwd_one_step_qwen(self, batch, train_generator, text_prompts):
        """Forward-backward step for QwenDMD model."""
        batch_size = len(text_prompts)

        # Encode text prompts
        with torch.no_grad():
            conditional_dict = self.model.encode_text(text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.encode_text(
                    [self.config.negative_prompt] * batch_size
                )
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

            # Encode source image if available
            edit_latent = None
            if batch.get("source_image") is not None and batch["source_image"][0] is not None:
                source_images = batch["source_image"].to(device=self.device, dtype=self.dtype)
                edit_latent = self.model.encode_image(source_images)

        # Train generator
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                batch_size=batch_size,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                edit_latent=edit_latent,
            )

            torch.cuda.empty_cache()

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator
            )

            generator_log_dict.update({
                "generator_loss": generator_loss,
                "generator_grad_norm": generator_grad_norm,
            })

            return generator_log_dict

        # Train critic
        critic_loss, critic_log_dict = self.model.critic_loss(
            batch_size=batch_size,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            edit_latent=edit_latent,
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic
        )

        critic_log_dict.update({
            "critic_loss": critic_loss,
            "critic_grad_norm": critic_grad_norm,
        })

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        start_step = self.step

        # Create progress bar (only on main process)
        pbar = None
        if self.is_main_process:
            pbar = tqdm(
                initial=self.step,
                total=self.total_steps,
                desc="Training",
                dynamic_ncols=True
            )

        while self.step < self.total_steps:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.ema_start_step) and \
                    (self.generator_ema is None) and (self.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    gen_loss = generator_log_dict["generator_loss"].mean().item()
                    gen_grad_norm = generator_log_dict["generator_grad_norm"].mean().item()
                    dmd_grad_norm = generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                    wandb_loss_dict.update({
                        "generator_loss": gen_loss,
                        "generator_grad_norm": gen_grad_norm,
                        "dmdtrain_gradient_norm": dmd_grad_norm
                    })

                critic_loss = critic_log_dict["critic_loss"].mean().item()
                critic_grad_norm = critic_log_dict["critic_grad_norm"].mean().item()
                wandb_loss_dict.update({
                    "critic_loss": critic_loss,
                    "critic_grad_norm": critic_grad_norm
                })

                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                    if TRAIN_GENERATOR:
                        pbar.set_postfix({
                            "g_loss": f"{gen_loss:.4f}",
                            "c_loss": f"{critic_loss:.4f}"
                        })
                    else:
                        pbar.set_postfix({"c_loss": f"{critic_loss:.4f}"})

                # Print losses every n steps
                if self.step % self.log_every_n_steps == 0:
                    if TRAIN_GENERATOR:
                        print(f"\n[Step {self.step}] Generator Loss: {gen_loss:.4f}, "
                              f"Critic Loss: {critic_loss:.4f}, "
                              f"Gen Grad: {gen_grad_norm:.4f}, Critic Grad: {critic_grad_norm:.4f}")
                    else:
                        print(f"\n[Step {self.step}] Critic Loss: {critic_loss:.4f}, "
                              f"Critic Grad: {critic_grad_norm:.4f}")

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

        # Close progress bar
        if pbar is not None:
            pbar.close()

        print(f"Training completed at step {self.step}")
