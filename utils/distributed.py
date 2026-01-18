from datetime import timedelta
from functools import partial
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy, lambda_auto_wrap_policy


# Import Qwen transformer blocks for FSDP wrapping
try:
    from qwen.models.qwen_image_dit import QwenImageTransformerBlock
    QWEN_TRANSFORMER_BLOCK = QwenImageTransformerBlock
except ImportError:
    QWEN_TRANSFORMER_BLOCK = None


def shard_qwen_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
    """
    Shard a Qwen model using FSDP with transformer block-level wrapping.
    Similar to wan/distributed/fsdp.py shard_model but for Qwen models.

    Args:
        model: QwenDiffusionWrapper or model with `blocks` attribute
        device_id: GPU device ID
        param_dtype: Parameter dtype for mixed precision
        reduce_dtype: Reduction dtype for gradients
        buffer_dtype: Buffer dtype
        process_group: Process group for distributed training
        sharding_strategy: FSDP sharding strategy
        sync_module_states: Whether to sync module states across ranks

    Returns:
        FSDP-wrapped model
    """
    # Get the transformer blocks for wrapping
    if hasattr(model, 'blocks'):
        blocks = model.blocks
    elif hasattr(model, 'transformer_blocks'):
        blocks = model.transformer_blocks
    elif hasattr(model, 'model') and hasattr(model.model, 'transformer_blocks'):
        blocks = model.model.transformer_blocks
    else:
        raise ValueError("Model does not have blocks/transformer_blocks attribute for FSDP wrapping")

    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in blocks
        ),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype
        ),
        device_id=device_id,
        use_orig_params=True,
        sync_module_states=sync_module_states
    )
    return model


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, ignored_modules=None, cpu_offload=False):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    elif wrap_strategy == "qwen":
        # Use Qwen transformer blocks for wrapping via transformer_auto_wrap_policy
        if QWEN_TRANSFORMER_BLOCK is None:
            raise ImportError("Qwen transformer block not available. Install qwen package.")
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={QWEN_TRANSFORMER_BLOCK}
        )
    elif wrap_strategy == "qwen_blocks":
        # Lambda-based wrapping using model.blocks attribute (like Wan's approach)
        # This requires the model to have a 'blocks' or 'transformer_blocks' attribute
        # Get blocks from module for the lambda
        if hasattr(module, 'blocks'):
            blocks = list(module.blocks)
        elif hasattr(module, 'transformer_blocks'):
            blocks = list(module.transformer_blocks)
        elif hasattr(module, 'model') and hasattr(module.model, 'transformer_blocks'):
            blocks = list(module.model.transformer_blocks)
        else:
            raise ValueError("Model does not have blocks attribute for qwen_blocks wrap strategy")
        auto_wrap_policy = partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in blocks
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        ignored_modules=ignored_modules,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=True  # Broadcast weights from rank 0 to all other ranks
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=30))
    torch.cuda.set_device(local_rank)


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False, offload_to_cpu=True, rank0_only=True):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False, offload_to_cpu=True, rank0_only=True):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        return self.shadow            # picklable

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, fsdp_module):
        # load EMA weights into an (unwrapped) copy of the generator
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for n, p in fsdp_module.module.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.dtype, device=p.device))
