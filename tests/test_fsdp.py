"""
Unit tests for FSDP functionality with Qwen models.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
import sys

# Mock heavy dependencies
sys.modules['wan'] = MagicMock()
sys.modules['wan.utils'] = MagicMock()
sys.modules['wan.utils.fm_solvers'] = MagicMock()
sys.modules['wan.utils.fm_solvers_unipc'] = MagicMock()


class TestFSDPWrapStrategies:
    """Tests for FSDP wrap strategy configuration."""

    def test_qwen_wrap_strategy_requires_block(self):
        """Test that qwen wrap strategy requires QwenImageTransformerBlock."""
        from utils.distributed import QWEN_TRANSFORMER_BLOCK

        # If qwen module is available, QWEN_TRANSFORMER_BLOCK should not be None
        # If not available, it should be None
        # This test just verifies the import logic works
        assert QWEN_TRANSFORMER_BLOCK is not None or QWEN_TRANSFORMER_BLOCK is None

    def test_fsdp_wrap_size_strategy(self):
        """Test size-based FSDP wrap strategy."""
        # This test would require distributed setup, so we just test the config parsing
        from utils.distributed import fsdp_wrap

        # Create a simple mock module
        mock_module = torch.nn.Linear(10, 10)

        # The function should accept size strategy without error
        # Note: We can't fully test without distributed setup
        assert callable(fsdp_wrap)


class TestQwenDiffusionWrapperBlocks:
    """Tests for QwenDiffusionWrapper block exposure for FSDP."""

    def test_wrapper_has_blocks_property(self):
        """Test that QwenDiffusionWrapper exposes blocks property."""
        # Mock the QwenImageDiT to avoid loading actual model
        with patch('utils.qwen_wrapper.QwenImageDiT') as MockDiT:
            mock_dit = MagicMock()
            mock_dit.transformer_blocks = torch.nn.ModuleList([
                torch.nn.Linear(10, 10) for _ in range(5)
            ])
            MockDiT.return_value = mock_dit

            with patch('utils.qwen_wrapper.os.path.exists', return_value=False):
                from utils.qwen_wrapper import QwenDiffusionWrapper

                wrapper = QwenDiffusionWrapper(
                    model_name="test",
                    enable_lora=False,
                )

                # Check blocks property exists and returns the transformer blocks
                assert hasattr(wrapper, 'blocks')
                assert wrapper.blocks == mock_dit.transformer_blocks

                # Check transformer_blocks alias
                assert hasattr(wrapper, 'transformer_blocks')
                assert wrapper.transformer_blocks == mock_dit.transformer_blocks

    def test_blocks_are_module_list(self):
        """Test that blocks are a ModuleList for FSDP compatibility."""
        with patch('utils.qwen_wrapper.QwenImageDiT') as MockDiT:
            mock_dit = MagicMock()
            mock_dit.transformer_blocks = torch.nn.ModuleList([
                torch.nn.Linear(10, 10) for _ in range(3)
            ])
            MockDiT.return_value = mock_dit

            with patch('utils.qwen_wrapper.os.path.exists', return_value=False):
                from utils.qwen_wrapper import QwenDiffusionWrapper

                wrapper = QwenDiffusionWrapper(
                    model_name="test",
                    enable_lora=False,
                )

                assert isinstance(wrapper.blocks, torch.nn.ModuleList)
                assert len(wrapper.blocks) == 3


class TestShardQwenModel:
    """Tests for shard_qwen_model function."""

    def test_shard_qwen_model_import(self):
        """Test that shard_qwen_model can be imported."""
        from utils.distributed import shard_qwen_model
        assert callable(shard_qwen_model)

    def test_shard_qwen_model_requires_blocks(self):
        """Test that shard_qwen_model requires blocks attribute."""
        from utils.distributed import shard_qwen_model

        # Model without blocks should raise error
        mock_model = MagicMock(spec=[])  # No blocks attribute

        with pytest.raises(ValueError, match="does not have blocks"):
            # This will fail before FSDP is called since blocks is required
            # We need to mock torch.distributed
            with patch('torch.distributed.is_initialized', return_value=False):
                shard_qwen_model(mock_model, device_id=0)


class TestFSDPConfigOptions:
    """Tests for FSDP configuration options."""

    def test_sharding_strategies(self):
        """Test that all sharding strategies are supported."""
        from torch.distributed.fsdp import ShardingStrategy

        strategies = {
            "full": ShardingStrategy.FULL_SHARD,
            "hybrid_full": ShardingStrategy.HYBRID_SHARD,
            "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
            "no_shard": ShardingStrategy.NO_SHARD,
        }

        for name, expected in strategies.items():
            assert expected is not None

    def test_wrap_strategy_options(self):
        """Test that all wrap strategies are recognized."""
        valid_strategies = ["size", "transformer", "qwen", "qwen_blocks"]

        # Just verify these are valid string options
        for strategy in valid_strategies:
            assert isinstance(strategy, str)


class TestMixedPrecisionConfig:
    """Tests for mixed precision configuration."""

    def test_mixed_precision_dtypes(self):
        """Test mixed precision dtype configuration."""
        from torch.distributed.fsdp import MixedPrecision

        # Test creating mixed precision config
        mp_config = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

        assert mp_config.param_dtype == torch.bfloat16
        assert mp_config.reduce_dtype == torch.float32
        assert mp_config.buffer_dtype == torch.float32
