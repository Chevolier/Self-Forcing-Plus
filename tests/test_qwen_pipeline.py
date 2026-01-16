"""
Unit tests for QwenImageTrainingPipeline and QwenImageInferencePipeline.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from pipeline.qwen_image_training import (
    QwenImageTrainingPipeline,
    QwenImageInferencePipeline,
)
from utils.qwen_wrapper import QwenFlowMatchScheduler


class MockGenerator:
    """Mock generator for testing pipelines."""

    def __init__(self):
        self.scheduler = QwenFlowMatchScheduler()

    def __call__(self, noisy_latent, conditional_dict, timestep, height, width):
        """Mock forward pass - returns dummy flow and x0 predictions."""
        batch_size = noisy_latent.shape[0]
        # Return (flow_pred, x0_pred)
        flow_pred = torch.randn_like(noisy_latent)
        x0_pred = torch.randn_like(noisy_latent)
        return flow_pred, x0_pred


class MockVAE:
    """Mock VAE for testing inference pipeline."""

    def decode(self, latents):
        """Mock decode - returns dummy pixel images."""
        batch_size, channels, h, w = latents.shape
        # Return [B, 3, H*8, W*8] pixel images in [-1, 1]
        return torch.randn(batch_size, 3, h * 8, w * 8)


class TestQwenImageTrainingPipeline:
    """Tests for the training pipeline."""

    def test_init_default(self, mock_config):
        """Test pipeline initialization with default config."""
        pipeline = QwenImageTrainingPipeline(
            config=mock_config,
            device=torch.device("cpu"),
        )

        assert pipeline.num_inference_steps == 8
        assert pipeline.same_step_exit == True

    def test_init_custom_steps(self):
        """Test pipeline with custom number of steps."""
        class Config:
            num_inference_steps = 16
            same_step_across_blocks = False

        pipeline = QwenImageTrainingPipeline(
            config=Config(),
            device=torch.device("cpu"),
        )

        assert pipeline.num_inference_steps == 16
        assert pipeline.same_step_exit == False

    def test_inference_with_trajectory_shape(self, mock_config):
        """Test that inference_with_trajectory returns correct shapes."""
        pipeline = QwenImageTrainingPipeline(
            config=mock_config,
            device=torch.device("cpu"),
        )

        generator = MockGenerator()
        noise = torch.randn(2, 16, 64, 64)
        conditional_dict = {"prompt_embeds": torch.randn(2, 77, 768)}

        # Patch distributed functions for non-distributed testing
        with patch('torch.distributed.is_initialized', return_value=False):
            pred_latent, gradient_mask, ts_from, ts_to = pipeline.inference_with_trajectory(
                generator=generator,
                noise=noise,
                conditional_dict=conditional_dict,
                height=512,
                width=512,
            )

        # Check output shapes
        assert pred_latent.shape == noise.shape
        assert gradient_mask is None  # Should be None for images
        assert isinstance(ts_from, float)
        assert isinstance(ts_to, float)
        assert 0.0 <= ts_from <= 1.0
        assert 0.0 <= ts_to <= 1.0

    def test_inference_with_trajectory_with_edit_latent(self, mock_config):
        """Test inference with edit latent (image editing mode)."""
        pipeline = QwenImageTrainingPipeline(
            config=mock_config,
            device=torch.device("cpu"),
        )

        generator = MockGenerator()
        noise = torch.randn(1, 16, 64, 64)
        edit_latent = torch.randn(1, 16, 64, 64)
        conditional_dict = {"prompt_embeds": torch.randn(1, 77, 768)}

        with patch('torch.distributed.is_initialized', return_value=False):
            pred_latent, _, _, _ = pipeline.inference_with_trajectory(
                generator=generator,
                noise=noise,
                conditional_dict=conditional_dict,
                edit_latent=edit_latent,
                height=512,
                width=512,
            )

        assert pred_latent.shape == noise.shape

    def test_sample_exit_step_non_distributed(self, mock_config):
        """Test exit step sampling in non-distributed setting."""
        pipeline = QwenImageTrainingPipeline(
            config=mock_config,
            device=torch.device("cpu"),
        )

        with patch('torch.distributed.is_initialized', return_value=False):
            # Sample many times to check distribution
            exit_steps = [pipeline._sample_exit_step(8) for _ in range(100)]

        # All should be in valid range
        assert all(0 <= s < 8 for s in exit_steps)
        # Should have some variety (not all same)
        assert len(set(exit_steps)) > 1

    def test_timesteps_computed_by_scheduler(self, mock_config):
        """Test that timesteps are computed by scheduler, not hardcoded."""
        pipeline = QwenImageTrainingPipeline(
            config=mock_config,
            device=torch.device("cpu"),
        )

        generator = MockGenerator()
        noise = torch.randn(1, 16, 32, 32)
        conditional_dict = {"prompt_embeds": torch.randn(1, 77, 768)}

        with patch('torch.distributed.is_initialized', return_value=False):
            pipeline.inference_with_trajectory(
                generator=generator,
                noise=noise,
                conditional_dict=conditional_dict,
                height=256,
                width=256,
            )

        # Check that scheduler has inference_timesteps set
        assert hasattr(generator.scheduler, 'inference_timesteps')
        assert len(generator.scheduler.inference_timesteps) == 8


class TestQwenImageInferencePipeline:
    """Tests for the inference pipeline."""

    def test_init_default(self, mock_config):
        """Test inference pipeline initialization."""
        pipeline = QwenImageInferencePipeline(
            config=mock_config,
            device=torch.device("cpu"),
        )

        assert pipeline.num_inference_steps == 8

    def test_inference_returns_images(self, mock_config):
        """Test that inference returns pixel images."""
        pipeline = QwenImageInferencePipeline(
            config=mock_config,
            device=torch.device("cpu"),
        )

        generator = MockGenerator()
        vae = MockVAE()
        noise = torch.randn(2, 16, 32, 32)
        conditional_dict = {"prompt_embeds": torch.randn(2, 77, 768)}

        images = pipeline.inference(
            generator=generator,
            vae=vae,
            noise=noise,
            conditional_dict=conditional_dict,
            height=256,
            width=256,
        )

        # Should return pixel images [B, C, H, W]
        assert images.shape[0] == 2
        assert images.shape[1] == 3
        # Should be normalized to [0, 1]
        assert images.min() >= 0.0
        assert images.max() <= 1.0

    def test_inference_returns_latents(self, mock_config):
        """Test inference with return_latents=True."""
        pipeline = QwenImageInferencePipeline(
            config=mock_config,
            device=torch.device("cpu"),
        )

        generator = MockGenerator()
        vae = MockVAE()
        noise = torch.randn(1, 16, 32, 32)
        conditional_dict = {"prompt_embeds": torch.randn(1, 77, 768)}

        latents = pipeline.inference(
            generator=generator,
            vae=vae,
            noise=noise,
            conditional_dict=conditional_dict,
            height=256,
            width=256,
            return_latents=True,
        )

        # Should return latents with same shape as noise
        assert latents.shape == noise.shape

    def test_inference_custom_steps(self, mock_config):
        """Test inference with custom number of steps."""
        pipeline = QwenImageInferencePipeline(
            config=mock_config,
            device=torch.device("cpu"),
        )

        generator = MockGenerator()
        vae = MockVAE()
        noise = torch.randn(1, 16, 32, 32)
        conditional_dict = {"prompt_embeds": torch.randn(1, 77, 768)}

        # Override num_inference_steps
        images = pipeline.inference(
            generator=generator,
            vae=vae,
            noise=noise,
            conditional_dict=conditional_dict,
            height=256,
            width=256,
            num_inference_steps=4,
        )

        # Check scheduler was called with 4 steps
        assert generator.scheduler.num_inference_steps == 4

    def test_inference_with_edit_latent(self, mock_config):
        """Test inference in image editing mode."""
        pipeline = QwenImageInferencePipeline(
            config=mock_config,
            device=torch.device("cpu"),
        )

        generator = MockGenerator()
        vae = MockVAE()
        noise = torch.randn(1, 16, 32, 32)
        edit_latent = torch.randn(1, 16, 32, 32)
        conditional_dict = {"prompt_embeds": torch.randn(1, 77, 768)}

        images = pipeline.inference(
            generator=generator,
            vae=vae,
            noise=noise,
            conditional_dict=conditional_dict,
            edit_latent=edit_latent,
            height=256,
            width=256,
        )

        assert images.shape[0] == 1
        assert images.shape[1] == 3


class TestPipelineIntegration:
    """Integration tests for pipelines."""

    def test_training_and_inference_consistency(self, mock_config):
        """Test that training and inference use same scheduler logic."""
        train_pipeline = QwenImageTrainingPipeline(
            config=mock_config,
            device=torch.device("cpu"),
        )
        infer_pipeline = QwenImageInferencePipeline(
            config=mock_config,
            device=torch.device("cpu"),
        )

        generator = MockGenerator()

        # Both should use same num_inference_steps
        assert train_pipeline.num_inference_steps == infer_pipeline.num_inference_steps

        # Set timesteps through training pipeline
        with patch('torch.distributed.is_initialized', return_value=False):
            train_pipeline.inference_with_trajectory(
                generator=generator,
                noise=torch.randn(1, 16, 32, 32),
                conditional_dict={"prompt_embeds": torch.randn(1, 77, 768)},
                height=256,
                width=256,
            )

        train_timesteps = generator.scheduler.inference_timesteps.clone()

        # Set timesteps through inference pipeline
        infer_pipeline.inference(
            generator=generator,
            vae=MockVAE(),
            noise=torch.randn(1, 16, 32, 32),
            conditional_dict={"prompt_embeds": torch.randn(1, 77, 768)},
            height=256,
            width=256,
            return_latents=True,
        )

        infer_timesteps = generator.scheduler.inference_timesteps

        # Should produce same timesteps
        assert torch.allclose(train_timesteps, infer_timesteps)

    def test_different_batch_sizes(self, mock_config):
        """Test pipelines with different batch sizes."""
        pipeline = QwenImageTrainingPipeline(
            config=mock_config,
            device=torch.device("cpu"),
        )
        generator = MockGenerator()

        with patch('torch.distributed.is_initialized', return_value=False):
            for batch_size in [1, 2, 4]:
                noise = torch.randn(batch_size, 16, 32, 32)
                conditional_dict = {"prompt_embeds": torch.randn(batch_size, 77, 768)}

                pred, _, _, _ = pipeline.inference_with_trajectory(
                    generator=generator,
                    noise=noise,
                    conditional_dict=conditional_dict,
                    height=256,
                    width=256,
                )

                assert pred.shape[0] == batch_size
