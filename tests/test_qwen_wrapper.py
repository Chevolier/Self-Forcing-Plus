"""
Unit tests for Qwen model wrappers.
These tests don't load actual models - they test the wrapper logic.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
import os

from utils.qwen_wrapper import (
    get_model_path,
    QwenFlowMatchScheduler,
)


class TestGetModelPath:
    """Tests for get_model_path utility function."""

    def test_absolute_path_unchanged(self):
        """Test that absolute paths are returned unchanged."""
        abs_path = "/home/user/models/Qwen-Image-Edit"
        result = get_model_path(abs_path)
        assert result == abs_path

    def test_relative_path_prefixed(self):
        """Test that relative paths get qwen_models/ prefix."""
        rel_path = "Qwen-Image-Edit-2509"
        result = get_model_path(rel_path)
        assert result == "qwen_models/Qwen-Image-Edit-2509"

    def test_windows_absolute_path(self):
        """Test Windows-style absolute paths (if on Windows)."""
        # os.path.isabs behavior depends on platform
        if os.name == 'nt':
            win_path = "C:\\models\\Qwen"
            result = get_model_path(win_path)
            assert result == win_path


class TestQwenFlowMatchSchedulerConversion:
    """Tests for flow/x0 conversion functions in the wrapper."""

    def test_scheduler_sigma_consistency(self):
        """Test that sigmas are consistent with timesteps."""
        scheduler = QwenFlowMatchScheduler()

        # Get sigmas for some timesteps
        timesteps = torch.tensor([100, 500, 900])
        sigmas = scheduler.get_sigma(timesteps)

        # Higher timestep should have higher sigma
        assert sigmas[0] < sigmas[1] < sigmas[2]

    def test_scheduler_add_noise_reversible(self):
        """Test that noise addition follows flow matching formula."""
        scheduler = QwenFlowMatchScheduler()

        original = torch.randn(1, 16, 32, 32)
        noise = torch.randn_like(original)
        timestep = torch.tensor([500])

        noisy = scheduler.add_noise(original, noise, timestep)

        # Get sigma for this timestep
        sigma = scheduler.get_sigma(timestep)

        # Verify: noisy = (1 - sigma) * original + sigma * noise
        expected = (1 - sigma) * original + sigma * noise
        assert torch.allclose(noisy, expected, atol=1e-5)

    def test_scheduler_step_direction(self):
        """Test that step moves towards cleaner sample."""
        scheduler = QwenFlowMatchScheduler()

        # Start with noisy sample
        noisy = torch.randn(1, 16, 32, 32)
        # Assume velocity points towards clean sample
        velocity = torch.randn_like(noisy) * 0.1

        current_t = torch.tensor([500])
        next_t = torch.tensor([250])

        stepped = scheduler.step(velocity, current_t, noisy, next_t)

        # Step should produce different sample
        assert not torch.allclose(stepped, noisy)
        assert stepped.shape == noisy.shape


class TestQwenWrapperIntegration:
    """Integration tests for wrapper components."""

    def test_scheduler_full_trajectory(self):
        """Test running a full denoising trajectory."""
        scheduler = QwenFlowMatchScheduler()
        scheduler.set_timesteps(num_inference_steps=8)

        timesteps = scheduler.get_inference_timesteps()
        latent = torch.randn(1, 16, 64, 64)
        original_noise = torch.randn_like(latent)

        # Simulate full denoising
        current = latent
        for i, t in enumerate(timesteps):
            # In real scenario, model predicts velocity
            mock_velocity = torch.randn_like(current) * 0.1

            if i + 1 < len(timesteps):
                next_t = timesteps[i + 1]
                current = scheduler.step(
                    mock_velocity,
                    torch.tensor([t.item()]),
                    current,
                    torch.tensor([next_t.item()]),
                )
            else:
                current = scheduler.step(
                    mock_velocity,
                    torch.tensor([t.item()]),
                    current,
                    None,
                )

        # Should complete without errors
        assert current.shape == latent.shape

    def test_scheduler_different_mu_values(self):
        """Test scheduler behavior with different mu values."""
        mus = [0.5, 0.8, 1.0, 1.5]
        schedulers = [QwenFlowMatchScheduler(mu=mu) for mu in mus]

        # All should have valid sigmas
        for scheduler in schedulers:
            assert scheduler.sigmas.min() >= 0.0
            assert scheduler.sigmas.max() <= 1.0

        # Different mu should give different distributions
        for i in range(len(schedulers) - 1):
            assert not torch.allclose(
                schedulers[i].sigmas,
                schedulers[i + 1].sigmas,
                atol=0.01
            )

    def test_scheduler_set_timesteps_idempotent(self):
        """Test that calling set_timesteps multiple times is idempotent."""
        scheduler = QwenFlowMatchScheduler()

        # Call multiple times
        sigmas1, ts1 = scheduler.set_timesteps(8)
        sigmas2, ts2 = scheduler.set_timesteps(8)
        sigmas3, ts3 = scheduler.set_timesteps(8)

        # All should be equal
        assert torch.allclose(sigmas1, sigmas2)
        assert torch.allclose(sigmas2, sigmas3)
        assert torch.allclose(ts1, ts2)
        assert torch.allclose(ts2, ts3)

    def test_scheduler_training_sigmas_preserved(self):
        """Test that training sigmas are preserved after set_timesteps."""
        scheduler = QwenFlowMatchScheduler()

        # Store original training sigmas
        original_sigmas = scheduler.sigmas.clone()
        original_timesteps = scheduler.timesteps.clone()

        # Call set_timesteps
        scheduler.set_timesteps(8)

        # Original training sigmas should still be accessible
        assert torch.allclose(scheduler.sigmas, original_sigmas)
        assert torch.allclose(scheduler.timesteps, original_timesteps)

        # Inference sigmas should be different
        assert len(scheduler.inference_sigmas) == 8
        assert len(scheduler.sigmas) == 1000


class TestQwenWrapperEdgeCases:
    """Edge case tests for wrappers."""

    def test_scheduler_single_step(self):
        """Test scheduler with single inference step."""
        scheduler = QwenFlowMatchScheduler()
        sigmas, timesteps = scheduler.set_timesteps(num_inference_steps=1)

        assert len(sigmas) == 1
        assert len(timesteps) == 1

    def test_scheduler_many_steps(self):
        """Test scheduler with many inference steps."""
        scheduler = QwenFlowMatchScheduler()
        sigmas, timesteps = scheduler.set_timesteps(num_inference_steps=100)

        assert len(sigmas) == 100
        assert len(timesteps) == 100
        # Should still be monotonically decreasing
        assert (sigmas[:-1] >= sigmas[1:]).all()

    def test_scheduler_batch_timesteps(self):
        """Test scheduler with batch of different timesteps."""
        scheduler = QwenFlowMatchScheduler()

        # Different timesteps for different batch items
        timesteps = torch.tensor([100, 300, 500, 700, 900])
        sigmas = scheduler.get_sigma(timesteps)

        assert sigmas.shape == (5,)
        # Each should be valid
        assert (sigmas >= 0).all()
        assert (sigmas <= 1).all()

    def test_add_noise_batch_consistency(self):
        """Test that add_noise handles batches correctly."""
        scheduler = QwenFlowMatchScheduler()

        batch_size = 4
        original = torch.randn(batch_size, 16, 32, 32)
        noise = torch.randn_like(original)

        # Same timestep for all
        timesteps = torch.full((batch_size,), 500)
        noisy = scheduler.add_noise(original, noise, timesteps)

        assert noisy.shape == original.shape

        # Different timesteps
        timesteps = torch.tensor([100, 300, 500, 700])
        noisy = scheduler.add_noise(original, noise, timesteps)

        assert noisy.shape == original.shape
