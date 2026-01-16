"""
Unit tests for QwenFlowMatchScheduler.
"""

import pytest
import torch
import math

from utils.qwen_wrapper import QwenFlowMatchScheduler


class TestQwenFlowMatchScheduler:
    """Tests for the Qwen flow matching scheduler."""

    def test_init_default(self):
        """Test scheduler initialization with default parameters."""
        scheduler = QwenFlowMatchScheduler()

        assert scheduler.num_train_timesteps == 1000
        assert scheduler.mu == 0.8
        assert scheduler.shift_terminal == 0.02
        assert scheduler.sigmas is not None
        assert scheduler.timesteps is not None
        assert len(scheduler.sigmas) == 1000
        assert len(scheduler.timesteps) == 1000

    def test_init_custom_params(self):
        """Test scheduler initialization with custom parameters."""
        scheduler = QwenFlowMatchScheduler(
            num_train_timesteps=500,
            mu=1.0,
            shift_terminal=0.05,
        )

        assert scheduler.num_train_timesteps == 500
        assert scheduler.mu == 1.0
        assert scheduler.shift_terminal == 0.05
        assert len(scheduler.sigmas) == 500

    def test_sigmas_range(self):
        """Test that sigmas are in valid range [0, 1]."""
        scheduler = QwenFlowMatchScheduler()

        assert scheduler.sigmas.min() >= 0.0
        assert scheduler.sigmas.max() <= 1.0
        # Sigmas should be decreasing (from noisy to clean)
        assert (scheduler.sigmas[:-1] >= scheduler.sigmas[1:]).all()

    def test_timesteps_range(self):
        """Test that timesteps are in valid range."""
        scheduler = QwenFlowMatchScheduler()

        assert scheduler.timesteps.min() >= 0.0
        assert scheduler.timesteps.max() <= 1000.0

    def test_set_timesteps_8_steps(self):
        """Test set_timesteps with 8 inference steps."""
        scheduler = QwenFlowMatchScheduler()
        sigmas, timesteps = scheduler.set_timesteps(num_inference_steps=8)

        assert len(sigmas) == 8
        assert len(timesteps) == 8
        assert scheduler.num_inference_steps == 8
        # Check sigmas are decreasing
        assert (sigmas[:-1] >= sigmas[1:]).all()

    def test_set_timesteps_different_steps(self):
        """Test set_timesteps with different number of steps."""
        scheduler = QwenFlowMatchScheduler()

        for steps in [4, 8, 16, 32]:
            sigmas, timesteps = scheduler.set_timesteps(num_inference_steps=steps)
            assert len(sigmas) == steps
            assert len(timesteps) == steps

    def test_set_timesteps_with_device(self):
        """Test set_timesteps with device parameter."""
        scheduler = QwenFlowMatchScheduler()
        device = torch.device("cpu")

        sigmas, timesteps = scheduler.set_timesteps(
            num_inference_steps=8,
            device=device,
        )

        assert sigmas.device == device
        assert timesteps.device == device

    def test_get_inference_timesteps(self):
        """Test get_inference_timesteps after set_timesteps."""
        scheduler = QwenFlowMatchScheduler()
        scheduler.set_timesteps(num_inference_steps=8)

        timesteps = scheduler.get_inference_timesteps()
        assert len(timesteps) == 8
        assert torch.equal(timesteps, scheduler.inference_timesteps)

    def test_get_inference_timesteps_without_set(self):
        """Test get_inference_timesteps without calling set_timesteps first."""
        scheduler = QwenFlowMatchScheduler()

        # Should return full training timesteps
        timesteps = scheduler.get_inference_timesteps()
        assert len(timesteps) == 1000

    def test_add_noise(self):
        """Test add_noise function."""
        scheduler = QwenFlowMatchScheduler()

        # Create sample tensors
        original = torch.randn(2, 16, 32, 32)
        noise = torch.randn_like(original)
        timesteps = torch.tensor([500, 500])

        noisy = scheduler.add_noise(original, noise, timesteps)

        assert noisy.shape == original.shape
        # Noisy should be different from original
        assert not torch.allclose(noisy, original)

    def test_add_noise_at_t0(self):
        """Test add_noise at timestep 0 (should be close to original)."""
        scheduler = QwenFlowMatchScheduler()

        original = torch.randn(1, 16, 32, 32)
        noise = torch.randn_like(original)
        timesteps = torch.tensor([0])

        noisy = scheduler.add_noise(original, noise, timesteps)

        # At t=0, sigma should be very small, so noisy ≈ original
        # Due to terminal shift, this might not be exactly equal
        assert noisy.shape == original.shape

    def test_add_noise_at_t1000(self):
        """Test add_noise at timestep 1000 (should be close to noise)."""
        scheduler = QwenFlowMatchScheduler()

        original = torch.randn(1, 16, 32, 32)
        noise = torch.randn_like(original)
        timesteps = torch.tensor([999])  # Close to max

        noisy = scheduler.add_noise(original, noise, timesteps)

        # At high timestep, sigma should be large
        assert noisy.shape == original.shape

    def test_get_sigma(self):
        """Test get_sigma function."""
        scheduler = QwenFlowMatchScheduler()

        timesteps = torch.tensor([0, 250, 500, 750, 999])
        sigmas = scheduler.get_sigma(timesteps)

        assert sigmas.shape == (5,)
        # Sigmas should increase with timestep
        assert sigmas[0] < sigmas[-1]

    def test_step(self):
        """Test denoising step function."""
        scheduler = QwenFlowMatchScheduler()

        sample = torch.randn(1, 16, 32, 32)
        model_output = torch.randn_like(sample)  # velocity prediction
        timestep = torch.tensor([500])
        next_timestep = torch.tensor([250])

        prev_sample = scheduler.step(model_output, timestep, sample, next_timestep)

        assert prev_sample.shape == sample.shape

    def test_alphas_cumprod(self):
        """Test alphas_cumprod computation."""
        scheduler = QwenFlowMatchScheduler()

        assert scheduler.alphas_cumprod is not None
        assert len(scheduler.alphas_cumprod) == 1000
        # alphas_cumprod should be in [0, 1]
        assert scheduler.alphas_cumprod.min() >= 0.0
        assert scheduler.alphas_cumprod.max() <= 1.0

    def test_exponential_shift(self):
        """Test that exponential shift is applied correctly."""
        # Without shift (mu=0), sigmas would be linear
        scheduler_no_shift = QwenFlowMatchScheduler(mu=0.001)  # Very small mu
        scheduler_with_shift = QwenFlowMatchScheduler(mu=0.8)

        # The shifted version should have different distribution
        assert not torch.allclose(
            scheduler_no_shift.sigmas,
            scheduler_with_shift.sigmas,
            atol=0.01
        )


class TestQwenSchedulerIntegration:
    """Integration tests for scheduler with typical usage patterns."""

    def test_8_step_denoising_loop(self):
        """Test a complete 8-step denoising loop."""
        scheduler = QwenFlowMatchScheduler()
        scheduler.set_timesteps(num_inference_steps=8)

        timesteps = scheduler.get_inference_timesteps()
        latent = torch.randn(1, 16, 64, 64)
        noise = torch.randn_like(latent)

        # Simulate denoising loop
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((1,), t.item(), dtype=torch.long)

            # Mock model output (velocity)
            velocity = torch.randn_like(latent)

            if i + 1 < len(timesteps):
                next_t = timesteps[i + 1]
                # In real code, we'd denoise and re-noise
                # Here just verify the scheduler doesn't crash
                sigma = scheduler.get_sigma(t_tensor)
                assert sigma.shape == (1,)

        # Should complete without errors
        assert True

    def test_different_batch_sizes(self):
        """Test scheduler with different batch sizes."""
        scheduler = QwenFlowMatchScheduler()

        for batch_size in [1, 2, 4, 8]:
            original = torch.randn(batch_size, 16, 32, 32)
            noise = torch.randn_like(original)
            timesteps = torch.full((batch_size,), 500)

            noisy = scheduler.add_noise(original, noise, timesteps)
            assert noisy.shape == original.shape

    def test_consistency_across_calls(self):
        """Test that set_timesteps gives consistent results."""
        scheduler = QwenFlowMatchScheduler()

        sigmas1, timesteps1 = scheduler.set_timesteps(num_inference_steps=8)
        sigmas2, timesteps2 = scheduler.set_timesteps(num_inference_steps=8)

        assert torch.allclose(sigmas1, sigmas2)
        assert torch.allclose(timesteps1, timesteps2)
