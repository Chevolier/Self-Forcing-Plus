"""
Pytest fixtures for Self-Forcing-Plus tests.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path


@pytest.fixture
def device():
    """Return the device to use for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 2


@pytest.fixture
def latent_shape():
    """Default latent shape [B, C, H, W]."""
    return (2, 16, 128, 128)  # 1024x1024 image -> 128x128 latent


@pytest.fixture
def small_latent_shape():
    """Smaller latent shape for faster tests."""
    return (1, 16, 32, 32)  # 256x256 image


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create prompts directory
        prompts_dir = Path(tmpdir) / "prompts"
        prompts_dir.mkdir()

        # Create some test prompt files
        for i in range(5):
            with open(prompts_dir / f"prompt_{i}.txt", "w") as f:
                f.write(f"Test prompt number {i}")

        yield tmpdir


@pytest.fixture
def temp_csv_data_dir():
    """Create a temporary directory with CSV metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create images directory
        images_dir = tmpdir / "images"
        images_dir.mkdir()

        # Create dummy image files (just empty files for path testing)
        for i in range(3):
            (images_dir / f"image_{i}.png").touch()

        # Create CSV metadata
        csv_content = """image,prompt
images/image_0.png,A beautiful sunset
images/image_1.png,A cute cat
images/image_2.png,A mountain landscape
"""
        with open(tmpdir / "metadata.csv", "w") as f:
            f.write(csv_content)

        yield str(tmpdir)


@pytest.fixture
def mock_config():
    """Create a mock config object for testing."""
    class MockConfig:
        def __init__(self):
            self.num_inference_steps = 8
            self.num_train_timestep = 1000
            self.scheduler_mu = 0.8
            self.height = 1024
            self.width = 1024
            self.guidance_scale = 4.0
            self.real_guidance_scale = 4.0
            self.fake_guidance_scale = 0.0
            self.ts_schedule = True
            self.ts_schedule_max = False
            self.timestep_shift = 1.0
            self.min_score_timestep = 0
            self.denoising_loss_type = "flow"
            self.mixed_precision = True
            self.same_step_across_blocks = True

        def get(self, key, default=None):
            return getattr(self, key, default)

    return MockConfig()
