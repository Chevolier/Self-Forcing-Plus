"""
Unit tests for ImageEditDataset.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np

from utils.dataset import ImageEditDataset


class TestImageEditDatasetPromptLoading:
    """Tests for loading prompts from different formats."""

    def test_load_from_text_folder(self, temp_data_dir):
        """Test loading prompts from a folder of text files."""
        # temp_data_dir has prompts/ folder with .txt files
        prompts_dir = Path(temp_data_dir) / "prompts"

        # Create the dataset pointing to prompts dir
        # Since there's no images/ dir, it should fall back to loading just prompts
        dataset = ImageEditDataset(
            data_path=str(prompts_dir),
            height=256,
            width=256,
        )

        assert len(dataset) == 5
        sample = dataset[0]
        assert "prompts" in sample
        assert "Test prompt number" in sample["prompts"]

    def test_load_with_fixed_prompt(self, temp_data_dir):
        """Test that fixed_prompt overrides file prompts."""
        prompts_dir = Path(temp_data_dir) / "prompts"

        dataset = ImageEditDataset(
            data_path=str(prompts_dir),
            height=256,
            width=256,
            fixed_prompt="Fixed prompt for all samples",
        )

        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample["prompts"] == "Fixed prompt for all samples"

    def test_load_from_prompts_txt(self):
        """Test loading from prompts.txt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create prompts.txt
            prompts = ["Prompt one", "Prompt two", "Prompt three"]
            with open(Path(tmpdir) / "prompts.txt", "w") as f:
                f.write("\n".join(prompts))

            dataset = ImageEditDataset(
                data_path=tmpdir,
                height=256,
                width=256,
            )

            assert len(dataset) == 3
            assert dataset[0]["prompts"] == "Prompt one"
            assert dataset[1]["prompts"] == "Prompt two"
            assert dataset[2]["prompts"] == "Prompt three"


class TestImageEditDatasetCSVLoading:
    """Tests for loading data from CSV metadata."""

    def test_load_from_csv_basic(self):
        """Test basic CSV loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create CSV with just prompts
            csv_content = """prompt
First prompt
Second prompt
Third prompt
"""
            with open(tmpdir / "metadata.csv", "w") as f:
                f.write(csv_content)

            dataset = ImageEditDataset(
                data_path=str(tmpdir),
                height=256,
                width=256,
            )

            assert len(dataset) == 3
            assert dataset[0]["prompts"] == "First prompt"

    def test_load_from_csv_with_images(self):
        """Test CSV loading with image paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create images directory and dummy images
            images_dir = tmpdir / "images"
            images_dir.mkdir()

            for i in range(2):
                img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
                img.save(images_dir / f"img_{i}.png")

            # Create CSV
            csv_content = """source_image,prompt
images/img_0.png,Edit image 0
images/img_1.png,Edit image 1
"""
            with open(tmpdir / "train_data.csv", "w") as f:
                f.write(csv_content)

            dataset = ImageEditDataset(
                data_path=str(tmpdir),
                height=64,
                width=64,
                edit_image_keys="source_image",
            )

            assert len(dataset) == 2
            sample = dataset[0]
            assert sample["prompts"] == "Edit image 0"
            # edit_images should be loaded
            assert sample["edit_images"] is not None
            assert sample["edit_images"].shape == (1, 3, 64, 64)

    def test_load_from_csv_with_multiple_edit_images(self):
        """Test CSV loading with multiple edit image columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create images
            images_dir = tmpdir / "images"
            images_dir.mkdir()

            for name in ["cloth_0.png", "model_0.png", "label_0.png"]:
                img = Image.new("RGB", (64, 64), color="blue")
                img.save(images_dir / name)

            # Create CSV (DiffSynth-Studio style)
            csv_content = """cloth_image,model_image,label_image
images/cloth_0.png,images/model_0.png,images/label_0.png
"""
            with open(tmpdir / "train_data.csv", "w") as f:
                f.write(csv_content)

            dataset = ImageEditDataset(
                data_path=str(tmpdir),
                height=64,
                width=64,
                data_file_keys="cloth_image,model_image,label_image",
                edit_image_keys="cloth_image,model_image",
                label_image_key="label_image",
                fixed_prompt="Test edit",
            )

            assert len(dataset) == 1
            sample = dataset[0]
            assert sample["prompts"] == "Test edit"
            # Should have 2 edit images
            assert sample["edit_images"] is not None
            assert sample["edit_images"].shape == (2, 3, 64, 64)
            # Should have label image
            assert sample["label_image"] is not None
            assert sample["label_image"].shape == (3, 64, 64)

    def test_load_from_csv_with_metadata_path(self):
        """Test loading with explicit metadata_path parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create CSV in a subdirectory
            meta_dir = tmpdir / "meta"
            meta_dir.mkdir()

            csv_content = """prompt
Custom metadata prompt
"""
            csv_path = meta_dir / "custom.csv"
            with open(csv_path, "w") as f:
                f.write(csv_content)

            dataset = ImageEditDataset(
                data_path=str(tmpdir),
                metadata_path=str(csv_path),
                height=256,
                width=256,
            )

            assert len(dataset) == 1
            assert dataset[0]["prompts"] == "Custom metadata prompt"


class TestImageEditDatasetImageProcessing:
    """Tests for image loading and processing."""

    def test_image_normalization(self):
        """Test that images are normalized to [-1, 1]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            images_dir = tmpdir / "images"
            images_dir.mkdir()
            prompts_dir = tmpdir / "prompts"
            prompts_dir.mkdir()

            # Create a test image
            img = Image.new("RGB", (64, 64), color=(128, 128, 128))
            img.save(images_dir / "test.png")

            with open(prompts_dir / "test.txt", "w") as f:
                f.write("Test prompt")

            dataset = ImageEditDataset(
                data_path=str(tmpdir),
                height=64,
                width=64,
            )

            sample = dataset[0]
            source_image = sample["source_image"]

            # Check normalization range
            assert source_image.min() >= -1.0
            assert source_image.max() <= 1.0

    def test_image_resize(self):
        """Test that images are resized to target dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            images_dir = tmpdir / "images"
            images_dir.mkdir()
            prompts_dir = tmpdir / "prompts"
            prompts_dir.mkdir()

            # Create image with different size
            img = Image.new("RGB", (200, 100), color="red")
            img.save(images_dir / "test.png")

            with open(prompts_dir / "test.txt", "w") as f:
                f.write("Test prompt")

            dataset = ImageEditDataset(
                data_path=str(tmpdir),
                height=64,
                width=128,
            )

            sample = dataset[0]
            source_image = sample["source_image"]

            # Check dimensions [C, H, W]
            assert source_image.shape == (3, 64, 128)

    def test_missing_image_returns_none(self):
        """Test that missing images return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # CSV with non-existent image
            csv_content = """image,prompt
nonexistent.png,Test prompt
"""
            with open(tmpdir / "metadata.csv", "w") as f:
                f.write(csv_content)

            dataset = ImageEditDataset(
                data_path=str(tmpdir),
                height=64,
                width=64,
                edit_image_keys="image",
            )

            sample = dataset[0]
            # edit_images should be None since image doesn't exist
            assert sample["edit_images"] is None


class TestImageEditDatasetMaxCount:
    """Tests for max_count parameter."""

    def test_max_count_limits_samples(self):
        """Test that max_count limits the number of samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create many prompts
            prompts = [f"Prompt {i}" for i in range(100)]
            with open(tmpdir / "prompts.txt", "w") as f:
                f.write("\n".join(prompts))

            dataset = ImageEditDataset(
                data_path=str(tmpdir),
                height=256,
                width=256,
                max_count=10,
            )

            assert len(dataset) == 10

    def test_max_count_larger_than_data(self):
        """Test max_count larger than available data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            prompts = ["Prompt 1", "Prompt 2"]
            with open(tmpdir / "prompts.txt", "w") as f:
                f.write("\n".join(prompts))

            dataset = ImageEditDataset(
                data_path=str(tmpdir),
                height=256,
                width=256,
                max_count=1000,
            )

            assert len(dataset) == 2


class TestImageEditDatasetOutput:
    """Tests for dataset output format."""

    def test_output_keys(self, temp_data_dir):
        """Test that output contains expected keys."""
        prompts_dir = Path(temp_data_dir) / "prompts"

        dataset = ImageEditDataset(
            data_path=str(prompts_dir),
            height=256,
            width=256,
        )

        sample = dataset[0]

        assert "prompts" in sample
        assert "idx" in sample
        assert "source_image" in sample
        assert "edit_images" in sample
        assert "label_image" in sample

    def test_idx_matches_index(self, temp_data_dir):
        """Test that idx in output matches the requested index."""
        prompts_dir = Path(temp_data_dir) / "prompts"

        dataset = ImageEditDataset(
            data_path=str(prompts_dir),
            height=256,
            width=256,
        )

        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample["idx"] == i
