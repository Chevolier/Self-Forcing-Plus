from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
import torchvision.transforms.functional as TF


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TextFolderDataset(Dataset):
    def __init__(self, data_path, max_count=30000):
        self.texts = []
        count = 1
        for file in os.listdir(data_path):
            if file.endswith(".txt"):
                with open(os.path.join(data_path, file), "r") as f:
                    text = f.read().strip()
                    self.texts.append(text)
                    count += 1
                    if count > max_count:
                        break

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"prompts": self.texts[idx], "idx": idx}


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        img = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "img", np.uint8, local_idx,
            shape=(480, 832, 3)
        )
        img = Image.fromarray(img)
        img = TF.to_tensor(img).sub_(0.5).div_(0.5)

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32),
            "img": img
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


class ImageEditDataset(Dataset):
    """
    Dataset for image editing with CSV metadata support.
    Compatible with DiffSynth-Studio data format.

    Supports multiple data formats:
    1. CSV with edit_image_keys and label_image_key (DiffSynth-Studio style)
    2. Simple folder with images/ and prompts/ directories
    3. metadata.json file
    4. Folder of text files (prompts only)
    """

    def __init__(
        self,
        data_path: str,
        height: int = None,
        width: int = None,
        max_pixels: int = 1024 * 1024,  # Default ~1M pixels (1024x1024)
        height_division_factor: int = 16,
        width_division_factor: int = 16,
        max_count: int = 100000,
        # CSV-specific arguments (DiffSynth-Studio compatible)
        metadata_path: str = None,
        data_file_keys: str = None,  # Comma-separated, e.g., "label_image,cloth_image,model_image"
        edit_image_keys: str = None,  # Comma-separated, e.g., "cloth_image,model_image"
        label_image_key: str = None,  # e.g., "label_image"
        fixed_prompt: str = None,  # Fixed prompt for all samples
    ):
        """
        Args:
            data_path: Base path for the dataset (resolves relative paths in CSV)
            height: Target image height (if None, computed from max_pixels)
            width: Target image width (if None, computed from max_pixels)
            max_pixels: Maximum total pixels (used when height/width are None)
            height_division_factor: Ensure height is divisible by this (default: 16)
            width_division_factor: Ensure width is divisible by this (default: 16)
            max_count: Maximum number of samples to load
            metadata_path: Path to CSV/JSON metadata file
            data_file_keys: Comma-separated column names containing file paths
            edit_image_keys: Comma-separated column names for edit input images
            label_image_key: Column name for target/label image
            fixed_prompt: Fixed prompt to use for all samples (overrides CSV prompt column)
        """
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.base_path = Path(data_path)
        self.fixed_prompt = fixed_prompt
        self.samples = []

        # Parse comma-separated keys
        self.data_file_keys = data_file_keys.split(",") if data_file_keys else []
        self.edit_image_keys = edit_image_keys.split(",") if edit_image_keys else []
        self.label_image_key = label_image_key

        # Determine metadata path
        if metadata_path is not None:
            self._load_from_metadata(metadata_path, max_count)
        elif (self.base_path / "train_data.csv").exists():
            self._load_from_metadata(str(self.base_path / "train_data.csv"), max_count)
        elif (self.base_path / "metadata.csv").exists():
            self._load_from_metadata(str(self.base_path / "metadata.csv"), max_count)
        else:
            self._load_from_directory(max_count)

        print(f"ImageEditDataset: loaded {len(self.samples)} samples")

    def _load_from_metadata(self, metadata_path: str, max_count: int):
        """Load data from CSV or JSON metadata file."""
        import pandas as pd

        metadata_path = Path(metadata_path)

        if metadata_path.suffix == ".csv":
            df = pd.read_csv(metadata_path)
            data = [df.iloc[i].to_dict() for i in range(min(len(df), max_count))]
        elif metadata_path.suffix == ".json":
            with open(metadata_path, "r") as f:
                data = json.load(f)[:max_count]
        elif metadata_path.suffix == ".jsonl":
            data = []
            with open(metadata_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= max_count:
                        break
                    data.append(json.loads(line.strip()))
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")

        for item in data:
            sample = {"raw_data": item}

            # Get prompt
            if self.fixed_prompt:
                sample["prompt"] = self.fixed_prompt
            else:
                sample["prompt"] = item.get("prompt", item.get("caption", ""))

            # Get edit image paths (input images for editing)
            sample["edit_image_paths"] = []
            for key in self.edit_image_keys:
                if key in item and item[key]:
                    path = self._resolve_path(item[key])
                    if path.exists():
                        sample["edit_image_paths"].append(str(path))

            # Get label/target image path
            if self.label_image_key and self.label_image_key in item:
                path = self._resolve_path(item[self.label_image_key])
                if path.exists():
                    sample["label_image_path"] = str(path)
                else:
                    sample["label_image_path"] = None
            else:
                sample["label_image_path"] = None

            self.samples.append(sample)

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve path relative to base_path if not absolute."""
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self.base_path / path

    def _load_from_directory(self, max_count: int):
        """Load data from directory structure (fallback)."""
        # Option 1: images/ and prompts/ directories
        images_dir = self.base_path / "images"
        prompts_dir = self.base_path / "prompts"

        if images_dir.exists() and prompts_dir.exists():
            for img_file in sorted(images_dir.glob("*")):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    prompt_file = prompts_dir / f"{img_file.stem}.txt"
                    if prompt_file.exists():
                        with open(prompt_file, "r") as f:
                            prompt = f.read().strip()
                        self.samples.append({
                            "edit_image_paths": [str(img_file)],
                            "label_image_path": None,
                            "prompt": self.fixed_prompt or prompt,
                        })
                        if len(self.samples) >= max_count:
                            break
            return

        # Option 2: Just prompts (for text-to-image without source)
        if (self.base_path / "prompts.txt").exists():
            with open(self.base_path / "prompts.txt", "r") as f:
                prompts = [line.strip() for line in f if line.strip()]
            for prompt in prompts[:max_count]:
                self.samples.append({
                    "edit_image_paths": [],
                    "label_image_path": None,
                    "prompt": self.fixed_prompt or prompt,
                })
            return

        # Option 3: Folder of text files
        for file in sorted(os.listdir(self.base_path)):
            if file.endswith(".txt"):
                with open(self.base_path / file, "r") as f:
                    prompt = f.read().strip()
                self.samples.append({
                    "edit_image_paths": [],
                    "label_image_path": None,
                    "prompt": self.fixed_prompt or prompt,
                })
                if len(self.samples) >= max_count:
                    break

    def _get_target_size(self, image: Image.Image) -> tuple:
        """
        Compute target height and width based on image size and max_pixels.
        Compatible with DiffSynth-Studio's ImageCropAndResize logic.
        """
        if self.height is not None and self.width is not None:
            return self.height, self.width

        width, height = image.size
        if width * height > self.max_pixels:
            scale = (width * height / self.max_pixels) ** 0.5
            height, width = int(height / scale), int(width / scale)

        # Round to division factors
        height = height // self.height_division_factor * self.height_division_factor
        width = width // self.width_division_factor * self.width_division_factor

        return height, width

    def _crop_and_resize(self, image: Image.Image, target_height: int, target_width: int) -> Image.Image:
        """
        Crop and resize image to target size while preserving aspect ratio.
        Compatible with DiffSynth-Studio's ImageCropAndResize logic.
        """
        width, height = image.size
        scale = max(target_width / width, target_height / height)

        # Resize maintaining aspect ratio
        new_width = round(width * scale)
        new_height = round(height * scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Center crop to target size
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        image = image.crop((left, top, left + target_width, top + target_height))

        return image

    def _load_and_process_image(self, image_path: str) -> tuple:
        """
        Load an image and process it to tensor.

        Returns:
            tuple: (image_tensor, height, width)
        """
        image = Image.open(image_path).convert("RGB")

        # Get target size based on max_pixels
        target_height, target_width = self._get_target_size(image)

        # Crop and resize
        image = self._crop_and_resize(image, target_height, target_width)

        # Convert to tensor and normalize to [-1, 1]
        image = TF.to_tensor(image)
        image = image * 2.0 - 1.0

        return image, target_height, target_width

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        result = {
            "prompts": sample["prompt"],
            "idx": idx,
        }

        # Load edit images (input images for editing task)
        edit_images = []
        target_height, target_width = None, None
        for path in sample.get("edit_image_paths", []):
            if path and os.path.exists(path):
                img, h, w = self._load_and_process_image(path)
                edit_images.append(img)
                # Use dimensions from first image
                if target_height is None:
                    target_height, target_width = h, w

        if edit_images:
            # Stack edit images: [N, C, H, W] where N is number of edit images
            result["edit_images"] = torch.stack(edit_images)
        else:
            result["edit_images"] = None

        # Load label/target image (for supervised training)
        label_path = sample.get("label_image_path")
        if label_path and os.path.exists(label_path):
            label_img, h, w = self._load_and_process_image(label_path)
            result["label_image"] = label_img
            # Use label image dimensions if no edit images
            if target_height is None:
                target_height, target_width = h, w
        else:
            result["label_image"] = None

        # For backward compatibility: source_image is first edit image
        if edit_images:
            result["source_image"] = edit_images[0]
        else:
            result["source_image"] = None

        # Include image dimensions in result
        result["height"] = target_height
        result["width"] = target_width

        return result


def cycle(dl):
    while True:
        for data in dl:
            yield data
