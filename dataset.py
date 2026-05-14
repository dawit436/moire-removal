import os
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def extract_index(filename):
    """Extract the numeric index from a filename like '0042_moire.jpg' → 42."""
    match = re.match(r"^(\d+)", Path(filename).stem)
    return int(match.group(1)) if match else None


class MoireDataset(Dataset):
    """
    Paired dataset of moiré-corrupted and clean ground-truth images.

    File naming convention:
        moire dir : 0000_moire.jpg, 0001_moire.jpg, ...
        clean dir : 0000_gt.jpg,    0001_gt.jpg,    ...

    Both directories are scanned and matched by their leading numeric index.
    """

    def __init__(self, root_dir: str, split: str = "train", crop_size: int = 512):
        """
        Args:
            root_dir  : Path to the dataset root (contains train/ and test/).
            split     : 'train' or 'test'.
            crop_size : Spatial size of crops fed to the model (output size).
        """
        assert split in ("train", "test"), "split must be 'train' or 'test'"
        self.split = split
        self.crop_size = crop_size

        moire_dir = Path(root_dir) / split / "moire"
        clean_dir = Path(root_dir) / split / "clean"

        # Build index → filepath maps for both sides
        moire_map = self._index_files(moire_dir)
        clean_map = self._index_files(clean_dir)

        # Keep only indices that exist in both directories
        common = sorted(set(moire_map) & set(clean_map))
        if len(common) == 0:
            raise RuntimeError(
                f"No paired images found in {moire_dir} and {clean_dir}. "
                "Check that filenames start with matching numeric indices."
            )

        self.pairs = [(moire_map[i], clean_map[i]) for i in common]

    @staticmethod
    def _index_files(directory: Path):
        """Return {index: filepath} for every image file in directory."""
        supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        index_map = {}
        for f in sorted(directory.iterdir()):
            if f.suffix.lower() in supported:
                idx = extract_index(f.name)
                if idx is not None:
                    index_map[idx] = f
        return index_map

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        moire_path, clean_path = self.pairs[idx]

        moire_img = Image.open(moire_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        moire_tensor, clean_tensor = self._apply_transforms(moire_img, clean_img)

        return {
            "moire": moire_tensor,          # float32, [3, H, W], range [0, 1]
            "clean": clean_tensor,          # float32, [3, H, W], range [0, 1]
            "filename": moire_path.name,
        }

    def _apply_transforms(self, moire_img, clean_img):
        """Crop and augment both images with identical spatial transforms."""

        if self.split == "train":
            # 1. Random crop size variation — multi-scale training
            crop_size = random.choice([384, 512, 640])

            # 2. Random crop (same region for both images)
            i, j, h, w = self._random_crop_params(moire_img, crop_size)
            moire_img = TF.crop(moire_img, i, j, h, w)
            clean_img = TF.crop(clean_img, i, j, h, w)

            # 3. Resize to output crop_size if the sampled size differs
            if crop_size != self.crop_size:
                moire_img = TF.resize(moire_img, [self.crop_size, self.crop_size])
                clean_img = TF.resize(clean_img, [self.crop_size, self.crop_size])

            # 4. Random horizontal flip (p=0.5)
            if torch.rand(1).item() > 0.5:
                moire_img = TF.hflip(moire_img)
                clean_img = TF.hflip(clean_img)

            # 5. Random vertical flip (p=0.5)
            if torch.rand(1).item() > 0.5:
                moire_img = TF.vflip(moire_img)
                clean_img = TF.vflip(clean_img)

            # 6. Random 90° / 180° / 270° rotation (p=0.5)
            if torch.rand(1).item() > 0.5:
                angle = random.choice([90, 180, 270])
                moire_img = TF.rotate(moire_img, angle)
                clean_img = TF.rotate(clean_img, angle)

        else:
            # Center crop for deterministic evaluation
            moire_img = TF.center_crop(moire_img, self.crop_size)
            clean_img = TF.center_crop(clean_img, self.crop_size)

        # Convert PIL → float32 tensor in [0, 1]
        moire_tensor = TF.to_tensor(moire_img)
        clean_tensor = TF.to_tensor(clean_img)

        return moire_tensor, clean_tensor

    def _random_crop_params(self, img, crop_size: int):
        """Return (top, left, height, width) for a random crop of given size."""
        w, h = img.size
        th = tw = crop_size
        if w < tw or h < th:
            raise ValueError(
                f"Image size ({w}×{h}) is smaller than crop size ({tw}×{th}). "
                "Resize your images or reduce crop_size."
            )
        top  = torch.randint(0, h - th + 1, (1,)).item()
        left = torch.randint(0, w - tw + 1, (1,)).item()
        return top, left, th, tw
