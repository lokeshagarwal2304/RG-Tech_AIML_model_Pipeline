"""
Rice Grain Classification PyTorch Dataset & DataLoader Pipeline
Project: Multi-Food-Grain Quality Analysis and Prediction Using Advanced Diffusion Models

Author: Senior PyTorch Computer Vision Engineer
Description:
    Clean, modular PyTorch Dataset implementation for reading the leakage-free
    rice manifest (splits/leak_free_split.csv).
    
    Includes:
      - 8-class label mapping (0_NOR to 7_IM).
      - Aspect-ratio preserving padding to prevent grain shape distortion.
      - Graceful dependency check for PyTorch and Pillow.
"""

import os
import csv
from pathlib import Path

# Try optional vision dependencies gracefully
try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from torchvision.transforms import functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    # Fallback base class if torch is not installed yet
    class Dataset:
        pass


# 1. Consistent, Documented Class Mapping
CLASS_NAMES = [
    "0_NOR",  # 0: Normal grains
    "1_F&S",  # 1: Fusarium & Scab damage
    "2_SD",   # 2: Sprouted damage
    "3_MY",   # 3: Moldy kernels
    "4_AP",   # 4: Attacked by pests
    "5_BN",   # 5: Broken grains
    "6_UN",   # 6: Unripe grains
    "7_IM",   # 7: Impurities
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}


# 2. Aspect-Ratio Preserving Transform (Square Pad)
class SquarePad:
    """Pads an image to a square aspect ratio using neutral background padding.
    
    Preserves natural grain elongation and aspect ratios without artificial stretching.
    """
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        if not _HAS_PIL:
            raise ImportError("Pillow is required for SquarePad. Run: pip install Pillow")
        w, h = img.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        pad_right = max_dim - w - pad_left
        pad_bottom = max_dim - h - pad_top
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return F.pad(img, padding, fill=self.fill, padding_mode="constant")


def get_transforms(split="train", target_size=224):
    """Returns image preprocessing transforms for train, val, or test.
    
    Args:
        split (str): 'train', 'val', or 'test'.
        target_size (int): Output square dimensions (default: 224).
    """
    if not _HAS_TORCH:
        return None

    # Normalization statistics (ImageNet standard)
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == "train":
        return transforms.Compose([
            SquarePad(fill=0),
            transforms.Resize((target_size, target_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            norm
        ])
    else:
        # Deterministic, augmentation-free evaluation pipeline
        return transforms.Compose([
            SquarePad(fill=0),
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            norm
        ])


# 3. PyTorch Dataset Implementation
class RiceGrainDataset(Dataset):
    """PyTorch Dataset for GrainDet Rice classification using the leak-free manifest.
    
    Args:
        manifest_path (str): Path to splits/leak_free_split.csv.
        split (str): One of 'train', 'val', 'test'.
        project_root (str): Project root directory to resolve relative image paths.
        transform (callable, optional): Transform pipeline to apply to images.
    """
    def __init__(self, manifest_path="splits/leak_free_split.csv", split="train",
                 project_root=None, transform=None):
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

        self.split = split
        self.manifest_path = manifest_path
        self.project_root = project_root if project_root else os.getcwd()
        self.transform = transform if transform else get_transforms(split)

        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        # Load and filter manifest
        self.samples = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_cols = {"relative_path", "class_name", "split"}
            if not required_cols.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"Manifest missing required columns: {required_cols}")

            for row_idx, row in enumerate(reader, start=2):
                if row["split"] != self.split:
                    continue

                class_name = row["class_name"]
                if class_name not in CLASS_TO_IDX:
                    raise ValueError(f"Unknown class '{class_name}' at row {row_idx}")

                label_idx = CLASS_TO_IDX[class_name]
                rel_path = row["relative_path"]
                full_path = os.path.join(self.project_root, rel_path)

                self.samples.append({
                    "full_path": full_path,
                    "relative_path": rel_path,
                    "class_name": class_name,
                    "label": label_idx
                })

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split '{self.split}' in manifest.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not _HAS_PIL:
            raise ImportError(
                "Pillow is not installed in the active environment. "
                "Image loading requires Pillow: pip install Pillow"
            )

        sample = self.samples[idx]
        image_path = sample["full_path"]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load image at '{image_path}': {e}")

        label = sample["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(manifest_path="splits/leak_free_split.csv",
                     project_root=None,
                     batch_size=32,
                     num_workers=0,
                     target_size=224):
    """Constructs DataLoader instances for train, val, and test splits."""
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is not installed in the active environment. "
            "DataLoader creation requires PyTorch: pip install torch torchvision"
        )

    train_ds = RiceGrainDataset(
        manifest_path=manifest_path,
        split="train",
        project_root=project_root,
        transform=get_transforms("train", target_size)
    )
    val_ds = RiceGrainDataset(
        manifest_path=manifest_path,
        split="val",
        project_root=project_root,
        transform=get_transforms("val", target_size)
    )
    test_ds = RiceGrainDataset(
        manifest_path=manifest_path,
        split="test",
        project_root=project_root,
        transform=get_transforms("test", target_size)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader
