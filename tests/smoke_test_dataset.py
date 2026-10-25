"""
Smoke-Test Suite for Rice Dataset Pipeline
Project: Multi-Food-Grain Quality Analysis and Prediction Using Advanced Diffusion Models

Author: Senior PyTorch Computer Vision Engineer
Usage:
    python tests/smoke_test_dataset.py
"""

import os
import sys
import csv

# Add project root and src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data.rice_dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    RiceGrainDataset,
    _HAS_PIL,
    _HAS_TORCH
)

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "splits", "leak_free_split.csv")


def test_manifest_integrity():
    print("[TEST 1/5] Checking manifest existence and split counts...")
    assert os.path.exists(MANIFEST_PATH), f"Manifest missing at {MANIFEST_PATH}"

    split_counts = {"train": 0, "val": 0, "test": 0}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_counts[row["split"]] += 1

    expected_counts = {"train": 17342, "val": 2472, "test": 4953}
    for s, exp in expected_counts.items():
        actual = split_counts[s]
        assert actual == exp, f"Split '{s}' count mismatch: expected {exp}, got {actual}"
        print(f"  - Split '{s}': {actual:,} images (matches expected {exp:,})")

    print("  -> PASSED: Manifest counts are 100% verified.\n")


def test_class_mappings():
    print("[TEST 2/5] Checking 8-class index mappings...")
    assert len(CLASS_NAMES) == 8, f"Expected 8 classes, got {len(CLASS_NAMES)}"
    for idx, c in enumerate(CLASS_NAMES):
        assert CLASS_TO_IDX[c] == idx, f"Mapping error for {c}"
        assert IDX_TO_CLASS[idx] == c, f"Reverse mapping error for {idx}"
        print(f"  - Class '{c}' -> Index {idx}")

    print("  -> PASSED: Class mappings are strictly 1-to-1 and consistent.\n")


def test_path_resolution():
    print("[TEST 3/5] Checking physical file path resolution...")
    for split in ["train", "val", "test"]:
        ds = RiceGrainDataset(
            manifest_path=MANIFEST_PATH,
            split=split,
            project_root=PROJECT_ROOT
        )
        sample = ds.samples[0]
        assert os.path.exists(sample["full_path"]), f"Sample image file not found: {sample['full_path']}"
        print(f"  - Split '{split}' sample verified: {sample['relative_path']}")

    print("  -> PASSED: Manifest paths cleanly resolve to files on disk.\n")


def test_error_handling():
    print("[TEST 4/5] Checking error handling on invalid inputs...")
    # Test invalid split
    try:
        RiceGrainDataset(manifest_path=MANIFEST_PATH, split="invalid_split", project_root=PROJECT_ROOT)
        raise AssertionError("Failed to raise ValueError on invalid split name!")
    except ValueError as e:
        print(f"  - Correctly raised ValueError on invalid split: {e}")

    # Test missing manifest
    try:
        RiceGrainDataset(manifest_path="non_existent_file.csv", split="train", project_root=PROJECT_ROOT)
        raise AssertionError("Failed to raise FileNotFoundError on missing manifest!")
    except FileNotFoundError as e:
        print(f"  - Correctly raised FileNotFoundError on missing file: {e}")

    print("  -> PASSED: Robust error handling verified.\n")


def test_dependency_boundary():
    print("[TEST 5/5] Checking image decoding & DataLoader dependency boundary...")
    
    # Check Pillow
    if _HAS_PIL:
        from PIL import Image
        print("  - Pillow is INSTALLED. Decoding sample images...")
        for split in ["train", "val", "test"]:
            ds = RiceGrainDataset(manifest_path=MANIFEST_PATH, split=split, project_root=PROJECT_ROOT)
            img, label = ds[0]
            print(f"    * Split '{split}' image loaded: mode={img.mode}, size={img.size}, label={label} ({IDX_TO_CLASS[label]})")
        print("  -> PASSED: Pillow decoding functional.")
    else:
        print("  - [DEPENDENCY BLOCKER] Pillow (PIL) is NOT installed in active .venv.")
        print("    -> Image decoding (__getitem__) cannot execute until Pillow is installed.")

    # Check PyTorch
    if _HAS_TORCH:
        from data.rice_dataset import get_data_loaders
        print("  - PyTorch is INSTALLED. Testing DataLoader batch creation...")
        train_loader, _, _ = get_data_loaders(
            manifest_path=MANIFEST_PATH,
            project_root=PROJECT_ROOT,
            batch_size=4,
            num_workers=0
        )
        images, labels = next(iter(train_loader))
        print(f"    * Batch loaded: images tensor shape={images.shape}, labels tensor shape={labels.shape}")
        print("  -> PASSED: PyTorch DataLoader batching functional.")
    else:
        print("  - [DEPENDENCY BLOCKER] PyTorch (torch/torchvision) is NOT installed in active .venv.")
        print("    -> Tensor transformation and DataLoader batching cannot execute until PyTorch is installed.")


def run_all_smoke_tests():
    print("=" * 70)
    print("RICE DATASET PIPELINE SMOKE-TEST SUITE")
    print("=" * 70)
    
    test_manifest_integrity()
    test_class_mappings()
    test_path_resolution()
    test_error_handling()
    test_dependency_boundary()
    
    print("=" * 70)
    print("SMOKE-TEST SUMMARY:")
    print("  - Core Pipeline Logic, Manifest Parsing & Path Resolution: PASS (100%)")
    print("  - Error Handling & Class Mappings: PASS (100%)")
    if not _HAS_PIL or not _HAS_TORCH:
        missing = []
        if not _HAS_PIL: missing.append("Pillow")
        if not _HAS_TORCH: missing.append("torch & torchvision")
        print(f"  - Next Action Required: Install missing runtime dependencies: {', '.join(missing)}")
    else:
        print("  - Full Pipeline (Pillow & PyTorch): READY")
    print("=" * 70)


if __name__ == "__main__":
    run_all_smoke_tests()
