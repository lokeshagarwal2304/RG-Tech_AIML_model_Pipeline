"""
Packaging Script for Google Colab Diffusion Augmentation Bundle
Project: Multi-Food-Grain Quality Analysis and Prediction Using Advanced Diffusion Models
Author: Senior ML Research Engineer

Reads splits/leak_free_split.csv, selects ONLY records assigned to split == 'train'
for classes ['2_SD', '1_F&S'], and packages their exact physical files (preserving relative_path)
plus the manifest into rice_colab_train_bundle.zip without altering the original dataset.
"""

import os
import sys
import zipfile
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "splits", "leak_free_split.csv")
OUTPUT_ZIP_PATH = os.path.join(PROJECT_ROOT, "rice_colab_train_bundle.zip")
TARGET_CLASSES = ["2_SD", "1_F&S"]


def create_colab_bundle():
    print("=" * 70)
    print("RG-TECH DIFFUSION RESEARCH: ZERO-LEAKAGE COLAB BUNDLE PACKAGER")
    print("=" * 70)

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}")

    # 1. Read manifest
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Total manifest records: {len(df)}")

    # 2. Strict Filter: split == 'train' and class in target classes
    train_mask = (df["split"] == "train") & (df["class_name"].isin(TARGET_CLASSES))
    train_df = df[train_mask].copy().reset_index(drop=True)

    # 3. Leakage Guard Assertions
    val_leakage = df[(df["split"] == "val") & (df["class_name"].isin(TARGET_CLASSES))]
    test_leakage = df[(df["split"] == "test") & (df["class_name"].isin(TARGET_CLASSES))]

    assert len(train_df) == 1672, f"Expected 1,672 training records, got {len(train_df)}"
    assert (train_df["split"] != "train").sum() == 0, "Non-training record found in selection!"
    print(f"Training records selected: {len(train_df)}")
    for c in TARGET_CLASSES:
        cnt = (train_df["class_name"] == c).sum()
        print(f"  - Class {c:6s}: {cnt:4d} samples")

    print(f"Frozen Val excluded      : {len(val_leakage)} samples (Zero Leakage)")
    print(f"Frozen Test excluded     : {len(test_leakage)} samples (Zero Leakage)")

    # 4. Verify all physical files exist
    missing_files = []
    physical_paths = []
    for idx, row in train_df.iterrows():
        rel_path = row["relative_path"].replace("\\", "/")
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            missing_files.append((rel_path, full_path))
        else:
            physical_paths.append((full_path, rel_path))

    if missing_files:
        raise FileNotFoundError(
            f"ERROR: {len(missing_files)} selected training images are missing on disk!\n"
            f"First missing file: {missing_files[0]}"
        )

    print(f"[OK] All {len(physical_paths)} physical files verified present on disk.")

    # 5. Build zip archive
    if os.path.exists(OUTPUT_ZIP_PATH):
        print(f"Removing existing archive: {OUTPUT_ZIP_PATH}")
        os.remove(OUTPUT_ZIP_PATH)

    print(f"Creating archive: {OUTPUT_ZIP_PATH} ...")
    with zipfile.ZipFile(OUTPUT_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        # Include manifest
        manifest_arcname = "splits/leak_free_split.csv"
        zf.write(MANIFEST_PATH, arcname=manifest_arcname)

        # Include all 1,672 image files preserving their relative paths
        for full_p, rel_p in physical_paths:
            # Normalize arcname to forward slashes
            arcname = rel_p.replace("\\", "/")
            zf.write(full_p, arcname=arcname)

    # 6. Verify zip contents
    zip_size_mb = os.path.getsize(OUTPUT_ZIP_PATH) / (1024 * 1024)
    with zipfile.ZipFile(OUTPUT_ZIP_PATH, "r") as zf:
        file_list = zf.namelist()
        total_in_zip = len(file_list)
        # 1 manifest + 1672 images = 1673 files
        assert total_in_zip == 1673, f"Expected 1,673 files in zip, found {total_in_zip}"
        assert manifest_arcname in file_list, "Manifest missing from zip!"

    print("=" * 70)
    print("BUNDLE PACKAGING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Archive file : {OUTPUT_ZIP_PATH}")
    print(f"Archive size : {zip_size_mb:.2f} MB")
    print(f"Total files  : {total_in_zip} (1 manifest + 1,672 training images)")
    print("Original dataset files remain completely untouched.")
    print("=" * 70)


if __name__ == "__main__":
    create_colab_bundle()
