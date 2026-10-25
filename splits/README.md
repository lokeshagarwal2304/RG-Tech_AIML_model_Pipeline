# Leakage-Free Dataset Split Manifest

## Overview
This directory contains the session-grouped, leakage-resistant split manifest for the GrainDet-Rice dataset (`data/rice/`), designed for reproducible, leak-free computer vision experimentation.

## Problem Solved
The original GrainDet folder partitions (`train`, `val`, `test`) suffered from **58.35% capture-session overlap** (89.59% of test sessions were also present in the training set). This caused models to memorize hardware illumination, sensor noise, and conveyor tray artifacts instead of true grain defect morphology.

## Manifest Details
- **File**: `leak_free_split.csv`
- **Total Images**: 24,767
- **Total Capture Sessions**: 7,513
- **Seed**: `42`
- **Columns**:
  - `relative_path`: Path from project root to the image (e.g. `data/rice/train/0_NOR/...png`).
  - `class_name`: Grain category label (`0_NOR` through `7_IM`).
  - `session_id`: Unique conveyor camera timestamp (`Grainset_rice_YYYY-MM-DD-HH-MM-SS`).
  - `split`: Target assignment (`train`, `val`, `test`).

## Partitions Summary
- **Train**: ~70% of images (Zero session overlap with val or test).
- **Validation**: ~10% of images (Zero session overlap with train or test).
- **Test**: ~20% of images (Completely unseen capture sessions for true generalization benchmarking).

## How to Regenerate or Validate
From the project root, run:
```powershell
python create_leak_free_splits.py --seed 42 --output splits/leak_free_split.csv
```
The script automatically validates:
1. Every discovered image exists exactly once in the manifest.
2. $\text{Train Sessions} \cap \text{Val Sessions} = \emptyset$
3. $\text{Train Sessions} \cap \text{Test Sessions} = \emptyset$
4. $\text{Val Sessions} \cap \text{Test Sessions} = \emptyset$
5. All 8 classes remain represented across every partition.
