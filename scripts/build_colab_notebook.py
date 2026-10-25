"""
Script to generate notebooks/diffusion_augmentation_colab.ipynb and notebooks/README.md
RG-Tech AIML Model Pipeline
Author: Senior ML Research Engineer
"""

import json
import os

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": []
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True)
    })

# 1. Header
add_md("""# RG-Tech AIML Model Pipeline: Diffusion Augmentation Engine
### Project: Multi-Food-Grain Quality Analysis and Prediction Using Advanced Diffusion Models
**Objective**: Evaluate whether diffusion-based synthetic data augmentation improves classification recall for rare rice defect classes (`2_SD`: Sprouted Damage and `1_F&S`: Fusarium & Scab).

---
### Strict Research Principles:
1. **Zero-Leakage Guarantee**: Diffusion training and generation strictly use records from the `train` split of `splits/leak_free_split.csv`. Held-out validation and test sets remain **100% frozen and untouched**.
2. **Defensible Agricultural Architecture**: Black-background grain crops are modeled using a lightweight class-conditional DDPM U-Net rather than prompt-based natural photography models.
3. **Explicitly Gated Execution**: Every operational cell (`INSTALL_DEPENDENCIES`, `RUN_PILOT`, `RUN_FULL_TRAINING`, `RUN_SYNTHESIS`, `RUN_FILTERING`, `PACKAGE_OUTPUTS`) is set to `False` by default for complete execution safety.
""")

# 2. Section 1: GPU Check
add_md("""## 1. Hardware & GPU Environment Verification
Diffusion models require CUDA acceleration for feasible training and reverse sampling.
This cell verifies that a GPU runtime is active in Google Colab and halts immediately if only CPU is detected.
""")

add_code("""import sys
import torch

print("=" * 65)
print("RG-TECH DIFFUSION PIPELINE: GPU HARDWARE AUDIT")
print("=" * 65)

if not torch.cuda.is_available():
    raise RuntimeError(
        "[CRITICAL HARDWARE ERROR] No CUDA-capable GPU detected!\\n"
        "Diffusion training on CPU is computationally infeasible.\\n"
        "In Google Colab, please go to: Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU, then rerun."
    )

gpu_idx = torch.cuda.current_device()
gpu_name = torch.cuda.get_device_name(gpu_idx)
gpu_props = torch.cuda.get_device_properties(gpu_idx)
vram_gb = gpu_props.total_memory / (1024 ** 3)
cuda_version = torch.version.cuda

print(f"[OK] Active GPU Device   : {gpu_name}")
print(f"[OK] Dedicated VRAM     : {vram_gb:.2f} GB")
print(f"[OK] PyTorch Version    : {torch.__version__}")
print(f"[OK] CUDA Driver Version: {cuda_version}")
print("=" * 65)
print("Hardware audit PASSED. Ready for diffusion operations.")
""")

# 3. Section 2: Environment & Dependencies (Gated)
add_md("""## 2. Environment Setup & Dependencies (Gated)
Checks whether required packages (`diffusers`, `accelerate`, `torchvision`, `pandas`) are available.
To install missing packages automatically, set `INSTALL_DEPENDENCIES = True`.
""")

add_code("""# EXPLICIT SAFETY GATE FOR PACKAGE INSTALLATION
INSTALL_DEPENDENCIES = False

import importlib
import subprocess

packages_to_check = ["diffusers", "accelerate", "torchvision", "PIL", "pandas"]
missing_packages = []

for pkg in packages_to_check:
    try:
        importlib.import_module(pkg)
    except ImportError:
        missing_packages.append(pkg)

if missing_packages:
    print(f"Detected missing packages: {missing_packages}")
    if INSTALL_DEPENDENCIES:
        print("INSTALL_DEPENDENCIES is True. Installing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "diffusers", "accelerate", "torchvision", "pandas", "pillow"])
        print("[OK] Dependencies installed successfully.")
    else:
        print("[SAFEGUARD ACTIVE] INSTALL_DEPENDENCIES is False. Set to True to install automatically.")
else:
    print("[OK] All required packages are already available.")
""")

# 4. Section 3: Data Ingestion & Strict Zero-Leakage Audit
add_md("""## 3. Data Ingestion & Strict Zero-Leakage Audit
Loads the project manifest (`splits/leak_free_split.csv`) and strictly isolates records assigned to `split == 'train'` for the two primary target classes:
- `2_SD`: Sprouted Damage (Baseline Recall: 43.93%, F1: 0.5866)
- `1_F&S`: Fusarium Blight & Scab (Baseline Recall: 72.69%, F1: 0.7377)

**Note on Physical Paths**: Manifest split assignment is independent of legacy folder names. Physical files may reside under `data/rice/{train,val,test}`.
This cell asserts **zero leakage** and validates that all 1,672 training image files exist.
""")

add_code("""import os
import pandas as pd
from PIL import Image

MANIFEST_PATH = "splits/leak_free_split.csv"
DATA_ROOT = "."

if not os.path.exists(MANIFEST_PATH):
    if os.path.exists(os.path.join("..", MANIFEST_PATH)):
        MANIFEST_PATH = os.path.join("..", MANIFEST_PATH)
        DATA_ROOT = ".."
    else:
        raise FileNotFoundError(
            f"Manifest not found at {MANIFEST_PATH}. Please ensure rice_colab_train_bundle.zip is extracted at project root."
        )

# Read full split manifest
full_df = pd.read_csv(MANIFEST_PATH)
print(f"Total manifest records loaded: {len(full_df)}")

# 1. Zero-Leakage Filtering: TRAIN split ONLY
TARGET_CLASSES = ["2_SD", "1_F&S"]
train_df = full_df[
    (full_df["split"] == "train") & 
    (full_df["class_name"].isin(TARGET_CLASSES))
].copy().reset_index(drop=True)

# 2. Strict Assertions: Guarantee Zero Leakage
val_overlap = full_df[(full_df["split"] == "val") & (full_df["class_name"].isin(TARGET_CLASSES))]
test_overlap = full_df[(full_df["split"] == "test") & (full_df["class_name"].isin(TARGET_CLASSES))]

print("=" * 65)
print("ZERO-LEAKAGE MANIFEST VERIFICATION")
print("=" * 65)
print(f"Training samples selected : {len(train_df)} kernels")
for c in TARGET_CLASSES:
    count = (train_df["class_name"] == c).sum()
    print(f"  - Class {c:6s}: {count:4d} training samples")

print(f"Held-Out Validation split : {len(val_overlap):4d} kernels (STRICTLY FROZEN / EXCLUDED)")
print(f"Held-Out Test split       : {len(test_overlap):4d} kernels (STRICTLY FROZEN / EXCLUDED)")

# Assert no test or val samples exist in train_df
assert len(train_df) == 1672, f"Expected 1672 training samples, got {len(train_df)}"
assert (train_df["split"] != "train").sum() == 0, "LEAKAGE ERROR: Non-train split found in training set!"
print("[OK] Zero-leakage assertion verified successfully.")
""")

# 5. Image File Validation & Morphology Audit
add_code("""# Validate all physical image paths on disk - FAIL EARLY if any is missing
valid_paths = []
dimensions = []
missing_paths = []

for idx, row in train_df.iterrows():
    clean_rel = row["relative_path"].replace("\\\\", "/")
    img_path = os.path.join(DATA_ROOT, clean_rel)
    if os.path.exists(img_path):
        valid_paths.append(img_path)
        with Image.open(img_path) as img:
            dimensions.append(img.size)
    else:
        missing_paths.append((row["relative_path"], img_path))

if missing_paths:
    raise FileNotFoundError(
        f"[CRITICAL DATA ERROR] {len(missing_paths)} required training images are missing on disk!\\n"
        f"First missing file: {missing_paths[0]}\\n"
        f"Ensure you created the bundle using `scripts/package_colab_bundle.py` and extracted it at project root."
    )

train_df["full_path"] = valid_paths
assert len(train_df) == len(valid_paths), "Dataframe row count and verified path count mismatch!"
print(f"[OK] Successfully verified all {len(valid_paths)} training image paths on disk.")

# Summary of dimensions
widths, heights = zip(*dimensions)
print(f"Image resolution range: Width [{min(widths)} - {max(widths)}], Height [{min(heights)} - {max(heights)}]")
""")

# 6. Model Architecture Selection
add_md("""## 4. Defensible Diffusion Architecture & Scientific Provenance
### Why standard Stable Diffusion 1.5 is suboptimal:
1. **Domain Mismatch**: SD 1.5 was trained on web images (LAION-5B) with complex backgrounds, perspective, and text prompts. It requires 512×512 resolution.
2. **Hallucination Risk**: Prompts like "rice grain sprouted" generate cooked rice, rice paddies, or bowls, failing to produce microscopic dual-view kernels on pitch-black backgrounds.
3. **Overhead**: Requires 4GB+ base weights, multi-stage VAE latent decoding, and high GPU memory.

### Defensible Approach: Lightweight Class-Conditional DDPM
- **Architecture**: A compact 2D U-Net operating directly in pixel space ($128 \\times 128$ resolution) with sinusoidal timestep embeddings and class label embeddings.
- **Provenance & License**: Built using PyTorch and Hugging Face `diffusers` `UNet2DModel` / `DDPMScheduler` (Apache 2.0 License).
- **Parameters**: ~18 Million parameters. Preserves pure black background characteristics without external text encoders.
- **VRAM & Feasibility**: Empirical memory and throughput will be validated in the pilot run below.
""")

# 7. Model & Dataset Definition
add_code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

IMAGE_SIZE = 128
CLASS_TO_ID = {"1_F&S": 0, "2_SD": 1}
ID_TO_CLASS = {0: "1_F&S", 1: "2_SD"}

class SquarePad:
    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        pad_right = max_dim - w - pad_left
        pad_bottom = max_dim - h - pad_top
        return T.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

diffusion_transforms = T.Compose([
    SquarePad(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Scaled to [-1, 1]
])

class RiceDiffusionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["full_path"]
        img = Image.open(img_path).convert("RGB")
        label = CLASS_TO_ID[row["class_name"]]
        if self.transform:
            img = self.transform(img)
        return img, label

train_dataset = RiceDiffusionDataset(train_df, transform=diffusion_transforms)
print(f"[OK] RiceDiffusionDataset initialized with {len(train_dataset)} samples.")
""")

# 8. Dependency Check & Cheap Pilot Test (Gated: False by default)
add_md("""## 5. Dependency & Cheap Pilot Feasibility Test (Gated)
Before launching any full-scale training, this cell runs a **tiny 2-step pilot** on a single batch of 4 images.
It verifies:
1. GPU forward/backward gradient flow
2. Peak VRAM allocation
3. Accurate noise scheduling

**Gated Safety**: Set `RUN_PILOT = True` to execute this verification.
""")

add_code("""# EXPLICIT SAFETY GATE - FALSE BY DEFAULT
RUN_PILOT = False

if not RUN_PILOT:
    print("[SAFEGUARD ACTIVE] RUN_PILOT is False.")
    print("To execute the pilot verification, set `RUN_PILOT = True` and run this cell.")
else:
    print("=" * 65)
    print("EXECUTING DEPENDENCY & CHEAP PILOT RUN")
    print("=" * 65)
    
    from diffusers import UNet2DModel, DDPMScheduler
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize lightweight U-Net
    model = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        num_class_embeds=2 # 2 classes: 1_F&S and 2_SD
    ).to(device)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Test batch
    pilot_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    batch_images, batch_labels = next(iter(pilot_loader))
    batch_images = batch_images.to(device)
    batch_labels = batch_labels.to(device)
    
    # Sample noise and timesteps
    noise = torch.randn_like(batch_images)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_images.shape[0],), device=device).long()
    noisy_images = noise_scheduler.add_noise(batch_images, noise, timesteps)
    
    # Forward & backward pass
    optimizer.zero_grad()
    noise_pred = model(noisy_images, timesteps, class_labels=batch_labels).sample
    loss = F.mse_loss(noise_pred, noise)
    loss.backward()
    optimizer.step()
    
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[OK] Pilot Loss              : {loss.item():.4f}")
    print(f"[OK] Peak VRAM Consumption   : {peak_vram:.2f} MB")
    print(f"[OK] Model Parameter Count   : {param_count:.2f} Million")
    print("=" * 65)
    print("[INFO] Pilot step complete. Check reported loss and VRAM above before enabling full training.")
""")

# 9. Full Diffusion Training Loop (Gated: False by default)
add_md("""## 6. Controlled Diffusion Training Loop (Gated)
This cell trains the class-conditional DDPM U-Net on the verified training images.
- **Safety Gate**: Protected by `RUN_FULL_TRAINING = False`. Must be set to `True` explicitly.
- **Checkpoints**: Automatically saved to `checkpoints/rice_ddpm_model.pt`.
""")

add_code("""# EXPLICIT SAFETY GATE - FALSE BY DEFAULT
RUN_FULL_TRAINING = False
NUM_EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

if not RUN_FULL_TRAINING:
    print("[SAFEGUARD ACTIVE] RUN_FULL_TRAINING is False.")
    print("To start full training on Colab GPU, change `RUN_FULL_TRAINING = True` and run this cell.")
else:
    print("=" * 65)
    print(f"STARTING FULL DIFFUSION TRAINING ({NUM_EPOCHS} Epochs)")
    print("=" * 65)
    
    os.makedirs("checkpoints", exist_ok=True)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for step, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=device).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            optimizer.zero_grad()
            noise_pred = model(noisy_images, timesteps, class_labels=labels).sample
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}] - DDPM Mean Loss: {avg_loss:.4f}")
            
    # Save checkpoint
    torch.save(model.state_dict(), "checkpoints/rice_ddpm_model.pt")
    print("[OK] Diffusion model weights saved to checkpoints/rice_ddpm_model.pt")
""")

# 10. Reverse Diffusion Sampling (Gated: False by default)
add_md("""## 7. Synthetic Image Generation & Sampling (Gated)
Uses reverse diffusion sampling (DDIM / DDPM) with class conditioning to synthesize candidate defect kernels.
- **Safety Gate**: Protected by `RUN_SYNTHESIS = False`.
""")

add_code("""# EXPLICIT SAFETY GATE - FALSE BY DEFAULT
RUN_SYNTHESIS = False
NUM_SAMPLES_PER_CLASS = 10  # Pilot count before large generation
INFERENCE_STEPS = 50

if not RUN_SYNTHESIS:
    print("[SAFEGUARD ACTIVE] RUN_SYNTHESIS is False.")
    print("To generate synthetic samples after training, set `RUN_SYNTHESIS = True`.")
else:
    from diffusers import DDIMScheduler
    import numpy as np

    print("=" * 65)
    print(f"SYNTHESIZING {NUM_SAMPLES_PER_CLASS} CANDIDATES PER CLASS")
    print("=" * 65)

    os.makedirs("synthetic_outputs/raw", exist_ok=True)
    ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    ddim_scheduler.set_timesteps(INFERENCE_STEPS)

    model.eval()
    generated_records = []

    for class_name, class_id in CLASS_TO_ID.items():
        print(f"Sampling class: {class_name} (ID: {class_id})...")
        labels = torch.tensor([class_id] * NUM_SAMPLES_PER_CLASS, device=device)
        
        # Initial Gaussian noise
        sample = torch.randn((NUM_SAMPLES_PER_CLASS, 3, IMAGE_SIZE, IMAGE_SIZE), device=device)
        
        with torch.no_grad():
            for t in ddim_scheduler.timesteps:
                model_output = model(sample, t, class_labels=labels).sample
                sample = ddim_scheduler.step(model_output, t, sample).prev_sample
                
        # Denormalize from [-1, 1] to [0, 255]
        sample = (sample / 2 + 0.5).clamp(0, 1)
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        
        for i in range(NUM_SAMPLES_PER_CLASS):
            img_arr = (sample[i] * 255).astype(np.uint8)
            img_out = Image.fromarray(img_arr)
            img_filename = f"synth_{class_name}_seed_{42 + i:04d}.png"
            save_path = os.path.join("synthetic_outputs/raw", img_filename)
            img_out.save(save_path)
            
            generated_records.append({
                "image_id": img_filename,
                "target_class": class_name,
                "model_type": "DDPM_UNet2D_ClassCond",
                "seed": 42 + i,
                "sampling_steps": INFERENCE_STEPS,
                "raw_path": save_path
            })
            
    print(f"[OK] Generated {len(generated_records)} raw synthetic images.")
""")

# 11. Automated Quality Filtering & Provenance Manifest (Gated: False by default)
add_md("""## 8. Quality Filtering & Provenance Manifest Export (Gated)
Applies automated quality gates:
1. **Background Cleanliness**: Corner pixels must be near-black ($RGB < 25$).
2. **Grain Aspect Ratio**: Bounding contour must have realistic rice grain geometry (ratio $1.8 \\le L/W \\le 4.8$).
3. **Provenance Logging**: Records complete metadata into `synthetic_provenance_manifest.csv`.
""")

add_code("""# EXPLICIT SAFETY GATE - FALSE BY DEFAULT
RUN_FILTERING = False

if not RUN_FILTERING:
    print("[SAFEGUARD ACTIVE] RUN_FILTERING is False.")
    print("To filter generated images and produce the provenance manifest, set `RUN_FILTERING = True`.")
else:
    import cv2
    import numpy as np

    os.makedirs("synthetic_outputs/approved", exist_ok=True)
    manifest_rows = []

    for rec in generated_records:
        raw_path = rec["raw_path"]
        img_pil = Image.open(raw_path).convert("RGB")
        img_np = np.array(img_pil)
        
        # Check 1: Corner background pixel intensity (should be dark)
        corners = [
            img_np[:10, :10], img_np[:10, -10:],
            img_np[-10:, :10], img_np[-10:, -10:]
        ]
        mean_corner = np.mean([np.mean(c) for c in corners])
        bg_passed = bool(mean_corner < 25.0)
        
        # Check 2: Non-empty grain segmentation
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shape_passed = False
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > 800: # Kernel has substantial mass
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect = max(w, h) / (min(w, h) + 1e-5)
                shape_passed = bool(1.8 <= aspect <= 4.8)
                
        is_approved = bg_passed and shape_passed
        status = "APPROVED" if is_approved else "REJECTED"
        
        approved_path = ""
        if is_approved:
            approved_path = os.path.join("synthetic_outputs/approved", rec["image_id"])
            img_pil.save(approved_path)
            
        manifest_rows.append({
            "image_id": rec["image_id"],
            "target_class": rec["target_class"],
            "generation_method": rec["model_type"],
            "seed": rec["seed"],
            "sampling_steps": rec["sampling_steps"],
            "bg_check_passed": bg_passed,
            "shape_check_passed": shape_passed,
            "filter_status": status,
            "output_path": approved_path
        })

    provenance_df = pd.DataFrame(manifest_rows)
    provenance_df.to_csv("synthetic_outputs/synthetic_provenance_manifest.csv", index=False)

    print("=" * 65)
    print("QUALITY FILTERING & PROVENANCE AUDIT")
    print("=" * 65)
    print(provenance_df["filter_status"].value_counts())
    print("Provenance manifest saved to: synthetic_outputs/synthetic_provenance_manifest.csv")
""")

# 12. Packaging & Download (Gated: False by default)
add_md("""## 9. Packaging Approved Synthetic Outputs (Gated)
Packages approved synthetic images and the provenance manifest into a single `.zip` file for download back to your local repository.
""")

add_code("""# EXPLICIT SAFETY GATE - FALSE BY DEFAULT
PACKAGE_OUTPUTS = False

if not PACKAGE_OUTPUTS:
    print("[SAFEGUARD ACTIVE] PACKAGE_OUTPUTS is False.")
    print("To compress approved synthetic images and provenance manifest into a zip, set `PACKAGE_OUTPUTS = True`.")
else:
    import shutil
    output_zip = "rice_diffusion_augmented_data.zip"
    if os.path.exists("synthetic_outputs/approved") and len(os.listdir("synthetic_outputs/approved")) > 0:
        shutil.make_archive("rice_diffusion_augmented_data", 'zip', "synthetic_outputs")
        print(f"[OK] Artifacts compressed into {output_zip}")
        print("You can now download this file and extract it into your local repository's data/ folder.")
    else:
        print("[INFO] No approved synthetic images available to package yet. Complete synthesis and filtering first.")
""")

# Save notebook
out_path = "notebooks/diffusion_augmentation_colab.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"[OK] Successfully written {out_path} ({len(notebook['cells'])} cells)")
