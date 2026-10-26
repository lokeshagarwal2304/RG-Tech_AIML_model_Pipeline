# GrainSight AI: Google Colab Diffusion Augmentation Workflow (v2.0)

**Author:** Lokesh Agarwal  
**Date:** October 26, 2026  
**Project**: Multi-Food-Grain Quality Analysis Using Diffusion Models and Deep Learning  
**Notebook Version**: [`notebooks/diffusion_augmentation_v2_colab.ipynb`](file:///C:/Users/Lokesh%20Agarwal/.gemini/antigravity/scratch/RG-Tech_AIML_model_Pipeline/notebooks/diffusion_augmentation_v2_colab.ipynb)  
**Target Rare Defect Classes**: `2_SD` (Sprouted Damage) & `1_F&S` (Fusarium Blight & Scab)  
**Data Bundle**: `rice_colab_train_bundle.zip` (1,672 training images, ~146.87 MB)

---

## 1. Architectural Diagnostics: Why v1.0 Failed & What v2.0 Fixes

| Issue in v1.0 Experiment | Ground-Truth Cause | Engineering Fix in v2.0 |
| :--- | :--- | :--- |
| **Random Colored Noise Output** | 3 Epochs = ~314 steps. 70%+ timesteps in $T=1000$ schedule were never sampled. Reverse drift $\mu_\theta$ remained unlearned. | Training candidate: **80 to 120 epochs** (~8.3k to 12.5k steps). Periodic visual inspection grid saved every 20 epochs to confirm grain emergence. |
| **Weak Class Conditioning** | U-Net always trained with labels; no unconditional null token. CFG could not be computed. | Added `UNCOND_NULL_ID = 2` (`num_class_embeds = 3`). **15% unconditional class dropout** during training enables true **Classifier-Free Guidance (CFG)** ($w=3.0$). |
| **Lost Checkpoints on Disconnect** | Only raw `state_dict()` saved. Colab timeout caused complete training loss. | **Full state checkpointing**: saves model weights, optimizer state, global step, loss history, and RNG states. Supports `RESUME_FROM_CHECKPOINT = True`. |
| **No In-Training Visual Feedback** | Training ran blind until the end. | **Visual Inspection Grids**: Logs a 4-sample inspection grid (`visual_progress_v2/inspection_grid_epoch_XXX.png`) every 20 epochs. |

---

## 2. Step-by-Step Google Colab Execution Order (Beginner-Friendly)

### Step 1: Open Google Colab & Enable T4 GPU
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select `notebooks/diffusion_augmentation_v2_colab.ipynb`.
3. In top menu: **Runtime $\to$ Change runtime type $\to$ Hardware accelerator $\to$ T4 GPU**. Click **Save**.

### Step 2: Upload Data Bundle
1. In Colab left sidebar, click the **Folder / Files icon**.
2. Upload `rice_colab_train_bundle.zip` (generated locally via `scripts/package_colab_bundle.py`).
3. In a code cell, unzip the bundle:
   ```python
   !unzip -q rice_colab_train_bundle.zip -d .
   ```

### Step 3: Run Hardware Audit & Dependencies (Section 1 & 2)
- Run Section 1: Confirms active NVIDIA T4 GPU and ~15 GB VRAM.
- In Section 2, set `INSTALL_DEPENDENCIES = True` and run: installs `diffusers`, `accelerate`, and dependencies.

### Step 4: Run Data Audit & Small Pilot Mode (Section 3, 4 & 5)
- Run Section 3 & 4: Validates all 1,672 training images on disk.
- In Section 5, set `RUN_PILOT = True` and run: executes 1 batch test to empirically verify gradient flow and peak VRAM (~2.4 GB).

### Step 5: Execute Checkpointed Training (Section 6)
- In Section 6, set `RUN_FULL_TRAINING = True` (target: 100 epochs, ~25-30 mins on T4 GPU).
- Check `visual_progress_v2/` folder every 20 epochs: observe how grains emerge from noise into crisp kernels!
- If Colab disconnects, set `RESUME_FROM_CHECKPOINT = True` to resume from the latest epoch.

### Step 6: Reverse Sampling with True CFG (Section 7)
- Set `RUN_SYNTHESIS = True`, `CFG_SCALE = 3.0`.
- DDIM reverse sampling generates 25 candidates per class using true Classifier-Free Guidance.

### Step 7: Dual Quality Filtering & Packaging (Section 8 & 9)
- Run Section 8 (`RUN_FILTERING = True`): verifies background darkness ($RGB < 25$) and natural elongation ($1.8 \le L/W \le 4.8$).
- Run Section 9 (`PACKAGE_OUTPUTS = True`): compresses approved outputs into `grainsight_diffusion_augmented_v2.zip`.

---

## 3. Downstream Comparative Retraining (Local Machine)

After downloading `grainsight_diffusion_augmented_v2.zip` from Colab:
1. Extract approved images into local folder `data/rice_synthetic/`.
2. Run comparative retraining on local CPU:
   ```powershell
   & ".\.venv\Scripts\python.exe" train_diffusion_augmented.py --epochs 3
   ```
3. The script automatically evaluates on the **frozen 4,953 test kernels** and outputs the exact $\Delta F1$ and recall improvement over the baseline!
