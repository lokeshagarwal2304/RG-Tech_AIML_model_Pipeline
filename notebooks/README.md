# RG-Tech AI Research: Google Colab Diffusion Augmentation Workflow

**Project**: Multi-Food-Grain Quality Analysis and Prediction Using Advanced Diffusion Models  
**Target Notebook**: [`diffusion_augmentation_colab.ipynb`](file:///C:/Users/Lokesh%20Agarwal/.gemini/antigravity/scratch/RG-Tech_AIML_model_Pipeline/notebooks/diffusion_augmentation_colab.ipynb)  
**Target Classes**: Rare defect classes `2_SD` (Sprouted Damage) & `1_F&S` (Fusarium Blight & Scab)  

---

## 1. Zero-Leakage Upload Bundle Creation (Local Machine)

Hamare session-grouped manifest (`splits/leak_free_split.csv`) mein files ka `train` split unke legacy folder names se independent hai (kuch training images physical folder `data/rice/test` ya `data/rice/val` ke andar hain).

Isliye, **single reproducible packaging command** run karein:

```powershell
& ".\.venv\Scripts\python.exe" scripts/package_colab_bundle.py
```

### Script Execution Summary:
- Reads `splits/leak_free_split.csv` and filters strictly `split == 'train'` for `['2_SD', '1_F&S']`.
- Selected: **1,672 training images** (`2_SD`: 837 samples, `1_F&S`: 835 samples).
- Excluded: **713 validation/test samples** (Zero cross-split leakage).
- All 1,672 images are verified on disk, packaging them into `rice_colab_train_bundle.zip` (~146.87 MB) preserving their exact `relative_path`.
- Original local dataset files remain completely untouched.

---

## 2. Google Colab Setup & Runtime Configuration

1. **Colab Open Karein**:
   - Google Drive ya [Google Colab](https://colab.research.google.com/) par jayein.
   - `notebooks/diffusion_augmentation_colab.ipynb` file upload karein.

2. **GPU Enable Karein**:
   - Top menu bar par click karein: **Runtime $\to$ Change runtime type**.
   - Hardware accelerator: **T4 GPU** (Free tier available).
   - Click **Save**.

3. **Data Bundle Upload & Extraction**:
   - Left sidebar mein Files icon par click karein aur `rice_colab_train_bundle.zip` upload karein.
   - Ek Colab code cell mein project root par extract karein:
     ```python
     !unzip -q rice_colab_train_bundle.zip -d .
     ```
   - Extraction ke baad `splits/leak_free_split.csv` aur `data/rice/...` paths exactly map ho jayenge. Agar koi file missing hoti hai, toh notebook fail-early karega.

---

## 3. Safe Gated Execution Workflow

Notebook ke sabhi execution toggles by-default **`False`** par set hain. Sirf wahi cell chalega jisko aap explicitly `True` karenge:

1. **Section 1: Hardware Audit**:
   - Run directly: Verifies active T4 GPU, VRAM, aur CUDA driver.
2. **Section 2: Dependencies (`INSTALL_DEPENDENCIES = False`)**:
   - Colab mein diffusers install karne ke liye toggle ko `True` karein.
3. **Section 3 & 4: Data Ingestion & Path Audit**:
   - Run directly: 1,672 training samples verify karta hai. Fail-early protection active hai.
4. **Section 5: Cheap Pilot Run (`RUN_PILOT = False`)**:
   - Pilot verify karne ke liye `RUN_PILOT = True` karein. Sirf 4 images par 1 forward/backward pass chalega, jisse peak VRAM aur loss empirically report hoga.
5. **Section 6: Controlled Training (`RUN_FULL_TRAINING = False`)**:
   - Training start karne ke liye `RUN_FULL_TRAINING = True` karein (~15-20 mins on T4 GPU). Weights `checkpoints/rice_ddpm_model.pt` mein save honge.
6. **Section 7: Reverse Sampling (`RUN_SYNTHESIS = False`)**:
   - DDIM reverse diffusion sampling se candidate synthetic defect grains generate karein.
7. **Section 8: Quality & Provenance Filtering (`RUN_FILTERING = False`)**:
   - Corner background intensity ($RGB < 25$) aur grain aspect ratio ($1.8 \le L/W \le 4.8$) verify karta hai.
   - Metadata `synthetic_outputs/synthetic_provenance_manifest.csv` mein log hota hai.
8. **Section 9: Packaging (`PACKAGE_OUTPUTS = False`)**:
   - Approved outputs aur manifest ko `rice_diffusion_augmented_data.zip` mein compress karta hai.

---

## 4. Download & Local Retraining

1. Colab se `rice_diffusion_augmented_data.zip` download karein.
2. Apne local laptop par project folder ke andar extract karein:
   - Approved synthetic images: `data/rice_synthetic/`
   - Manifest: `splits/synthetic_provenance_manifest.csv`
3. Local CPU par downstream comparative retraining run karein:
   ```powershell
   & ".\.venv\Scripts\python.exe" train_diffusion_augmented.py
   ```
