# 🌾 GrainSight AI: Core Source Code Architecture

**Author:** Lokesh Agarwal  
**Date:** October 26, 2026  
**Repository:** https://github.com/lokeshagarwal2304/RG-Tech_AIML_model_Pipeline  

---

## 🏗️ Module Architecture

### 1. `src/inference/` (Production Edge Engine)
- **`predictor.py`**: MobileNetV3-Small singleton inference engine, pure NumPy Grad-CAM (`features[11]`), morphological contour extraction and defect segmentation masking (`generate_pixel_segmentation`), species conditioning (`Rice`, `Wheat`, `Sorghum`).
- **`grading.py`**: Universal multi-grain grading rules (AGMARK, USDA GIPSA, Codex Alimentarius), morphometric aspect ratio validation, physiological moisture prediction (`predict_quality_attributes`), 20-kernel batch lot simulator.
- **`pdf_report.py`**: 1-click lab inspection certificate generator with embedded optical crop, Grad-CAM, segmentation mask, moisture preservation attributes, and AGMARK/USDA grade badge.
- **`audit_logger.py`**: Enterprise audit logger writing immutable logs to `logs/inspection_audit.csv`.

### 2. `src/models/`
- **`classifier.py`**: MobileNetV3-Small neural backbone with Squeeze-and-Excitation attention blocks (~48ms CPU latency, 6.1 MB).

### 3. `src/data/`
- **`rice_dataset.py`**: Aspect-preserving `SquarePad` transforms and PyTorch Dataset loaders with class mapping (`0_NOR` to `7_IM`).
