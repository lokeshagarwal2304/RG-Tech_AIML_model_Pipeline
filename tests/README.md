# 🧪 GrainSight AI: Automated Unit & Integration Tests

**Author:** Lokesh Agarwal  
**Date:** October 26, 2026  
**Repository:** https://github.com/lokeshagarwal2304/RG-Tech_AIML_model_Pipeline  

---

## 📋 Test Suite (12/12 Passing)
- `tests/test_inference.py`: Validates MobileNetV3 loading, SquarePad, top-3 probabilities, Grad-CAM heatmap generation, and pixel segmentation masks.
- `tests/test_grading.py`: Validates AGMARK, USDA GIPSA, and Codex Alimentarius grading rules, aspect-ratio thresholds, and moisture estimation.
- `tests/test_pdf_report.py`: Validates one-click PDF lab certificate generation, embedding optical crop, heatmap, and segmentation mask.

### Run All Tests:
```powershell
python -m unittest discover -s tests -p "test_*.py"
```
Output:
```text
Ran 12 tests in 0.103s — OK
```
