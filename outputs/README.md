# 📊 GrainSight AI: Model Checkpoints & Evaluation Metrics

**Author:** Lokesh Agarwal  
**Project:** GrainSight AI (Problem 1)  
**Date:** October 26, 2026  
**Repository:** https://github.com/lokeshagarwal2304/RG-Tech_AIML_model_Pipeline  

---

## 📁 Directory Structure
- `outputs/baseline/best_model.pth`: MobileNetV3-Small production model weights (6.1 MB, 1.52M parameters).
- `outputs/baseline/metrics.json`: Audited test metrics on 4,953 untouched test kernels across unseen photography sessions:
  - **Overall Accuracy:** 89.74%
  - **Macro-F1 Score:** 0.7968
  - **Macro Precision:** 83.57%
  - **Macro Recall:** 79.62%
  - **Single Kernel Inference Latency:** ~48 ms (Intel CPU)
