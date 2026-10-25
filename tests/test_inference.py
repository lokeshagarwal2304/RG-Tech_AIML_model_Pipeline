"""
Inference Smoke-Test Suite
Project: Multi-Food-Grain Quality Analysis and Prediction Using Advanced Diffusion Models

Author: Senior PyTorch Computer Vision Engineer
Usage:
    python tests/test_inference.py
"""

import os
import sys

# Add project root and src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.inference.predictor import RiceGrainPredictor, HUMAN_LABELS


def run_inference_smoke_test():
    print("=" * 70)
    print("RICE CLASSIFICATION INFERENCE SMOKE-TEST")
    print("=" * 70)

    # 1. Test Checkpoint Loading
    print("[TEST 1/4] Loading trained checkpoint via RiceGrainPredictor singleton...")
    ckpt_path = os.path.join(PROJECT_ROOT, "outputs", "baseline", "best_model.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}"
    
    predictor = RiceGrainPredictor.get_instance(checkpoint_path=ckpt_path, device="cpu")
    assert len(predictor.class_names) == 8, f"Expected 8 classes, got {len(predictor.class_names)}"
    print(f"  - Loaded checkpoint: {ckpt_path}")
    print(f"  - Model architecture: MobileNetV3-Small (8 classes)")
    print(f"  - Preprocessing target size: {predictor.preprocessing_cfg.get('target_size')}x{predictor.preprocessing_cfg.get('target_size')}")
    print("  -> PASSED: Model and checkpoint loaded cleanly on CPU.\n")

    # 2. Test Prediction on Real Normal Sample
    print("[TEST 2/4] Testing prediction on real test dataset sample (Normal grain)...")
    sample_normal = os.path.join(PROJECT_ROOT, "data", "rice", "test", "0_NOR", "Grainset_rice_2020-02-12-18-18-42_1_p600s.png")
    assert os.path.exists(sample_normal), f"Sample image not found: {sample_normal}"

    res = predictor.predict(sample_normal)
    assert "predicted_code" in res
    assert "human_label" in res
    assert "confidence" in res
    assert "top3_predictions" in res
    assert "all_probabilities" in res

    assert 0.0 <= res["confidence"] <= 1.0, f"Invalid confidence: {res['confidence']}"
    assert len(res["top3_predictions"]) == 3, f"Expected 3 top predictions, got {len(res['top3_predictions'])}"
    assert len(res["all_probabilities"]) == 8, f"Expected 8 probabilities, got {len(res['all_probabilities'])}"

    print(f"  - Sample Image: {os.path.basename(sample_normal)}")
    print(f"  - Predicted: {res['predicted_code']} ({res['human_label']})")
    print(f"  - Confidence: {res['confidence_percentage']}%")
    print("  - Top-3 Probabilities:")
    for rank, item in enumerate(res["top3_predictions"], start=1):
        print(f"    {rank}. {item['class_code']}: {item['percentage']}% ({item['label']})")
    print("  -> PASSED: Output structure and probabilities verified.\n")

    # 3. Test Prediction on Real Defect Sample (Broken Grain)
    print("[TEST 3/4] Testing prediction on real defect sample (Broken grain)...")
    sample_broken = os.path.join(PROJECT_ROOT, "data", "rice", "test", "5_BN", "Grainset_rice_2020-02-13-08-56-30_0_p600s.png")
    assert os.path.exists(sample_broken), f"Sample broken image not found: {sample_broken}"
    res_bn = predictor.predict(sample_broken)
    print(f"  - Sample Image: {os.path.basename(sample_broken)}")
    print(f"  - Predicted: {res_bn['predicted_code']} ({res_bn['human_label']})")
    print(f"  - Confidence: {res_bn['confidence_percentage']}%")
    print("  - Top-3 Probabilities:")
    for rank, item in enumerate(res_bn["top3_predictions"], start=1):
        print(f"    {rank}. {item['class_code']}: {item['percentage']}% ({item['label']})")
    print("  -> PASSED: Defect sample classified successfully.\n")

    # 4. Test Error Handling
    print("[TEST 4/4] Testing error handling on nonexistent image...")
    try:
        predictor.predict("nonexistent_image_12345.png")
        raise AssertionError("Failed to raise FileNotFoundError on missing image!")
    except FileNotFoundError as e:
        print(f"  - Correctly raised FileNotFoundError: {e}")
    print("  -> PASSED: Error handling verified.\n")

    # Check Web Interface Dependency Status
    print("=" * 70)
    print("WEB DEMO INTERFACE DEPENDENCY STATUS:")
    try:
        import streamlit
        print("  - Streamlit: INSTALLED (Version " + str(streamlit.__version__) + ")")
        print("  -> Ready to launch: streamlit run app.py")
    except ImportError:
        print("  - Streamlit: NOT INSTALLED in active .venv")
        print("  -> Command to install: pip install streamlit")
        print("  -> Command to launch after install: streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    run_inference_smoke_test()
