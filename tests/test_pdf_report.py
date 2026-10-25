"""
Unit Test Suite: Official PDF Inspection Report Generator & Audit Logging
Project: GrainSight AI - Multi-Food-Grain Quality Analysis
"""

import os
import sys
import unittest
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.inference.pdf_report import generate_inspection_pdf
from src.inference.grading import predict_quality_attributes
from src.inference.audit_logger import record_inspection_audit, get_recent_audit_logs, AUDIT_LOG_FILE


class TestPDFReportGenerator(unittest.TestCase):
    def setUp(self):
        self.dummy_img = Image.new("RGB", (224, 224), color="ivory")
        self.dummy_prediction = {
            "predicted_code": "0_NOR",
            "human_label": "Normal Grain (Sound / Healthy)",
            "confidence_percentage": 97.8,
            "confidence": 0.978,
            "aspect_ratio": 2.45,
            "top3_predictions": [
                {"rank": 1, "class_code": "0_NOR", "label": "Normal Grain (Sound)", "percentage": 97.8, "probability": 0.978},
                {"rank": 2, "class_code": "5_BN", "label": "Broken Grain", "percentage": 1.2, "probability": 0.012},
                {"rank": 3, "class_code": "2_SD", "label": "Sprouted Grain", "percentage": 0.5, "probability": 0.005}
            ],
            "grading": {
                "grade_badge": "GRADE A",
                "grade_color": "#16a34a",
                "commercial_grade": "Premium Quality Food Grain",
                "actionable_verdict": "Approved for commercial dispatch and retail packaging.",
                "safety_status": "Commercial Food Grade",
                "standards_body": "AGMARK & USDA Standards"
            }
        }
        self.moist_info = predict_quality_attributes("0_NOR")

    def test_pdf_generation_full(self):
        cam_img = Image.new("RGB", (224, 224), color="blue")
        seg_img = Image.new("RGB", (224, 224), color="green")
        pdf_bytes = generate_inspection_pdf(
            original_pil=self.dummy_img,
            gradcam_pil=cam_img,
            prediction_result=self.dummy_prediction,
            grain_species="Rice (Oryza sativa)",
            moisture_info=self.moist_info,
            segmented_pil=seg_img
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        # Check PDF signature: files start with %PDF-
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

    def test_pdf_generation_without_gradcam(self):
        pdf_bytes = generate_inspection_pdf(
            original_pil=self.dummy_img,
            gradcam_pil=None,
            prediction_result=self.dummy_prediction,
            grain_species="Wheat (Triticum aestivum)"
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

    def test_audit_logging(self):
        cert_id = "GS-TEST-001"
        log_path = record_inspection_audit(
            cert_id=cert_id,
            grain_species="Rice (Oryza sativa)",
            prediction_result=self.dummy_prediction,
            moisture_info=self.moist_info,
            segmentation_info={"grain_pixel_area": 5000, "defect_pixel_area": 200}
        )
        self.assertTrue(os.path.exists(log_path))
        logs = get_recent_audit_logs(limit=5)
        self.assertGreater(len(logs), 0)
        self.assertEqual(logs[0]["Certificate_ID"], cert_id)


if __name__ == "__main__":
    unittest.main()
