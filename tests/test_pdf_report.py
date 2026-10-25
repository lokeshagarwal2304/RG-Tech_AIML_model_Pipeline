"""
Unit Test Suite: Official PDF Inspection Report Generator
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


class TestPDFReportGenerator(unittest.TestCase):
    def setUp(self):
        self.dummy_img = Image.new("RGB", (224, 224), color="ivory")
        self.dummy_prediction = {
            "predicted_code": "0_NOR",
            "human_label": "Normal Grain (Sound / Healthy)",
            "confidence_percentage": 97.8,
            "confidence": 0.978,
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
                "safety_status": "Commercial Food Grade"
            }
        }

    def test_pdf_generation_with_gradcam(self):
        cam_img = Image.new("RGB", (224, 224), color="blue")
        pdf_bytes = generate_inspection_pdf(
            original_pil=self.dummy_img,
            gradcam_pil=cam_img,
            prediction_result=self.dummy_prediction,
            grain_species="Rice (Oryza sativa)"
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


if __name__ == "__main__":
    unittest.main()
