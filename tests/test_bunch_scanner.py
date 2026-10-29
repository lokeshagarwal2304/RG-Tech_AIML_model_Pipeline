"""
Unit test for Multi-Grain Bunch Scanner
"""

import os
import sys
import unittest
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.inference.predictor import RiceGrainPredictor

class TestMultiGrainBunchScanner(unittest.TestCase):

    def setUp(self):
        self.predictor = RiceGrainPredictor()
        self.tray_path = os.path.join(PROJECT_ROOT, "data", "sample_trays", "sample_rice_tray_16_grains.png")

    def test_predict_bunch_detection(self):
        self.assertTrue(os.path.exists(self.tray_path), "Sample tray image must exist")
        res = self.predictor.predict_bunch(self.tray_path)
        
        self.assertIn("annotated_image", res)
        self.assertIn("detections", res)
        self.assertIn("lot_summary", res)
        
        total_detected = res["lot_summary"]["total_grains_detected"]
        print(f"[TEST] Total grains detected in sample tray: {total_detected}")
        self.assertGreaterEqual(total_detected, 8, "Should detect at least 8 grains on tray")
        
        # Verify bounding box and classification fields
        first_det = res["detections"][0]
        self.assertIn("bbox", first_det)
        self.assertIn("class_code", first_det)
        self.assertIn("confidence_percentage", first_det)
        self.assertIn("status", first_det)
        self.assertIn("commercial_grade", first_det)

if __name__ == "__main__":
    unittest.main()
