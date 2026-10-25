"""
Unit Test Suite: Commercial Quality Grading Engine
Project: GrainSight AI - Multi-Food-Grain Quality Analysis
Standards: AGMARK & USDA Commercial Food Grain Quality Frameworks
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.inference.grading import grade_single_kernel, compute_lot_grading, GRAIN_SPECIES_PROFILES


class TestGrainGrading(unittest.TestCase):
    def test_single_kernel_sound(self):
        result = grade_single_kernel("0_NOR", confidence=0.98)
        self.assertEqual(result["grade_badge"], "GRADE A")
        self.assertEqual(result["safety_status"], "SAFE")
        self.assertEqual(result["grade_color"], "#10b981")

    def test_single_kernel_broken(self):
        result = grade_single_kernel("5_BN", confidence=0.95)
        self.assertEqual(result["grade_badge"], "GRADE B")
        self.assertEqual(result["safety_status"], "COMMERCIAL DIVERTER")
        self.assertEqual(result["grade_color"], "#3b82f6")

    def test_single_kernel_moldy_reject(self):
        result = grade_single_kernel("3_MY", confidence=0.99)
        self.assertEqual(result["grade_badge"], "REJECT")
        self.assertEqual(result["safety_status"], "HAZARDOUS")
        self.assertEqual(result["grade_color"], "#ef4444")

    def test_single_kernel_fusarium_reject(self):
        result = grade_single_kernel("1_F&S")
        self.assertEqual(result["grade_badge"], "REJECT")
        self.assertEqual(result["safety_status"], "HAZARDOUS")
        self.assertEqual(result["grade_color"], "#ef4444")

    def test_lot_grading_premium(self):
        # 19 normal kernels, 1 broken kernel -> sound 95%, broken 5% -> Grade A
        codes = ["0_NOR"] * 19 + ["5_BN"] * 1
        lot = compute_lot_grading(codes)
        self.assertEqual(lot["grade_badge"], "GRADE A")
        self.assertEqual(lot["total_kernels"], 20)
        self.assertEqual(lot["sound_percentage"], 95.0)
        self.assertEqual(lot["broken_percentage"], 5.0)

    def test_lot_grading_reject_on_mold(self):
        # Even with 19 sound grains, 1 moldy grain triggers strict biological hazard rejection
        codes = ["0_NOR"] * 19 + ["3_MY"] * 1
        lot = compute_lot_grading(codes)
        self.assertEqual(lot["grade_badge"], "REJECT")
        self.assertGreater(lot["biological_defect_percentage"], 0.0)

    def test_species_profiles_exist(self):
        self.assertIn("Rice (Oryza sativa)", GRAIN_SPECIES_PROFILES)
        self.assertIn("Wheat (Triticum aestivum)", GRAIN_SPECIES_PROFILES)
        self.assertIn("Sorghum (Sorghum bicolor)", GRAIN_SPECIES_PROFILES)


if __name__ == "__main__":
    unittest.main()
