"""
Unit Test Suite: Universal Multi-Food-Grain Quality Grading Engine
Project: GrainSight AI - Multi-Food-Grain Quality Analysis
Standards: AGMARK, USDA GIPSA & Codex Alimentarius Frameworks
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.inference.grading import grade_single_kernel, compute_lot_grading, GRAIN_SPECIES_PROFILES


class TestMultiGrainGrading(unittest.TestCase):
    # 1. Rice Standards (AGMARK)
    def test_rice_single_kernel_sound(self):
        result = grade_single_kernel("0_NOR", confidence=0.98, grain_species="Rice (Oryza sativa)", aspect_ratio=2.8)
        self.assertEqual(result["grade_badge"], "GRADE A")
        self.assertEqual(result["safety_status"], "SAFE")
        self.assertIn("Oryza sativa", result["morphology_note"])

    def test_rice_single_kernel_moldy(self):
        result = grade_single_kernel("3_MY", confidence=0.99, grain_species="Rice (Oryza sativa)")
        self.assertEqual(result["grade_badge"], "REJECT")
        self.assertEqual(result["safety_status"], "HAZARDOUS")

    # 2. Wheat Standards (USDA GIPSA)
    def test_wheat_single_kernel_sound(self):
        result = grade_single_kernel("0_NOR", confidence=0.96, grain_species="Wheat (Triticum aestivum)", aspect_ratio=2.1)
        self.assertEqual(result["grade_badge"], "US NO. 1")
        self.assertEqual(result["safety_status"], "SAFE")
        self.assertIn("USDA GIPSA", result["standards_body"])

    def test_wheat_single_kernel_fusarium_reject(self):
        result = grade_single_kernel("1_F&S", confidence=0.92, grain_species="Wheat (Triticum aestivum)")
        self.assertEqual(result["grade_badge"], "REJECT")
        self.assertEqual(result["safety_status"], "HAZARDOUS")

    # 3. Sorghum Standards (Codex Alimentarius)
    def test_sorghum_single_kernel_sound(self):
        result = grade_single_kernel("0_NOR", confidence=0.95, grain_species="Sorghum (Sorghum bicolor)", aspect_ratio=1.2)
        self.assertEqual(result["grade_badge"], "GRADE 1")
        self.assertEqual(result["safety_status"], "SAFE")
        self.assertIn("Codex", result["standards_body"])

    # 4. Multi-Grain Lot Grading
    def test_rice_lot_grading_premium(self):
        codes = ["0_NOR"] * 19 + ["5_BN"] * 1
        lot = compute_lot_grading(codes, grain_species="Rice (Oryza sativa)")
        self.assertEqual(lot["grade_badge"], "GRADE A")
        self.assertEqual(lot["total_kernels"], 20)

    def test_wheat_lot_grading_premium(self):
        codes = ["0_NOR"] * 19 + ["5_BN"] * 1
        lot = compute_lot_grading(codes, grain_species="Wheat (Triticum aestivum)")
        self.assertEqual(lot["grade_badge"], "US NO. 1")

    def test_sorghum_lot_grading_premium(self):
        codes = ["0_NOR"] * 19 + ["5_BN"] * 1
        lot = compute_lot_grading(codes, grain_species="Sorghum (Sorghum bicolor)")
        self.assertEqual(lot["grade_badge"], "GRADE 1")

    def test_species_profiles_exist(self):
        self.assertIn("Rice (Oryza sativa)", GRAIN_SPECIES_PROFILES)
        self.assertIn("Wheat (Triticum aestivum)", GRAIN_SPECIES_PROFILES)
        self.assertIn("Sorghum (Sorghum bicolor)", GRAIN_SPECIES_PROFILES)


if __name__ == "__main__":
    unittest.main()
