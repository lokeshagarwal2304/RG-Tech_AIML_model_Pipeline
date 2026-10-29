"""
GrainSight AI: Multi-Grain Tray Synthesizer
Author: Lokesh Agarwal | Date: October 29, 2026
Generates realistic multi-grain optical tray images for testing the bunch/cluster scanner.
"""

import os
import sys
import random
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_sample_tray():
    trays_dir = os.path.join(PROJECT_ROOT, "data", "sample_trays")
    os.makedirs(trays_dir, exist_ok=True)
    
    rice_test_dir = os.path.join(PROJECT_ROOT, "data", "rice", "test")
    if not os.path.exists(rice_test_dir):
        print(f"Directory not found: {rice_test_dir}")
        return

    # Gather sample crops from sound and defective classes
    classes = ["0_NOR", "1_F&S", "2_SD", "3_MY", "5_BN", "6_UN"]
    crop_pool = []
    for cls in classes:
        cls_dir = os.path.join(rice_test_dir, cls)
        if os.path.exists(cls_dir):
            files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(".png")]
            if files:
                crop_pool.extend([(f, cls) for f in files[:8]])

    if not crop_pool:
        print("No rice crops found to generate sample tray.")
        return

    # Create dark matte tray background (800 x 600)
    canvas_w, canvas_h = 900, 650
    tray = Image.new("RGB", (canvas_w, canvas_h), color=(18, 20, 24))
    
    # Place 16 grains in a natural grid with random rotations and slight jitter
    rows, cols = 4, 4
    dx = canvas_w // (cols + 1)
    dy = canvas_h // (rows + 1)
    
    random.seed(42)
    selected_samples = random.sample(crop_pool, min(16, len(crop_pool)))
    
    for idx, (path, cls) in enumerate(selected_samples):
        r = idx // cols
        c = idx % cols
        cx = dx * (c + 1) + random.randint(-15, 15)
        cy = dy * (r + 1) + random.randint(-15, 15)
        
        grain_img = Image.open(path).convert("RGBA")
        angle = random.randint(-40, 40)
        rotated = grain_img.rotate(angle, expand=True)
        
        # Paste centered at cx, cy
        gw, gh = rotated.size
        paste_x = max(10, cx - gw // 2)
        paste_y = max(10, cy - gh // 2)
        
        tray.paste(rotated.convert("RGB"), (paste_x, paste_y), mask=rotated.split()[3] if "A" in rotated.getbands() else None)

    out_path = os.path.join(trays_dir, "sample_rice_tray_16_grains.png")
    tray.save(out_path)
    print(f"[SUCCESS] Sample Multi-Grain Tray saved at: {out_path}")
    return out_path

if __name__ == "__main__":
    generate_sample_tray()
