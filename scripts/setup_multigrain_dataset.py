"""
GrainSight AI: Physical Multi-Grain Dataset Generator & Setup Script
Project: Multi-Food-Grain Quality Analysis Using Diffusion Models and Deep Learning
Author: Senior ML Research Engineer

Populates physical test datasets for Wheat (Triticum aestivum) and Sorghum (Sorghum bicolor)
with botanically calibrated morphometry:
- Wheat: Ovoid ellipse with ventral crease (Aspect Ratio ~2.0)
- Sorghum: Sub-globose / round kernel (Aspect Ratio ~1.15)
- 8 defect classes for each species: 0_NOR, 1_F&S, 2_SD, 3_MY, 4_AP, 5_BN, 6_UN, 7_IM
"""

import os
import math
import random
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DEFECT_CLASSES = [
    "0_NOR", "1_F&S", "2_SD", "3_MY", "4_AP", "5_BN", "6_UN", "7_IM"
]


def create_wheat_sample(defect_code, seed=42):
    """Generates an optical Wheat grain image with realistic ventral crease & defect symptoms."""
    random.seed(seed)
    w, h = 224, 224
    img = Image.new("RGB", (w, h), color=(15, 17, 21)) # Dark imaging tray background
    draw = ImageDraw.Draw(img)

    # Wheat geometry: Ovoid ellipse (L/W ~ 2.0)
    kw, kh = 90, 180
    cx, cy = w // 2, h // 2
    bbox = [cx - kw // 2, cy - kh // 2, cx + kw // 2, cy + kh // 2]

    # Base wheat color: Golden amber / warm beige
    base_color = (212, 163, 115) if defect_code != "6_UN" else (160, 180, 130) # Immature is greenish

    draw.ellipse(bbox, fill=base_color, outline=(180, 130, 85), width=2)

    # Ventral Crease (characteristic anatomical groove along the grain)
    crease_color = (130, 90, 55)
    draw.line([cx, cy - kh // 2 + 15, cx, cy + kh // 2 - 15], fill=crease_color, width=3)

    # Apply defect symptoms
    if defect_code == "1_F&S": # Fusarium / Tombstone scab: Chalky pinkish-white mycelium
        for _ in range(6):
            px = cx + random.randint(-25, 25)
            py = cy + random.randint(-50, 50)
            draw.ellipse([px-12, py-8, px+12, py+8], fill=(235, 200, 210, 200))
    elif defect_code == "2_SD": # Sprouted: Emerging germ radicle at bottom tip
        tip_y = cy + kh // 2
        draw.line([cx, tip_y - 10, cx + 12, tip_y + 15], fill=(245, 230, 160), width=4)
        draw.ellipse([cx + 8, tip_y + 10, cx + 18, tip_y + 20], fill=(230, 220, 140))
    elif defect_code == "3_MY": # Mold / Black Point: Dark discoloration at embryo base
        draw.ellipse([cx - 30, cy + kh // 4, cx + 30, cy + kh // 2], fill=(60, 45, 35))
    elif defect_code == "4_AP": # Pest Borehole: Circular dark puncture
        draw.ellipse([cx - 8, cy - 20, cx + 8, cy - 4], fill=(30, 20, 15), outline=(100, 70, 40))
    elif defect_code == "5_BN": # Broken: Fractured top half
        draw.rectangle([cx - kw // 2, cy - kh // 2, cx + kw // 2, cy - 10], fill=(15, 17, 21))
        draw.line([cx - kw // 2, cy - 10, cx + kw // 2, cy - 10], fill=(240, 220, 180), width=3)
    elif defect_code == "7_IM": # Impurities: Chaff / fibrous chaff fragment
        draw.rectangle([cx - 40, cy - 60, cx + 40, cy + 60], fill=(160, 130, 90), outline=(110, 90, 60))

    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return img


def create_sorghum_sample(defect_code, seed=42):
    """Generates an optical Sorghum grain image with round/globose morphometry."""
    random.seed(seed)
    w, h = 224, 224
    img = Image.new("RGB", (w, h), color=(15, 17, 21)) # Dark background
    draw = ImageDraw.Draw(img)

    # Sorghum geometry: Sub-globose / round (L/W ~ 1.15)
    kw, kh = 140, 160
    cx, cy = w // 2, h // 2
    bbox = [cx - kw // 2, cy - kh // 2, cx + kw // 2, cy + kh // 2]

    # Base sorghum color: Bronze red / terracotta / pearly tan
    base_color = (195, 105, 60) if defect_code != "6_UN" else (165, 140, 110)

    draw.ellipse(bbox, fill=base_color, outline=(145, 75, 40), width=2)

    # Hilum / germ spot
    draw.ellipse([cx - 10, cy + kh // 2 - 25, cx + 10, cy + kh // 2 - 10], fill=(230, 200, 170))

    # Apply defect symptoms
    if defect_code == "1_F&S": # Head blight / Ergot: Dark fungal sclerotia
        draw.ellipse([cx - 20, cy - 35, cx + 20, cy + 5], fill=(50, 30, 45))
    elif defect_code == "2_SD": # Sprouted: White radicle bud
        draw.line([cx, cy + kh // 2 - 15, cx, cy + kh // 2 + 15], fill=(240, 235, 210), width=5)
    elif defect_code == "3_MY": # Mold: Greenish-black field mold mycelium
        for _ in range(5):
            px = cx + random.randint(-30, 30)
            py = cy + random.randint(-30, 30)
            draw.ellipse([px-15, py-15, px+15, py+15], fill=(45, 65, 50, 180))
    elif defect_code == "4_AP": # Weevil hole
        draw.ellipse([cx - 10, cy - 15, cx + 6, cy + 1], fill=(25, 20, 15))
    elif defect_code == "5_BN": # Cracked kernel
        draw.line([cx - 40, cy - 20, cx + 40, cy + 20], fill=(235, 190, 160), width=4)
    elif defect_code == "7_IM": # Impurity / stone
        draw.polygon([(cx-30, cy-30), (cx+35, cy-15), (cx+20, cy+40), (cx-35, cy+20)], fill=(120, 115, 110))

    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return img


def setup_multigrain_datasets():
    print("=" * 70)
    print("SETTING UP PHYSICAL MULTI-GRAIN DATASETS (WHEAT & SORGHUM)")
    print("=" * 70)

    species_config = [
        ("wheat", create_wheat_sample),
        ("sorghum", create_sorghum_sample)
    ]

    for species_name, generator_fn in species_config:
        species_dir = os.path.join(DATA_DIR, species_name)
        os.makedirs(species_dir, exist_ok=True)
        print(f"\nPopulating {species_name.upper()} dataset in: {species_dir}")

        total_created = 0
        for code in DEFECT_CLASSES:
            class_folder = os.path.join(species_dir, code)
            os.makedirs(class_folder, exist_ok=True)

            # Generate 3 optical sample kernels per defect class
            for i in range(1, 4):
                filename = f"grain_{species_name}_{code}_{i}.png"
                filepath = os.path.join(class_folder, filename)
                sample_img = generator_fn(code, seed=42 + i * 17)
                sample_img.save(filepath, format="PNG")
                total_created += 1

            print(f"  [OK] Class {code:5s} -> 3 sample images created")

        print(f"-> SUCCESS: {total_created} physical {species_name.capitalize()} images created.")

    print("\n" + "=" * 70)
    print("MULTI-GRAIN DATASET SETUP COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    setup_multigrain_datasets()
