"""
Rice Grain Classifier Inference Engine
Project: Multi-Food-Grain Quality Analysis and Prediction Using Advanced Diffusion Models

Author: Senior PyTorch Computer Vision Engineer
Description:
    Production-ready inference module that loads the trained MobileNetV3-Small
    checkpoint once per session, applies exact aspect-ratio preserving preprocessing,
    and returns predictions, confidence, and Top-3 probabilities with human-readable labels.
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data.rice_dataset import get_transforms, SquarePad, CLASS_NAMES
from models.classifier import build_mobilenet_v3_small

# Human-Readable Category Names and Descriptions
HUMAN_LABELS = {
    "0_NOR": "Normal Grain (Sound / Healthy)",
    "1_F&S": "Fusarium & Scab Damage (Fungal Blight)",
    "2_SD":  "Sprouted Damage (Germinated / Sprouted)",
    "3_MY":  "Moldy Kernel (Visible Mold Growth)",
    "4_AP":  "Pest / Insect Damage (Attacked by Pests)",
    "5_BN":  "Broken Grain (Mechanically Fractured)",
    "6_UN":  "Unripe / Immature Grain (Chalky / Green)",
    "7_IM":  "Impurities (Chaff / Foreign Organic Matter)"
}


class RiceGrainPredictor:
    """Inference wrapper for the trained MobileNetV3-Small model."""

    _instance = None  # Singleton instance to ensure model loads only once per process

    def __init__(self, checkpoint_path=None, device="cpu"):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(PROJECT_ROOT, "outputs", "baseline", "best_model.pth")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at: {checkpoint_path}. "
                "Ensure train_baseline.py has been run first."
            )

        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        # Load checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.class_names = ckpt.get("class_names", CLASS_NAMES)
        self.class_to_idx = ckpt.get("class_to_idx", {name: i for i, name in enumerate(self.class_names)})
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}
        self.preprocessing_cfg = ckpt.get("preprocessing", {"target_size": 224})

        # Build and load model architecture
        self.model = build_mobilenet_v3_small(num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Deterministic evaluation transform (SquarePad + Resize(224) + ImageNet Normalize)
        self.transform = get_transforms(split="test", target_size=self.preprocessing_cfg.get("target_size", 224))

    @classmethod
    def get_instance(cls, checkpoint_path=None, device="cpu"):
        """Singleton accessor to guarantee model is loaded only once per session."""
        if cls._instance is None:
            cls._instance = cls(checkpoint_path=checkpoint_path, device=device)
        return cls._instance

    def predict(self, image_input):
        """Runs single-grain classification inference.
        
        Args:
            image_input (str, Path, or PIL.Image.Image): File path or loaded PIL Image.
            
        Returns:
            dict containing:
                - predicted_code: e.g. '0_NOR'
                - human_label: e.g. 'Normal Grain (Sound / Healthy)'
                - confidence: float between 0.0 and 1.0
                - top3_predictions: list of Top-3 dicts with code, label, and probability
                - all_probabilities: dict mapping all 8 class codes to probabilities
        """
        # 1. Image Loading and Validation
        if isinstance(image_input, (str, os.PathLike)):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Input image not found: {image_input}")
            try:
                img = Image.open(image_input).convert("RGB")
            except Exception as e:
                raise ValueError(f"Could not open image file: {e}")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}. Expected path or PIL Image.")

        # 2. Preprocess
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # 3. Model Forward Pass
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        # 4. Format Results
        all_probs = {self.idx_to_class[i]: float(probs[i].item()) for i in range(len(self.class_names))}

        top_k = torch.topk(probs, k=min(3, len(self.class_names)))
        top3 = []
        for p, idx in zip(top_k.values.tolist(), top_k.indices.tolist()):
            code = self.idx_to_class[idx]
            top3.append({
                "class_code": code,
                "label": HUMAN_LABELS.get(code, code),
                "probability": round(float(p), 4),
                "percentage": round(float(p) * 100, 2)
            })

        best_idx = int(top_k.indices[0].item())
        best_code = self.idx_to_class[best_idx]
        best_confidence = float(top_k.values[0].item())

        return {
            "predicted_code": best_code,
            "human_label": HUMAN_LABELS.get(best_code, best_code),
            "confidence": round(best_confidence, 4),
            "confidence_percentage": round(best_confidence * 100, 2),
            "top3_predictions": top3,
            "all_probabilities": {k: round(v, 4) for k, v in all_probs.items()}
        }
