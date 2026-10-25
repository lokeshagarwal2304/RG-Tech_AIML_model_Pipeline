"""
GrainSight AI: Multi-Food-Grain Inference Engine with Grad-CAM Explainability
Project: Multi-Food-Grain Quality Analysis Using Diffusion Models and Deep Learning
Author: Senior PyTorch Computer Vision Engineer

Description:
    Production inference module featuring:
    - Cached singleton model loading (MobileNetV3-Small)
    - Aspect-ratio preserving SquarePad preprocessing
    - Top-3 class probability distribution
    - Grad-CAM (XAI) visual defect attention maps
    - AGMARK / USDA Commercial Grade assignment
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def _jet_colormap_numpy(val):
    """Pure NumPy Jet colormap (Blue -> Cyan -> Yellow -> Red) without external dependencies."""
    r = np.clip(1.5 - np.abs(4.0 * val - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * val - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * val - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data.rice_dataset import get_transforms, SquarePad, CLASS_NAMES
from models.classifier import build_mobilenet_v3_small
from inference.grading import grade_single_kernel, GRAIN_SPECIES_PROFILES

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
    """Inference wrapper for MobileNetV3-Small with Grad-CAM and Grading."""

    _instance = None  # Singleton instance

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

        # Deterministic evaluation transform
        self.transform = get_transforms(split="test", target_size=self.preprocessing_cfg.get("target_size", 224))

    @classmethod
    def get_instance(cls, checkpoint_path=None, device="cpu"):
        """Singleton accessor to guarantee model is loaded only once per session."""
        if cls._instance is None:
            cls._instance = cls(checkpoint_path=checkpoint_path, device=device)
        return cls._instance

    def _load_image(self, image_input):
        if isinstance(image_input, (str, os.PathLike)):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Input image not found: {image_input}")
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

    def predict(self, image_input, grain_species="Rice (Oryza sativa)"):
        """Runs single-grain classification and commercial grading inference with species awareness."""
        img = self._load_image(image_input)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Morphological aspect ratio calculation
        w, h = img.size
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

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

        # Species-aware commercial grade assignment
        grade_info = grade_single_kernel(
            best_code,
            best_confidence,
            grain_species=grain_species,
            aspect_ratio=aspect_ratio
        )

        return {
            "predicted_code": best_code,
            "human_label": HUMAN_LABELS.get(best_code, best_code),
            "confidence": round(best_confidence, 4),
            "confidence_percentage": round(best_confidence * 100, 2),
            "grain_species": grain_species,
            "aspect_ratio": round(aspect_ratio, 2),
            "dimensions": (w, h),
            "top3_predictions": top3,
            "all_probabilities": {k: round(v, 4) for k, v in all_probs.items()},
            "grading": grade_info
        }

    def generate_gradcam(self, image_input, target_class_code=None):
        """Generates Explainable AI (Grad-CAM) visual attention heatmap.
        
        Highlights the exact morphological defect region on the grain kernel.
        """
        img = self._load_image(image_input)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        activations = []
        gradients = []

        def forward_hook(m, inp, out):
            activations.append(out)

        def backward_hook(m, grad_in, grad_out):
            gradients.append(grad_out[0])

        target_layer = self.model.features[-1]
        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        self.model.zero_grad()
        logits = self.model(tensor)

        if target_class_code is not None and target_class_code in self.class_to_idx:
            class_idx = self.class_to_idx[target_class_code]
        else:
            class_idx = logits.argmax(dim=1).item()

        score = logits[0, class_idx]
        score.backward()

        fwd_handle.remove()
        bwd_handle.remove()

        act = activations[0].detach()  # [1, 576, 7, 7]
        grad = gradients[0].detach()   # [1, 576, 7, 7]

        # Channel-wise global average pooling of gradients
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Apply Jet Colormap and blend with original image
        heatmap_rgb = _jet_colormap_numpy(cam)  # [224, 224, 3] in [0, 1]
        orig_resized = np.array(img.resize((224, 224))) / 255.0
        blended = 0.55 * orig_resized + 0.45 * heatmap_rgb
        blended = (np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8)

        blended_pil = Image.fromarray(blended)
        return {
            "blended_image": blended_pil,
            "raw_heatmap": cam,
            "target_class": self.idx_to_class[class_idx]
        }

    def generate_pixel_segmentation(self, image_input, raw_heatmap=None):
        """Generates pixel-level optical grain contour and defect segmentation masks."""
        img = self._load_image(image_input).resize((224, 224))
        img_np = np.array(img)

        # Grayscale intensity
        gray = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]

        # Grain body threshold (isolates grain from dark optical tray)
        threshold = max(35.0, float(np.mean(gray)) * 0.55)
        grain_mask = (gray > threshold)

        # Morphological border extraction (contour outline)
        shifted_up = np.roll(grain_mask, -1, axis=0)
        shifted_down = np.roll(grain_mask, 1, axis=0)
        shifted_left = np.roll(grain_mask, -1, axis=1)
        shifted_right = np.roll(grain_mask, 1, axis=1)
        eroded = grain_mask & shifted_up & shifted_down & shifted_left & shifted_right
        contour_mask = grain_mask & (~eroded)

        # Defect segmentation mask (using Grad-CAM activation inside grain mask)
        if raw_heatmap is not None:
            defect_mask = grain_mask & (raw_heatmap > 0.60)
        else:
            defect_mask = np.zeros_like(grain_mask)

        grain_px = int(np.sum(grain_mask))
        defect_px = int(np.sum(defect_mask))
        defect_ratio = float(defect_px / (grain_px + 1e-5))

        # Composite visual overlay
        vis = img_np.copy()
        vis[contour_mask] = [16, 185, 129]  # Emerald green outline

        if defect_px > 0:
            # Alpha blend crimson overlay over defect region
            vis[defect_mask] = (0.45 * vis[defect_mask] + 0.55 * np.array([239, 68, 68])).astype(np.uint8)

        return {
            "segmented_image": Image.fromarray(vis),
            "grain_pixel_area": grain_px,
            "defect_pixel_area": defect_px,
            "defect_area_ratio": round(defect_ratio * 100, 2)
        }
