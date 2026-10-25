"""
MobileNetV3-Small Classification Model
Project: Multi-Food-Grain Quality Analysis and Prediction Using Advanced Diffusion Models

Author: Senior PyTorch Computer Vision Engineer
Description:
    Lightweight, CPU-compatible MobileNetV3-Small architecture for 8-class
    rice grain defect classification. Supports clean offline initialization.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


def build_mobilenet_v3_small(num_classes=8, pretrained=False):
    """Builds a MobileNetV3-Small classifier.
    
    Args:
        num_classes (int): Number of target defect categories (default: 8).
        pretrained (bool): Whether to load ImageNet weights (default: False for offline compatibility).
    """
    weights = "DEFAULT" if pretrained else None
    model = mobilenet_v3_small(weights=weights)

    # Replace classifier head for 8 classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model
