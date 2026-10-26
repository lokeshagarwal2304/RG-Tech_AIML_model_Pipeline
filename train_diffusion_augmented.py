"""
GrainSight AI: Downstream Comparative Retraining Script
Project: Multi-Food-Grain Quality Analysis Using Diffusion Models and Deep Learning
Author: Lokesh Agarwal | Contact: lokeshagarwal2304@gmail.com | Updated: October 26, 2026

Usage:
    & ".\.venv\Scripts\python.exe" train_diffusion_augmented.py [--epochs 3] [--batch_size 64]

Fair Baseline Comparison Protocol:
- Identical Architecture: MobileNetV3-Small
- Identical Random Seed: 42
- Identical Preprocessing: SquarePad(0) + Resize(224) + Normalize(ImageNet)
- Identical Held-Out Test Set: 4,953 unseen kernels from splits/leak_free_split.csv
- Compares outputs/diffusion_augmented/metrics.json against outputs/baseline/metrics.json
"""

import os
import sys
import csv
import time
import json
import random
import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data.rice_dataset import (
    RiceGrainDataset,
    get_transforms,
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    SquarePad
)
from models.classifier import build_mobilenet_v3_small


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AugmentedRiceDataset(Dataset):
    """Combines real training images from leak_free_split.csv with approved synthetic images."""
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = rec["full_path"]
        img = Image.open(img_path).convert("RGB")
        label = rec["label_idx"]
        if self.transform:
            img = self.transform(img)
        return img, label


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    num_classes = len(CLASS_NAMES)
    conf_matrix = [[0] * num_classes for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

            preds = torch.argmax(outputs, dim=1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf_matrix[t.item()][p.item()] += 1

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    total_correct = sum(conf_matrix[i][i] for i in range(num_classes))
    overall_acc = total_correct / num_samples if num_samples > 0 else 0.0

    per_class = {}
    macro_f1_sum = 0.0
    for i, c_name in enumerate(CLASS_NAMES):
        tp = conf_matrix[i][i]
        fp = sum(conf_matrix[r][i] for r in range(num_classes)) - tp
        fn = sum(conf_matrix[i][c] for c in range(num_classes)) - tp
        support = sum(conf_matrix[i][c] for c in range(num_classes))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        macro_f1_sum += f1

        per_class[c_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support
        }

    macro_f1 = macro_f1_sum / num_classes

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(overall_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": conf_matrix,
        "sample_count": num_samples
    }


def main():
    parser = argparse.ArgumentParser(description="Retrain MobileNetV3 with Diffusion-Augmented Dataset")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic_dir", type=str, default="data/rice_synthetic")
    parser.add_argument("--manifest_path", type=str, default="splits/leak_free_split.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/diffusion_augmented")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("GRAINSIGHT AI: DOWNSTREAM DIFFUSION-AUGMENTED RETRAINING")
    print(f"Device: {device} | Seed: {args.seed} | Epochs: {args.epochs}")
    print("=" * 70)

    # 1. Load Real Dataset
    print(f"Loading real manifest: {args.manifest_path}")
    raw_df = []
    with open(args.manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_df.append(row)

    train_records = []
    val_records = []
    test_records = []

    for r in raw_df:
        rel = r["relative_path"].replace("\\", "/")
        full_p = os.path.join(PROJECT_ROOT, rel)
        entry = {
            "full_path": full_p,
            "class_name": r["class_name"],
            "label_idx": CLASS_TO_IDX[r["class_name"]],
            "split": r["split"],
            "is_synthetic": False
        }
        if r["split"] == "train":
            train_records.append(entry)
        elif r["split"] == "val":
            val_records.append(entry)
        elif r["split"] == "test":
            test_records.append(entry)

    print(f"Real Train: {len(train_records)} | Real Val: {len(val_records)} | Held-Out Test: {len(test_records)}")

    # 2. Ingest Approved Synthetic Data (if available)
    synth_dir = os.path.join(PROJECT_ROOT, args.synthetic_dir)
    synth_added = 0
    if os.path.exists(synth_dir):
        for fname in os.listdir(synth_dir):
            if fname.endswith(".png"):
                # Detect target class from filename (e.g. synth_2_SD_seed_0042.png)
                matched_cls = None
                for c in CLASS_NAMES:
                    if c in fname:
                        matched_cls = c
                        break
                if matched_cls:
                    train_records.append({
                        "full_path": os.path.join(synth_dir, fname),
                        "class_name": matched_cls,
                        "label_idx": CLASS_TO_IDX[matched_cls],
                        "split": "train",
                        "is_synthetic": True
                    })
                    synth_added += 1

    print(f"Synthetic Images Ingested into Training Split: {synth_added}")
    if synth_added == 0:
        print("[WARNING] No synthetic images found in data/rice_synthetic/.")
        print("To run with synthetic augmentation, extract grainsight_diffusion_augmented_v2.zip first.")

    total_train = len(train_records)
    print(f"Total Augmented Training Set Size: {total_train} samples")

    # 3. DataLoaders
    train_tf = get_transforms(split="train", target_size=224)
    val_tf = get_transforms(split="val", target_size=224)
    test_tf = get_transforms(split="test", target_size=224)

    train_ds = AugmentedRiceDataset(train_records, transform=train_tf)
    val_ds = AugmentedRiceDataset(val_records, transform=val_tf)
    test_ds = AugmentedRiceDataset(test_records, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 4. Model Setup
    model = build_mobilenet_v3_small(num_classes=8, pretrained=False)
    model.to(device)

    # Class weighting
    counts = Counter(r["label_idx"] for r in train_records)
    weights = [total_train / (8 * counts.get(i, 1)) for i in range(8)]
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    weight_tensor = weight_tensor / weight_tensor.mean()

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5. Training Loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_f1 = -1.0
    best_ckpt_path = os.path.join(args.output_dir, "best_augmented_model.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / total_train
        val_res = evaluate(model, val_loader, criterion, device)
        val_f1 = val_res["macro_f1"]
        val_acc = val_res["accuracy"]

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": best_val_f1,
                "val_accuracy": val_acc,
                "synthetic_samples_used": synth_added,
                "class_names": CLASS_NAMES
            }, best_ckpt_path)

        tag = "(*BEST*)" if is_best else ""
        print(f"Epoch [{epoch}/{args.epochs}] Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f} {tag}")

    # 6. Evaluation on FROZEN Test Set
    print("\nEVALUATING ON FROZEN HELD-OUT TEST SET (4,953 Kernels)...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    # 7. Load Baseline Metrics for Comparison Delta
    baseline_metrics_path = os.path.join(PROJECT_ROOT, "outputs", "baseline", "metrics.json")
    baseline_m = None
    if os.path.exists(baseline_metrics_path):
        with open(baseline_metrics_path, "r") as f:
            baseline_m = json.load(f)["test_metrics"]

    acc_curr = test_metrics["accuracy"]
    f1_curr = test_metrics["macro_f1"]
    acc_base = baseline_m["accuracy"] if baseline_m else acc_curr
    f1_base = baseline_m["macro_f1"] if baseline_m else f1_curr

    delta_acc = (acc_curr - acc_base) * 100
    delta_f1 = f1_curr - f1_base

    print("=" * 70)
    print("GRAINSIGHT AI: COMPARATIVE BENCHMARK REPORT")
    print("=" * 70)
    print(f"Baseline Accuracy       : {acc_base*100:.2f}%  -->  Augmented: {acc_curr*100:.2f}%  (Δ: {delta_acc:+.2f}%)")
    print(f"Baseline Macro-F1       : {f1_base:.4f}  -->  Augmented: {f1_curr:.4f}  (Δ: {delta_f1:+.4f})")
    print(f"2_SD (Sprouted) Recall : {baseline_m['per_class']['2_SD']['recall']*100:.2f}%  -->  {test_metrics['per_class']['2_SD']['recall']*100:.2f}%")
    print(f"1_F&S (Fusarium) Recall: {baseline_m['per_class']['1_F&S']['recall']*100:.2f}%  -->  {test_metrics['per_class']['1_F&S']['recall']*100:.2f}%")
    print("=" * 70)

    # Save metrics
    res_dump = {
        "model_architecture": "MobileNetV3-Small",
        "synthetic_samples_added": synth_added,
        "delta_accuracy_pct": round(delta_acc, 2),
        "delta_macro_f1": round(delta_f1, 4),
        "test_metrics": test_metrics
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(res_dump, f, indent=2)

    print(f"Metrics saved to: {args.output_dir}/metrics.json")


if __name__ == "__main__":
    main()
