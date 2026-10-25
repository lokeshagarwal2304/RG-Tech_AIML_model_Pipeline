"""
Baseline Training and Evaluation Script: MobileNetV3-Small (8-Class Rice Classification)
Project: Multi-Food-Grain Quality Analysis and Prediction Using Advanced Diffusion Models

Author: Senior PyTorch Computer Vision Engineer
Usage:
    python train_baseline.py [--epochs 3] [--batch_size 64] [--lr 0.001]
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
from torch.utils.data import DataLoader

# Add project root and src to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data.rice_dataset import (
    RiceGrainDataset,
    get_transforms,
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS
)
from models.classifier import build_mobilenet_v3_small


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(manifest_path, split="train"):
    """Computes inverse class frequency weights from training manifest records."""
    class_counts = Counter()
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split:
                class_counts[row["class_name"]] += 1

    total_samples = sum(class_counts.values())
    num_classes = len(CLASS_NAMES)
    weights = []
    for c in CLASS_NAMES:
        cnt = class_counts.get(c, 1)
        w = total_samples / (num_classes * cnt)
        weights.append(w)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize weights so mean is 1.0
    weight_tensor = weight_tensor / weight_tensor.mean()
    return weight_tensor, dict(class_counts)


def evaluate(model, dataloader, criterion, device):
    """Evaluates the model and computes full confusion matrix and per-class metrics."""
    model.eval()
    total_loss = 0.0
    num_samples = 0
    num_classes = len(CLASS_NAMES)

    # Confusion matrix: rows = true, cols = predicted
    confusion = [[0] * num_classes for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                confusion[t][p] += 1

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0

    # Calculate per-class metrics
    per_class = {}
    f1_list = []
    total_correct = 0

    for i in range(num_classes):
        tp = confusion[i][i]
        total_correct += tp
        fp = sum(confusion[r][i] for r in range(num_classes) if r != i)
        fn = sum(confusion[i][c] for c in range(num_classes) if c != i)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

        per_class[CLASS_NAMES[i]] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[i])
        }

    accuracy = total_correct / num_samples if num_samples > 0 else 0.0
    macro_f1 = sum(f1_list) / num_classes if num_classes > 0 else 0.0

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "sample_count": num_samples
    }


def format_confusion_matrix(confusion):
    """Formats confusion matrix into a clean ASCII table."""
    header = f"{'True/Pred':<10}" + "".join(f"{c:>8}" for c in CLASS_NAMES)
    lines = [header, "-" * len(header)]
    for i, row in enumerate(confusion):
        row_str = f"{CLASS_NAMES[i]:<10}" + "".join(f"{val:>8}" for val in row)
        lines.append(row_str)
    return "\n".join(lines)


def train_baseline(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device        : {device}")
    print(f"Random Seed   : {args.seed}")
    print(f"Batch Size    : {args.batch_size}")
    print(f"Epochs        : {args.epochs}")
    print(f"Learning Rate : {args.lr}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Compute Class Weights
    weights, train_counts = compute_class_weights(args.manifest, split="train")
    print(f"\nTraining Class Counts: {train_counts}")
    print("Computed Normalized Loss Weights:")
    for c, w in zip(CLASS_NAMES, weights.tolist()):
        print(f"  - {c:<6}: weight = {w:.4f}")

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    # 2. Build Datasets and DataLoaders
    print("\nLoading datasets from manifest:", args.manifest)
    train_ds = RiceGrainDataset(
        manifest_path=args.manifest,
        split="train",
        project_root=PROJECT_ROOT,
        transform=get_transforms("train", target_size=args.target_size)
    )
    val_ds = RiceGrainDataset(
        manifest_path=args.manifest,
        split="val",
        project_root=PROJECT_ROOT,
        transform=get_transforms("val", target_size=args.target_size)
    )
    test_ds = RiceGrainDataset(
        manifest_path=args.manifest,
        split="test",
        project_root=PROJECT_ROOT,
        transform=get_transforms("test", target_size=args.target_size)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    print(f"Dataset splits: Train={len(train_ds):,} | Val={len(val_ds):,} | Test={len(test_ds):,}")

    # 3. Initialize Model (Offline initialization)
    print("\nInitializing MobileNetV3-Small (8-class output, offline random init)...")
    model = build_mobilenet_v3_small(num_classes=len(CLASS_NAMES), pretrained=False)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 4. Training Loop
    best_val_f1 = -1.0
    best_checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
    epoch_history = []
    total_start_time = time.time()

    print("\n" + "=" * 70)
    print("STARTING TRAINING LOOP")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        processed_samples = 0
        batch_count = len(train_loader)

        for b_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            processed_samples += images.size(0)

            if b_idx % 50 == 0 or b_idx == batch_count:
                curr_loss = running_loss / processed_samples
                print(f"Epoch [{epoch}/{args.epochs}] Step [{b_idx}/{batch_count}] - Loss: {curr_loss:.4f}")

        scheduler.step()
        epoch_time = time.time() - epoch_start
        train_loss = running_loss / processed_samples

        # Validation Step
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]
        val_loss = val_metrics["loss"]

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_val_f1,
                "val_accuracy": val_acc,
                "class_names": CLASS_NAMES,
                "class_to_idx": CLASS_TO_IDX,
                "preprocessing": {
                    "target_size": args.target_size,
                    "transform": "SquarePad + Resize + Normalize",
                    "normalization": "ImageNet std"
                }
            }, best_checkpoint_path)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "epoch_time_sec": round(epoch_time, 2),
            "is_best": is_best
        }
        epoch_history.append(epoch_record)

        tag = "(*BEST*)" if is_best else ""
        print(f"--> Epoch {epoch} Complete in {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f} {tag}\n")

    total_training_time = time.time() - total_start_time
    print("=" * 70)
    print(f"TRAINING COMPLETE in {total_training_time:.1f}s. Best Val Macro-F1: {best_val_f1:.4f}")
    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    print("=" * 70)

    # 5. Final Held-Out Test Evaluation
    print("\nEVALUATING ON HELD-OUT TEST SET (Unseen Capture Sessions)...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 70)
    print("HELD-OUT TEST SET EVALUATION RESULTS")
    print("=" * 70)
    print(f"Test Samples    : {test_metrics['sample_count']:,}")
    print(f"Overall Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Macro F1-Score  : {test_metrics['macro_f1']:.4f}")
    print(f"Test Loss       : {test_metrics['loss']:.4f}\n")

    print("--- PER-CLASS PERFORMANCE ---")
    print(f"{'Class':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 52)
    for c in CLASS_NAMES:
        m = test_metrics["per_class"][c]
        print(f"{c:<8} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f} | {m['support']:<8}")

    print("\n--- CONFUSION MATRIX (Rows=True, Cols=Predicted) ---")
    print(format_confusion_matrix(test_metrics["confusion_matrix"]))
    print("=" * 70)

    # 6. Save Machine-Readable JSON & Human-Readable Markdown Report
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    report_path = os.path.join(args.output_dir, "eval_report.md")

    full_results = {
        "model_architecture": "MobileNetV3-Small",
        "num_classes": 8,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "total_training_time_seconds": round(total_training_time, 2),
        "epoch_history": epoch_history,
        "best_epoch": checkpoint["epoch"],
        "best_val_macro_f1": checkpoint["val_macro_f1"],
        "test_metrics": test_metrics
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Baseline Rice Classification Evaluation Report\n\n")
        f.write(f"- **Model**: MobileNetV3-Small (8 Classes)\n")
        f.write(f"- **Manifest**: `{args.manifest}` (Session-Grouped, Leakage-Free)\n")
        f.write(f"- **Training Device**: `{device}`\n")
        f.write(f"- **Total Training Time**: {total_training_time:.1f}s ({total_training_time/60:.2f} min)\n")
        f.write(f"- **Best Epoch**: {checkpoint['epoch']} (Val Macro-F1: {checkpoint['val_macro_f1']:.4f})\n\n")

        f.write("## Test Set Performance\n")
        f.write(f"- **Accuracy**: {test_metrics['accuracy']*100:.2f}%\n")
        f.write(f"- **Macro-F1**: {test_metrics['macro_f1']:.4f}\n")
        f.write(f"- **Test Loss**: {test_metrics['loss']:.4f}\n")
        f.write(f"- **Test Samples**: {test_metrics['sample_count']:,}\n\n")

        f.write("### Per-Class Metrics\n\n")
        f.write("| Class | Category | Precision | Recall | F1-Score | Support |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        desc_map = {
            "0_NOR": "Normal", "1_F&S": "Fusarium & Scab", "2_SD": "Sprouted Damage",
            "3_MY": "Moldy", "4_AP": "Attacked by Pests", "5_BN": "Broken",
            "6_UN": "Unripe", "7_IM": "Impurities"
        }
        for c in CLASS_NAMES:
            m = test_metrics["per_class"][c]
            f.write(f"| `{c}` | {desc_map[c]} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |\n")

        f.write("\n### Confusion Matrix\n```text\n")
        f.write(format_confusion_matrix(test_metrics["confusion_matrix"]))
        f.write("\n```\n")

    print(f"\nSaved machine-readable metrics to: {metrics_path}")
    print(f"Saved human-readable report to    : {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline MobileNetV3-Small on Rice Grain Dataset.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--target_size", type=int, default=224, help="Input image dimension (default: 224)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default: 0 for Windows)")
    parser.add_argument("--manifest", type=str, default="splits/leak_free_split.csv", help="Path to manifest")
    parser.add_argument("--output_dir", type=str, default="outputs/baseline", help="Directory for checkpoint/metrics")
    args = parser.parse_args()

    train_baseline(args)
