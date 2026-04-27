import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import os

from config import CFG
from dataset import get_dataloaders, load_class_names
from model import build_model, load_checkpoint
from utils.visualization import (
    plot_sample_predictions,
    plot_top_confused_classes,
    denormalize,
)
# ─── Core evaluation ─────────────────────────────────────────────────────────
@torch.no_grad()
def run_evaluation(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    collect_images: bool = True,
    max_images:     int  = 24,
) -> dict:
    model.eval()

    all_preds  = []
    all_labels = []
    all_probs  = []

    sample_images = []
    sample_preds  = []
    sample_labels = []
    sample_probs  = []
    sample_correct_done = False

    pbar = tqdm(loader, desc="Evaluating", unit="batch", colour="cyan")

    for images, labels in pbar:
        images_dev = images.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=CFG.use_amp):
            outputs = model(images_dev)
            probs   = torch.softmax(outputs, dim=1)

        preds      = outputs.argmax(dim=1)
        top_probs  = probs.max(dim=1).values

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(top_probs.detach().cpu().numpy())

        # Thu thập ảnh mẫu (cả đúng lẫn sai) để visualize
        if collect_images and len(sample_images) < max_images:
            for i in range(len(images)):
                if len(sample_images) >= max_images:
                    break
                sample_images.append(images[i])
                sample_preds.append(preds[i].cpu().item())
                sample_labels.append(labels[i].item())
                sample_probs.append(top_probs[i].cpu().item() * 100)

        acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
        pbar.set_postfix({"acc": f"{acc:.2f}%"})

    return {
        "all_preds":    np.array(all_preds),
        "all_labels":   np.array(all_labels),
        "all_probs":    np.array(all_probs),
        "sample_images": sample_images,
        "sample_preds":  sample_preds,
        "sample_labels": sample_labels,
        "sample_probs":  sample_probs,
    }
# ─── Report ──────────────────────────────────────────────────────────────────
def print_classification_report(
    all_preds:   np.ndarray,
    all_labels:  np.ndarray,
    class_names: list,
) -> dict:
    overall_acc = (all_preds == all_labels).mean() * 100

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Overall Accuracy (Top-1): {overall_acc:.2f}%")
    print(f"  Total Samples:            {len(all_labels):,}")
    print(f"  Correct Predictions:      {(all_preds == all_labels).sum():,}")
    print(f"{'=' * 60}\n")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=[n.replace("_", " ") for n in class_names],
        digits=3,
        output_dict=True,
    )

    # In report dạng text
    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=[n.replace("_", " ") for n in class_names],
        digits=3,
    )
    print(report_text)

    # Tìm top 5 classes tốt nhất và kém nhất
    per_class_f1 = {
        class_names[i]: report.get(n.replace("_", " "), {}).get("f1-score", 0)
        for i, n in enumerate(class_names)
        if n.replace("_", " ") in report
    }

    if per_class_f1:
        sorted_classes = sorted(per_class_f1.items(), key=lambda x: x[1])
        print("\n  Bottom 5 classes (lowest F1):")
        for cls, f1 in sorted_classes[:5]:
            print(f"    {cls:<30} F1: {f1:.3f}")

        print("\n  Top 5 classes (highest F1):")
        for cls, f1 in sorted_classes[-5:]:
            print(f"    {cls:<30} F1: {f1:.3f}")

    return {"overall_acc": overall_acc, "report": report}

# ─── Confusion Matrix ─────────────────────────────────────────────────────────
def plot_confusion_matrix(
    all_preds:   np.ndarray,
    all_labels:  np.ndarray,
    class_names: list,
    save_path:   str,
    normalize:   bool = True,
) -> None:
    cm = confusion_matrix(all_labels, all_preds)

    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt     = ".2f"
        title   = "Normalized Confusion Matrix"
    else:
        cm_plot = cm
        fmt     = "d"
        title   = "Confusion Matrix"

    fig_size = max(20, len(class_names) * 0.25)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    sns.heatmap(
        cm_plot,
        ax=ax,
        cmap="Blues",
        xticklabels=[n.replace("_", " ") for n in class_names],
        yticklabels=[n.replace("_", " ") for n in class_names],
        linewidths=0.1,
        linecolor="gray",
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.tick_params(axis="x", rotation=90, labelsize=5)
    ax.tick_params(axis="y", rotation=0,  labelsize=5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Confusion matrix saved to {save_path}")

# ─── Per-class accuracy bar chart ─────────────────────────────────────────────
def plot_per_class_accuracy(
    all_preds:   np.ndarray,
    all_labels:  np.ndarray,
    class_names: list,
    save_path:   str,
) -> None:
    per_class_acc = []
    for cls_idx in range(len(class_names)):
        mask = all_labels == cls_idx
        if mask.sum() == 0:
            per_class_acc.append(0.0)
            continue
        acc = (all_preds[mask] == all_labels[mask]).mean() * 100
        per_class_acc.append(acc)

    # Sort theo accuracy
    sorted_idx  = np.argsort(per_class_acc)
    sorted_accs = [per_class_acc[i] for i in sorted_idx]
    sorted_names = [class_names[i].replace("_", " ") for i in sorted_idx]

    # Color by accuracy range
    colors = ["#e74c3c" if a < 60 else "#f39c12" if a < 80 else "#2ecc71"
              for a in sorted_accs]

    fig, ax = plt.subplots(figsize=(14, max(10, len(class_names) * 0.2)))
    ax.barh(range(len(sorted_names)), sorted_accs, color=colors, edgecolor="white", height=0.8)
    ax.axvline(x=np.mean(per_class_acc), color="navy", linestyle="--",
               linewidth=1.5, label=f"Mean: {np.mean(per_class_acc):.1f}%")
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=7)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy (sorted)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)

    # Thêm giá trị vào cuối bar
    for i, acc in enumerate(sorted_accs):
        ax.text(acc + 0.5, i, f"{acc:.0f}%", va="center", fontsize=5.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Per-class accuracy saved to {save_path}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate Food-101 Classifier")
    parser.add_argument("--split",      type=str, default="test",
                        choices=["val", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--checkpoint", type=str, default=CFG.best_model_path,
                        help="Path to model checkpoint")
    parser.add_argument("--no-plots",   action="store_true",
                        help="Skip generating plots (faster)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Device: {device}")
    print(f"[Evaluate] Split:  {args.split}")
    print(f"[Evaluate] Checkpoint: {args.checkpoint}")

    # ── Load data ─────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders()
    class_names = load_class_names()
    loader      = test_loader if args.split == "test" else val_loader

    # ── Load model ────────────────────────────────────────────────────────
    model = build_model(num_classes=len(class_names)).to(device)
    load_checkpoint(args.checkpoint, model, device)

    # ── Run evaluation ────────────────────────────────────────────────────
    results = run_evaluation(model, loader, device, collect_images=True, max_images=24)

    all_preds   = results["all_preds"]
    all_labels  = results["all_labels"]

    # ── Print report ──────────────────────────────────────────────────────
    metrics = print_classification_report(all_preds, all_labels, class_names)

    if args.no_plots:
        print("\n[Evaluate] Skipping plots (--no-plots flag).")
        return

    # ── Generate all plots ─────────────────────────────────────────────────
    print("\n[Evaluate] Generating plots...")

    # 1. Sample predictions grid
    plot_sample_predictions(
        images      = results["sample_images"],
        true_labels = results["sample_labels"],
        pred_labels = results["sample_preds"],
        pred_probs  = results["sample_probs"],
        class_names = class_names,
        save_path = os.path.join(CFG.output_dir, "predictions.png"),
        n_cols=6, n_rows=4,
    )

    # 2. Confusion matrix
    plot_confusion_matrix(
        all_preds, all_labels, class_names,
        save_path = os.path.join(CFG.output_dir, "confusion_matrix.png"),
        normalize=True,
    )

    # 3. Top confused class pairs
    plot_top_confused_classes(
        all_preds, all_labels, class_names,
        save_path = os.path.join(CFG.output_dir, "top_confused_classes.png"),
        top_n=20,
    )

    # 4. Per-class accuracy
    plot_per_class_accuracy(
        all_preds, all_labels, class_names,
        save_path = os.path.join(CFG.output_dir, "per_class_accuracy.png")
    )

    print(f"\n[Evaluate] All plots saved to outputs/")
    print(f"[Evaluate] Final accuracy: {metrics['overall_acc']:.2f}%")


if __name__ == "__main__":
    main()