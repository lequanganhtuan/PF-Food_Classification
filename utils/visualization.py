
import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ImageNet normalization parameters — dùng để denormalize khi hiển thị
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor_img):

    img = tensor_img.cpu().numpy().transpose(1, 2, 0)  # (C,H,W) → (H,W,C)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def plot_training_curves(history_path: str, save_path: str) -> None:

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    # ── Loss curve ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"],   "r-o", markersize=3, label="Val Loss",   linewidth=2)

    # Đánh dấu điểm best val loss
    best_epoch = np.argmin(history["val_loss"]) + 1
    best_loss  = min(history["val_loss"])
    ax.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax.annotate(f"{best_loss:.4f}", xy=(best_epoch, best_loss),
                xytext=(best_epoch + 0.5, best_loss + 0.05),
                fontsize=9, color="red")

    # Vẽ đường kẻ phân chia Phase 1 / Phase 2 nếu có trong history
    if "phase2_start_epoch" in history:
        p2 = history["phase2_start_epoch"]
        ax.axvline(x=p2, color="green", linestyle=":", alpha=0.8, linewidth=1.5)
        ax.text(p2 + 0.2, ax.get_ylim()[1] * 0.95, "Phase 2", color="green", fontsize=8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Accuracy curve ───────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc", linewidth=2)
    ax.plot(epochs, history["val_acc"],   "r-o", markersize=3, label="Val Acc",   linewidth=2)

    # Đánh dấu best val accuracy
    best_epoch_acc = np.argmax(history["val_acc"]) + 1
    best_acc       = max(history["val_acc"])
    ax.axvline(x=best_epoch_acc, color="gray", linestyle="--", alpha=0.7,
               label=f"Best epoch ({best_epoch_acc})")
    ax.annotate(f"{best_acc:.1f}%", xy=(best_epoch_acc, best_acc),
                xytext=(best_epoch_acc + 0.5, best_acc - 3),
                fontsize=9, color="red", fontweight="bold")

    if "phase2_start_epoch" in history:
        p2 = history["phase2_start_epoch"]
        ax.axvline(x=p2, color="green", linestyle=":", alpha=0.8, linewidth=1.5)
        ax.text(p2 + 0.2, 10, "Phase 2", color="green", fontsize=8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Training curves saved to {save_path}")


def plot_sample_predictions(
    images,
    true_labels: List[int],
    pred_labels: List[int],
    pred_probs:  List[float],
    class_names: List[str],
    save_path:   str,
    n_cols:      int = 4,
    n_rows:      int = 3,
) -> None:
    n = min(n_cols * n_rows, len(images))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
    fig.suptitle("Model Predictions (Green = Correct, Red = Wrong)", fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue

        img        = denormalize(images[i])
        pred_name  = class_names[pred_labels[i]].replace("_", " ").title()
        true_name  = class_names[true_labels[i]].replace("_", " ").title()
        correct    = pred_labels[i] == true_labels[i]
        color      = "#2ecc71" if correct else "#e74c3c"

        ax.imshow(img)
        ax.set_title(
            f"{'✓' if correct else '✗'} {pred_name}\n({pred_probs[i]:.1f}%)\nTrue: {true_name}",
            color=color,
            fontsize=8,
            fontweight="bold" if correct else "normal",
        )

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Predictions saved to {save_path}")


def plot_top_confused_classes(
    all_preds:   np.ndarray,
    all_labels:  np.ndarray,
    class_names: List[str],
    save_path:   str,
    top_n:       int = 15,
) -> None:
    # Tìm những trường hợp dự đoán sai
    wrong_mask   = all_preds != all_labels
    wrong_preds  = all_preds[wrong_mask]
    wrong_labels = all_labels[wrong_mask]

    # Đếm số lần nhầm cho từng cặp (true, predicted)
    confusion_pairs: Dict[tuple, int] = {}
    for true, pred in zip(wrong_labels, wrong_preds):
        pair = (int(true), int(pred))
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    # Sắp xếp theo số lần nhầm giảm dần
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:top_n]

    if not sorted_pairs:
        print("[Visualization] No errors found — perfect model!")
        return

    labels  = [f"{class_names[t]} → {class_names[p]}" for (t, p), _ in sorted_pairs]
    counts  = [c for _, c in sorted_pairs]
    colors  = plt.cm.Reds(np.linspace(0.4, 0.9, len(counts)))

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.5)))
    bars = ax.barh(range(len(labels)), counts, color=colors, edgecolor="white")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l.replace("_", " ") for l in labels], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Misclassifications")
    ax.set_title(f"Top {top_n} Most Confused Class Pairs\n(True Class → Predicted As)", fontsize=12, fontweight="bold")

    # Thêm số vào cuối mỗi bar
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", ha="left", fontsize=9, fontweight="bold")

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Confusion analysis saved to {save_path}")