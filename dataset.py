

import json
import math
from pathlib import Path
from typing import Tuple, Dict
import random

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import Food101

from config import CFG


# ─── Transforms ──────────────────────────────────────────────────────────────

def get_train_transforms() -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(
            CFG.img_size,
            scale=(0.7, 1.0),       #
            ratio=(0.75, 1.33),     
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
        ),
        T.RandomGrayscale(p=0.05),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],  
            std=[0.229, 0.224, 0.225],    
        ),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
    ])


def get_val_transforms() -> T.Compose:
    return T.Compose([
        T.Resize(CFG.img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(CFG.img_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ─── Dataset & DataLoader factory ────────────────────────────────────────────

def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    print("[Dataset] Loading Food-101...")

    train_full = Food101(
        root=CFG.data_dir,
        split="train",
        download=True,
        transform=get_train_transforms(),
    )
    val_dataset_base = Food101(
        root=CFG.data_dir,
        split="train",
        download=False,                   
        transform=get_val_transforms(),   #
    )
    test_dataset = Food101(
        root=CFG.data_dir,
        split="test",
        download=False,
        transform=get_val_transforms(),
    )

    total_train = len(train_full)
    indices     = list(range(total_train))
    
    random.seed(CFG.seed)
    random.shuffle(indices)
    
    val_size    = int(total_train * CFG.val_split)
    train_size  = total_train - val_size

    train_indices = indices[:train_size]
    val_indices   = indices[train_size:]

    train_dataset = Subset(train_full, train_indices)
    val_dataset   = Subset(val_dataset_base, val_indices)

    _save_class_names(train_full.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,                  
        drop_last=True,                  
        persistent_workers=CFG.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size * 2,  
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=CFG.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=CFG.num_workers > 0,
    )

    print(f"[Dataset] Train samples : {len(train_dataset):,}")
    print(f"[Dataset] Val samples   : {len(val_dataset):,}")
    print(f"[Dataset] Test samples  : {len(test_dataset):,}")
    print(f"[Dataset] Num classes   : {len(train_full.classes)}")
    print(f"[Dataset] Train batches : {len(train_loader)}")
    print(f"[Dataset] Val batches   : {len(val_loader)}")

    return train_loader, val_loader, test_loader


def _save_class_names(classes: list) -> None:
    """Save class names list to JSON to json file for reused in inference."""
    save_path = Path(CFG.class_names_path)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)
    print(f"[Dataset] Class names saved to {save_path}")


def load_class_names() -> list:
    """Load classname"""
    with open(CFG.class_names_path, "r", encoding="utf-8") as f:
        return json.load(f)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])

    train_loader, val_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))
    print(f"\n[Sanity Check]")
    print(f"  Batch shape  : {images.shape}")      # Kỳ vọng: (32, 3, 224, 224)
    print(f"  Labels shape : {labels.shape}")      # Kỳ vọng: (32,)
    print(f"  Label range  : [{labels.min()}, {labels.max()}]")  # Kỳ vọng: [0, 100]
    print(f"  Pixel range  : [{images.min():.3f}, {images.max():.3f}]")

    class_names = load_class_names()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Training Batch — After Augmentation", fontsize=13)

    for i, ax in enumerate(axes.flat):
        img = images[i].numpy().transpose(1, 2, 0)
        img = np.clip(img * STD + MEAN, 0, 1)
        ax.imshow(img)
        ax.set_title(class_names[labels[i].item()].replace("_", " "), fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/sample_batch.png", dpi=120)
    print("\n[Sanity Check] Sample batch saved to outputs/sample_batch.png")
    plt.show()