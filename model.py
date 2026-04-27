from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from config import CFG


# ─── Model Builder ────────────────────────────────────────────────────────────

def build_model(num_classes: int = CFG.num_classes) -> nn.Module:
    print(f"[Model] Building ResNet-50 (pretrained={CFG.pretrained})...")

    weights = models.ResNet50_Weights.IMAGENET1K_V2 if CFG.pretrained else None
    model   = models.resnet50(weights=weights)

    in_features = model.fc.in_features  # 2048

    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=CFG.dropout),
        nn.Linear(in_features, num_classes),
    )

    nn.init.kaiming_uniform_(model.fc[2].weight, nonlinearity="relu")
    nn.init.zeros_(model.fc[2].bias)

    print(f"[Model] Classification head: {in_features} → {num_classes}")
    return model


# ─── Freeze / Unfreeze ────────────────────────────────────────────────────────

def freeze_backbone(model: nn.Module) -> None:
    # Freeze tất cả trước
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze chỉ riêng head
    for param in model.fc.parameters():
        param.requires_grad = True

    trainable = count_trainable_params(model)
    print(f"[Model] Backbone FROZEN. Trainable params: {trainable:,} (head only)")


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True

    trainable = count_trainable_params(model)
    print(f"[Model] ALL layers UNFROZEN. Trainable params: {trainable:,}")


def unfreeze_progressive(model: nn.Module, stage: int) -> None:
    for param in model.parameters():
        param.requires_grad = False
        
    layers_to_unfreeze = ["fc"]
    if stage >= 1:
        layers_to_unfreeze.append("layer4")
    if stage >= 2:
        layers_to_unfreeze.append("layer3")
    if stage >= 3:
        layers_to_unfreeze.append("layer2")
    if stage >= 4:
        layers_to_unfreeze.extend(["layer1", "bn1", "conv1"])

    for name, module in model.named_children():
        if name in layers_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True

    trainable = count_trainable_params(model)
    print(f"[Model] Progressive unfreeze stage {stage}. "
          f"Layers: {layers_to_unfreeze}. Trainable: {trainable:,}")


# ─── Optimizer Factory ────────────────────────────────────────────────────────

def get_optimizer_phase1(model: nn.Module) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params,
        lr=CFG.lr_phase1,
        weight_decay=CFG.weight_decay,
    )
    print(f"[Optimizer] Phase 1 — Adam, lr={CFG.lr_phase1}, "
          f"param groups: 1, params: {sum(p.numel() for p in params):,}")
    return optimizer


def get_optimizer_phase2(model: nn.Module) -> torch.optim.Optimizer:
    backbone_params = []
    head_params     = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc"):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": CFG.lr_backbone_phase2, "name": "backbone"},
        {"params": head_params,     "lr": CFG.lr_head_phase2,     "name": "head"},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=CFG.weight_decay,
    )
    print(f"[Optimizer] Phase 2 — AdamW, differential LR:")
    print(f"  Backbone: lr={CFG.lr_backbone_phase2}, params={len(backbone_params):,} tensors")
    print(f"  Head:     lr={CFG.lr_head_phase2},  params={len(head_params):,} tensors")
    return optimizer


# ─── Scheduler Factory ────────────────────────────────────────────────────────

def get_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int, num_steps_per_epoch: int):
    total_steps = num_epochs * num_steps_per_epoch
    max_lrs     = [pg["lr"] for pg in optimizer.param_groups]

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=CFG.warmup_epochs / num_epochs,   # % thời gian warmup
        anneal_strategy="cos",
        div_factor=25,                              # initial_lr = max_lr / 25
        final_div_factor=1e4,                       # final_lr = initial_lr / 1e4
    )
    print(f"[Scheduler] OneCycleLR — total_steps={total_steps}, "
          f"warmup={CFG.warmup_epochs} epochs, max_lrs={max_lrs}")
    return scheduler


# ─── Utilities ────────────────────────────────────────────────────────────────

def count_trainable_params(model: nn.Module) -> int:
    """Đếm số parameters đang được train (requires_grad=True)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    """Đếm tổng số parameters của model."""
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    val_acc:   float,
    path:      str,
    extra:     dict = None,
) -> None:
    checkpoint = {
        "epoch":      epoch,
        "val_acc":    val_acc,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "model_name":   CFG.model_name,
            "num_classes":  CFG.num_classes,
            "img_size":     CFG.img_size,
        },
    }
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, path)
    print(f"[Checkpoint] Saved to {path} (epoch={epoch}, val_acc={val_acc:.2f}%)")


def load_checkpoint(path: str, model: nn.Module, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Checkpoint] Loaded from {path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, "
          f"Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    return checkpoint


# ─── Quick test khi chạy trực tiếp ───────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Test] Device: {device}")

    model = build_model()

    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print(f"\n[Test] Input shape:  {dummy.shape}")
    print(f"[Test] Output shape: {out.shape}")  # Kỳ vọng: (2, 101)
    assert out.shape == (2, CFG.num_classes), "Output shape sai!"

    print(f"\n[Test] Total params: {count_total_params(model):,}")
    freeze_backbone(model)
    print(f"       After freeze:")
    print(f"       Trainable: {count_trainable_params(model):,}")
    unfreeze_all(model)
    print(f"       After unfreeze:")
    print(f"       Trainable: {count_trainable_params(model):,}")

    freeze_backbone(model)
    opt1 = get_optimizer_phase1(model)
    unfreeze_all(model)
    opt2 = get_optimizer_phase2(model)

    print("\n[Test] All tests PASSED ✓")