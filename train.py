import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from config import CFG
from dataset import get_dataloaders, load_class_names
from model import (
    build_model,
    freeze_backbone,
    unfreeze_all,
    get_optimizer_phase1,
    get_optimizer_phase2,
    get_scheduler,
    save_checkpoint,
    count_trainable_params,
    count_total_params,
)
from utils.seed import set_seed
from utils.metrics import AverageMeter, accuracy, format_time
from utils.visualization import plot_training_curves


# ─── Training & Validation loops ─────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scheduler,
    scaler:    torch.cuda.amp.GradScaler,
    epoch:     int,
) -> tuple:
    model.train()

    loss_meter = AverageMeter("Loss")
    top1_meter = AverageMeter("Top1")
    top5_meter = AverageMeter("Top5")

    pbar = tqdm(loader, desc=f"  Train Epoch {epoch}", leave=False,
                unit="batch", colour="blue")

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True) 

        # Automatic Mixed Precision forward pass
        with torch.cuda.amp.autocast(enabled=CFG.use_amp):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        # Backward with gradient scaling (AMP)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Gradient clipping — tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Scheduler step theo batch (OneCycleLR)
        if scheduler is not None:
            scheduler.step()

        # Tính metrics
        with torch.no_grad():
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))

        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(top1, batch_size)
        top5_meter.update(top5, batch_size)

        # Update progress bar
        if batch_idx % CFG.log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss":  f"{loss_meter.avg:.4f}",
                "top1":  f"{top1_meter.avg:.1f}%",
                "lr":    f"{current_lr:.2e}",
            })

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int,
    split:     str = "Val",
) -> tuple:
    model.eval()

    loss_meter = AverageMeter("Loss")
    top1_meter = AverageMeter("Top1")
    top5_meter = AverageMeter("Top5")

    pbar = tqdm(loader, desc=f"  {split} Epoch {epoch}", leave=False,
                unit="batch", colour="green")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.cuda.amp.autocast(enabled=CFG.use_amp):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        top1, top5 = accuracy(outputs, labels, topk=(1, 5))
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(top1, batch_size)
        top5_meter.update(top5, batch_size)

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "top1": f"{top1_meter.avg:.1f}%",
            "top5": f"{top5_meter.avg:.1f}%",
        })

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


# ─── Training Phase ───────────────────────────────────────────────────────────

def run_phase(
    phase:        int,
    model:        nn.Module,
    train_loader,
    val_loader,
    criterion:    nn.Module,
    device:       torch.device,
    history:      dict,
    phase2_best_val_acc: float = 0.0,
) -> float:
    print(f"\n{'=' * 60}")
    print(f"  PHASE {phase}: {'HEAD TRAINING' if phase == 1 else 'FULL FINE-TUNING'}")
    print(f"{'=' * 60}")

    # Setup phase
    if phase == 1:
        freeze_backbone(model)
        optimizer = get_optimizer_phase1(model)
        epochs    = CFG.epoch_phase1
    else:
        unfreeze_all(model)
        optimizer = get_optimizer_phase2(model)
        epochs    = CFG.epochs_phase2
        history["phase2_start_epoch"] = len(history["train_loss"]) + 1

    scheduler = get_scheduler(optimizer, epochs, len(train_loader))
    scaler = torch.amp.GradScaler('cuda', enabled=CFG.use_amp)

    best_val_acc   = phase2_best_val_acc if phase == 2 else 0.0
    patience_count = 0
    phase_start    = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training
        train_loss, train_acc, train_top5 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler, scaler, epoch
        )

        # Validation
        val_loss, val_acc, val_top5 = validate(
            model, val_loader, criterion, device, epoch, split="Val"
        )

        epoch_time = time.time() - epoch_start

        # Ghi history
        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))

        # Log
        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% (Top5: {val_top5:.1f}%) | "
            f"Time: {format_time(epoch_time)}"
        )

        # Checkpoint — lưu nếu val accuracy cải thiện
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                path=CFG.best_model_path,
                extra={"phase": phase},
            )
            print(f"  ★ New best: {val_acc:.2f}% — checkpoint saved!")
        else:
            patience_count += 1
            if patience_count >= CFG.patience and phase == 2:
                print(f"\n  [EarlyStopping] No improvement for {CFG.patience} epochs. Stopping Phase 2.")
                break

    # Lưu last checkpoint
    save_checkpoint(model, optimizer, epoch, val_acc, path=CFG.last_model_path)

    phase_time = time.time() - phase_start
    print(f"\n  Phase {phase} completed in {format_time(phase_time)}")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")

    return best_val_acc


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():

    total_start = time.time()

    # ── Setup ─────────────────────────────────────────────────────────────
    set_seed(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Setup] Device: {device}")
    if device.type == "cuda":
        print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Setup] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"[Setup] AMP: {'Enabled' if CFG.use_amp else 'Disabled'}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders()

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model()
    model = model.to(device)

    print(f"\n[Model] Total params:     {count_total_params(model):,}")
    print(f"[Model] Trainable params: {count_trainable_params(model):,}")

    # ── Loss function ─────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing).to(device)

    # ── History tracking ──────────────────────────────────────────────────
    history = {
        "train_loss": [],
        "val_loss":   [],
        "train_acc":  [],
        "val_acc":    [],
    }

    # ── Phase 1: Head training ─────────────────────────────────────────────
    best_val_acc_p1 = run_phase(
        phase=1, model=model,
        train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, device=device, history=history,
    )

    # ── Phase 2: Full fine-tuning ──────────────────────────────────────────
    best_val_acc_p2 = run_phase(
        phase=2, model=model,
        train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, device=device, history=history,
        phase2_best_val_acc=best_val_acc_p1,
    )

    # ── Final test evaluation ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  FINAL TEST EVALUATION (best checkpoint)")
    print(f"{'=' * 60}")

    # Load best model để evaluate trên test set
    from model import load_checkpoint
    checkpoint = load_checkpoint(CFG.best_model_path, model, device)

    test_loss, test_acc, test_top5 = validate(
        model, test_loader, criterion, device, epoch=0, split="Test"
    )
    print(f"\n  Test Accuracy (Top-1): {test_acc:.2f}%")
    print(f"  Test Accuracy (Top-5): {test_top5:.2f}%")
    print(f"  Test Loss:             {test_loss:.4f}")

    # ── Lưu training history ───────────────────────────────────────────────
    history["best_val_acc_phase1"] = round(best_val_acc_p1, 4)
    history["best_val_acc_phase2"] = round(best_val_acc_p2, 4)
    history["test_acc_top1"]       = round(test_acc, 4)
    history["test_acc_top5"]       = round(test_top5, 4)
    history["total_epochs"]        = len(history["train_loss"])

    with open(CFG.history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[History] Saved to {CFG.history_path}")

    # ── Vẽ training curves ─────────────────────────────────────────────────
    plot_training_curves(
        history_path=CFG.history_path,
        save_path="outputs/training_curves.png",
    )

    # ── Tổng kết ───────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print("  TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Best Val Acc (Phase 1): {best_val_acc_p1:.2f}%")
    print(f"  Best Val Acc (Phase 2): {best_val_acc_p2:.2f}%")
    print(f"  Test Acc (Top-1):       {test_acc:.2f}%")
    print(f"  Test Acc (Top-5):       {test_top5:.2f}%")
    print(f"  Total Training Time:    {format_time(total_time)}")
    print(f"  Best checkpoint:        {CFG.best_model_path}")

    target_met = "✓ TARGET MET!" if best_val_acc_p2 >= 85.0 else "✗ Below 85% target"
    print(f"\n  {target_met}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()