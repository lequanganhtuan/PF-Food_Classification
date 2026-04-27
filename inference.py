"""
inference.py — CLI demo script for single image prediction.

Usage:
    # Predict one:
    python inference.py --image samples/pizza.jpg

    # Predict many:
    python inference.py --image samples/pizza.jpg samples/sushi.jpg

    # Change top predictions:
    python inference.py --image samples/burger.jpg --top-k 10

    # Use another checkpoint :
    python inference.py --image samples/pizza.jpg --checkpoint outputs/last_model.pth
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from config import CFG
from model import build_model

# Transform 
def get_inference_transform() -> T.Compose:
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(CFG.img_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
def load_model_for_inference(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load model from checkpoint, set eval mode.

    Args:
        checkpoint_path: path to file .pth
        device:          Device for load model

    Returns:
        Model ở eval mode, sẵn sàng inference.

    Raises:
        FileNotFoundError: if checkpoint doesnt exist.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please run 'python train.py' before for train model."
        )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint.get("config", {}).get("num_classes", CFG.num_classes)

    model = build_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[Model] Loaded from: {checkpoint_path}")
    print(f"        Trained epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"        Val accuracy:  {checkpoint.get('val_acc', 'N/A'):.2f}%")

    return model

# Predict
def load_class_names(class_names_path: str) -> list:
    """Load class names from JSON file."""
    path = Path(class_names_path)
    if not path.exists():
        raise FileNotFoundError(
            f"class_names.json not found : {class_names_path}\n"
            f"Please run 'python dataset.py' or 'python train.py' before."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
    
@torch.no_grad()
def predict_single(
    image_path:  str,
    model:       torch.nn.Module,
    class_names: list,
    device:      torch.device,
    top_k:       int = 5,
) -> list:
    """
    Predict class from one image, return top-K results.

    Args:
        image_path:  image path (jpg, png, webp, bmp...)
        model:       Model has loaded, ở eval mode
        class_names: List 101 classes
        device:      Device
        top_k:       Number of predictions need return

    Returns:
        List of dicts: [{"rank": 1, "class": "pizza", "probability": 89.3, "label_idx": 77}, ...]
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Ảnh không tìm thấy: {image_path}")

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Không thể đọc ảnh {image_path}: {e}")

    original_size = image.size  # (W, H)

    # Transform
    transform = get_inference_transform()
    tensor    = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Forward pass
    with torch.cuda.amp.autocast(enabled=CFG.use_amp and device.type == "cuda"):
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]  # (num_classes,)

    # Top-K
    top_k       = min(top_k, len(class_names))
    top_probs, top_indices = probs.topk(top_k)

    results = []
    for rank, (prob, idx) in enumerate(
        zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()), start=1
    ):
        results.append({
            "rank":        rank,
            "class":       class_names[idx],
            "class_display": class_names[idx].replace("_", " ").title(),
            "probability": round(prob * 100, 2),
            "label_idx":   idx,
        })

    return results

@torch.no_grad()
def predict_batch(
    image_paths: list,
    model:       torch.nn.Module,
    class_names: list,
    device:      torch.device,
    top_k:       int = 5,
) -> dict:
    transform = get_inference_transform()
    results   = {}

    batch_size = 16
    for i in range(0, len(image_paths), batch_size):
        batch_paths  = image_paths[i:i + batch_size]
        valid_paths  = []
        tensors      = []

        for path in batch_paths:
            try:
                img    = Image.open(path).convert("RGB")
                tensor = transform(img)
                tensors.append(tensor)
                valid_paths.append(path)
            except Exception as e:
                print(f"[Warning] Bỏ qua {path}: {e}")
                results[path] = {"error": str(e)}

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)

        with torch.amp.autocast('cuda', enabled=CFG.use_amp and device.type == "cuda"):
            logits = model(batch_tensor)
            probs  = F.softmax(logits, dim=1)

        for path, prob_vec in zip(valid_paths, probs):
            top_probs_b, top_indices_b = prob_vec.topk(min(top_k, len(class_names)))
            
            img_preds = []
            for rank, (p, idx_tensor) in enumerate(zip(top_probs_b, top_indices_b), start=1):
                idx = idx_tensor.item() 
                if isinstance(class_names, dict):
                    class_name_raw = class_names.get(str(idx), class_names.get(idx, f"ID_{idx}"))
                else:
                    class_name_raw = class_names[idx] if idx < len(class_names) else f"ID_{idx}"
                
                class_name_raw = str(class_name_raw) 
                
                img_preds.append({
                    "rank":          rank,
                    "class":         class_name_raw,
                    "class_display": class_name_raw.replace("_", " ").title(),
                    "probability":   round(p.item() * 100, 2),
                    "label_idx":     idx,
                })
            
            results[path] = img_preds

    return results

# Display

def print_predictions(image_path: str, predictions: list, show_bar: bool = True) -> None:
    """Print prediction result to terminal ưith format."""
    filename = Path(image_path).name
    width    = 60

    print(f"\n{'─' * width}")
    print(f"{filename}")
    print(f"{'─' * width}")

    for pred in predictions:
        rank    = pred["rank"]
        name    = pred["class_display"]
        prob    = pred["probability"]

        # Progress bar visual
        bar_len   = 30
        filled    = int(bar_len * prob / 100)
        bar     = "█" * filled + "░" * (bar_len - filled)
        medal     = ["1", "2", "3"][rank - 1] if rank <= 3 else f" {rank}."

        print(f"  {medal}  {name:<28}  {bar}  {prob:5.1f}%")

    print(f"{'─' * width}")
    
def main():
    parser = argparse.ArgumentParser(
        description="Car Image Classifier — Inference Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python inference.py --image samples/pizza.jpg
    python inference.py --image samples/pizza.jpg samples/sushi.jpg --top-k 10
    python inference.py --image samples/burger.jpg --checkpoint outputs/last_model.pth
            """,
        )
    parser.add_argument(
        "--image",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Image path for predict (có thể nhiều ảnh)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CFG.best_model_path,
        help=f" checkpoint path (default: {CFG.best_model_path})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number top predictions need display (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda', 'cpu', hoặc 'cuda:0' (default: auto-detect)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Json output",
    )
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.json:
        print(f"\n  Car Classifier — Inference")
        print(f"   Device: {device}")
        if device.type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ── Load model & class names ───────────────────────────────────────────
    try:
        model       = load_model_for_inference(args.checkpoint, device)
        class_names = load_class_names(CFG.class_names_path)
    except FileNotFoundError as e:
        print(f"\n[Error] {e}", file=sys.stderr)
        sys.exit(1)

    # ── Predict ───────────────────────────────────────────────────────────
    all_results = {}
    total_start = time.time()

    if len(args.image) == 1:
        # Single image
        try:
            preds = predict_single(args.image[0], model, class_names, device, top_k=args.top_k)
            all_results[args.image[0]] = preds
        except (FileNotFoundError, ValueError) as e:
            print(f"\n[Error] {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Batch
        all_results = predict_batch(args.image, model, class_names, device, top_k=args.top_k)

    elapsed = time.time() - total_start

    # ── Output ────────────────────────────────────────────────────────────
    if args.json:
        print(json.dumps(all_results, ensure_ascii=False, indent=2))
    else:
        for img_path, preds in all_results.items():
            if isinstance(preds, dict) and "error" in preds:
                print(f"\n[Error] {img_path}: {preds['error']}")
            else:
                print_predictions(img_path, preds)

        n_images = len([v for v in all_results.values() if not isinstance(v, dict) or "error" not in v])
        if n_images > 0:
            print(f"\n  ⏱  {n_images} image(s) in {elapsed:.2f}s "
                  f"({elapsed / n_images * 1000:.0f}ms/image)\n")


if __name__ == "__main__":
    main()