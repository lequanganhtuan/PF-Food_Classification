"""
app.py — Gradio web demo for Food-101 Classifier.

Run locally:
    python app.py

Deploy on Hugging Face:
    Push this repo to HF Spaces (auto-detect app.py)

Extra requirements:
    pip install gradio
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import os
import sys


import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image

from config import CFG
from inference import (
    load_model_for_inference,
    load_class_names,
    get_inference_transform,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

CFG.class_names_path = os.path.join(BASE_DIR, "outputs", "class_names.json")
CFG.history_path = os.path.join(BASE_DIR, "outputs", "history.json")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "best_model.pth")


# Global State
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
CLASS_NAMES = None


def _load_resources():
    """Load model and class names into global state (lazy loading)."""
    global MODEL, CLASS_NAMES

    if MODEL is not None:
        return True, ""

    try:
        CLASS_NAMES = load_class_names(CFG.class_names_path)
        MODEL = load_model_for_inference(MODEL_PATH, DEVICE)
        return True, ""
    except FileNotFoundError as e:
        return False, str(e)


# Prediction Function
def predict(image: Image.Image, top_k: int = 5) -> Tuple[Dict, str]:
    """
    Main function called by Gradio when the user uploads an image.

    Args:
        image: PIL Image from Gradio input
        top_k: Number of results to display

    Returns:
        Tuple:
            - label_confidences_dict: {class_name: confidence}
            - info_text: Markdown string
    """
    ok, err = _load_resources()
    if not ok:
        return {}, f"❌ **Error:** {err}"

    if image is None:
        return {}, "⬆️ Please upload a food image to start."

    # Ensure RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = get_inference_transform()
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=CFG.use_amp and DEVICE.type == "cuda"):
            logits = MODEL(tensor)
            probs = F.softmax(logits, dim=1)[0]

    top_k = min(int(top_k), len(CLASS_NAMES))
    top_probs, top_indices = probs.topk(top_k)

    label_conf = {}
    results = []

    for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        display_name = CLASS_NAMES[idx].replace("_", " ").title()
        label_conf[display_name] = round(prob, 4)
        results.append((display_name, round(prob * 100, 1)))

    # Build info text
    top1_name, top1_conf = results[0]

    confidence_level = (
        "High confidence" if top1_conf >= 80 else
        "Medium confidence" if top1_conf >= 50 else
        "Low confidence"
    )

    info_md = f"""
### Top Prediction: **{top1_name}**
**Confidence:** {top1_conf:.1f}%  
**Status:** {confidence_level}

---
**Top {len(results)} Predictions:**
"""

    for i, (name, conf) in enumerate(results, 1):
        medal = ["1", "2", "3"][i - 1] if i <= 3 else f"{i}."
        bar_filled = int(conf / 5)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        
        info_md += f"\n\n{medal} **{name}** — `{bar}` {conf:.1f}%"
        

    return label_conf, info_md


# UI Layout
def create_demo() -> gr.Blocks:
    """Create Gradio Blocks UI."""

    # Load class count
    try:
        class_names = load_class_names(CFG.class_names_path)
        total_classes = len(class_names)
    except FileNotFoundError:
        total_classes = 101

    # Load sample images
    sample_images = (
        sorted(Path("samples").glob("*.jpg"))[:6]
        if Path("samples").exists()
        else []
    )
    sample_images = [[str(p)] for p in sample_images]

    with gr.Blocks(
        title="Food-101 Image Classifier",
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="amber"),
        css="""
        .title-container { text-align: center; padding: 20px 0 10px 0; }
        .subtitle { color: #6b7280; text-align: center; margin-bottom: 20px; }
        .result-box { border-radius: 12px; }
        """,
    ) as demo:

        # Header
        with gr.Column(elem_classes="title-container"):
            gr.Markdown("# 🍔 Food-101 Image Classifier")
            gr.Markdown(
                f"Fine-tuned **ResNet-50** classifying **{total_classes} food categories** "
                f"with >85% validation accuracy.",
                elem_classes="subtitle",
            )

        # Main layout
        with gr.Row():

            # Input
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload food image",
                    height=350,
                )

                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Number of predictions",
                    )

                with gr.Row():
                    predict_btn = gr.Button("Analyze", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                

            # Output
            with gr.Column(scale=1):
                label_output = gr.Label(
                    label="Results (confidence)",
                    num_top_classes=10,
                    elem_classes="result-box",
                )

                info_output = gr.Markdown(
                    value="⬆Upload an image and click **Analyze** to begin.",
                )

        # Examples
        if sample_images:
            gr.Markdown("### Sample Images")
            gr.Examples(
                examples=sample_images,
                inputs=[image_input],
                label="Click to try:",
                examples_per_page=6,
            )

        # Model info
        with gr.Accordion("ℹModel Information", open=False):
            try:
                with open(CFG.history_path) as f:
                    history = json.load(f)

                best_acc = history.get(
                    "best_val_acc_phase2",
                    history.get("test_acc_top1", "N/A")
                )
                test_acc = history.get("test_acc_top1", "N/A")

                info_text = f"""
| Info | Value |
|------|-------|
| **Model** | ResNet-50 (pre-trained on ImageNet) |
| **Dataset** | Food-101 (101,000 images, 101 classes) |
| **Validation Accuracy** | {best_acc:.2f}% |
| **Test Accuracy** | {test_acc:.2f}% |
| **Fine-tuning** | 2-phase: head → full backbone |
| **Input Size** | 224 × 224 pixels |
| **Device** | {str(DEVICE).upper()} |
                """
            except Exception:
                info_text = f"""
| Info | Value |
|------|-------|
| **Model** | ResNet-50 (pre-trained on ImageNet) |
| **Dataset** | Food-101 (101,000 images, 101 classes) |
| **Fine-tuning** | 2-phase: head → full backbone |
| **Input Size** | 224 × 224 pixels |
| **Device** | {str(DEVICE).upper()} |
                """

            gr.Markdown(info_text)
            
        with gr.Accordion("Food's List", open=False):
            try:
                food_list = sorted([name.replace("_", " ").title() for name in CLASS_NAMES])
                
                cols = 3
                rows = (len(food_list) + cols - 1) // cols
                table_md = "| | | |\n|---|---|---|\n" # Table header ẩn
                
                for i in range(rows):
                    row_str = "|"
                    for j in range(cols):
                        idx = i + j * rows
                        if idx < len(food_list):
                            row_str += f" {food_list[idx]} |"
                        else:
                            row_str += " |"
                    table_md += row_str + "\n"
                
                gr.Markdown(table_md)
            except:
                gr.Markdown("Loading...")

        # Event handlers
        def predict_wrapper(image, top_k):
            return predict(image, top_k)

        predict_btn.click(
            fn=predict_wrapper,
            inputs=[image_input, top_k_slider],
            outputs=[label_output, info_output],
            show_progress="minimal",
        )

        image_input.change(
            fn=predict_wrapper,
            inputs=[image_input, top_k_slider],
            outputs=[label_output, info_output],
            show_progress="minimal",
        )

        clear_btn.click(
            fn=lambda: (None, None, "Upload an image and click **Analyze** to begin."),
            outputs=[image_input, label_output, info_output],
        )

    return demo


# Main Entry
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Food-101 Gradio Demo")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Enable public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print("\nFood-101 Gradio Demo")
    print(f"   Device: {DEVICE}")

    # Preload model
    ok, err = _load_resources()
    if not ok:
        print(f"\nWarning: {err}")
        print("   Model will be loaded on the first request.")
    else:
        print("   Model loaded successfully ✓")

    demo = create_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )
    
