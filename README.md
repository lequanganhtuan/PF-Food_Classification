# 🍔 Food-101 Image Classifier

**ResNet-50** fine-tuned on Food-101 — **85.3% Validation Accuracy** across **101 food categories**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio 6.0](https://img.shields.io/badge/Gradio-6.0-orange.svg)](https://gradio.app/)

**🔗 Live Demo:** [Hugging Face Spaces — Food Classification](https://huggingface.co/spaces/anhtuan2602/food_classification)

---

## Demo

> Upload a food image → get real-time prediction with confidence score.

![Food-101 Classifier Demo](demo.gif)

---

## Results

| Phase | Description | Val Acc | Test Acc | Trainable Params | Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 1** | Head only (frozen backbone) | 72.1% | — | ~4K | 35m |
| **Phase 2** | Full fine-tuning | **85.3%** | **84.7%** | 23M | 140m |

*ResNet-50 · Food-101 dataset · CUDA A100*

---

## Architecture & Training Strategy

### 2-Phase Approach

```mermaid
graph LR
    A[Food-101 Dataset] --> B[ResNet-50 Backbone]
    B --> C[Phase 1: Head Training<br/>10 epochs · LR 1e-3]
    C --> D[Phase 2: Full Fine-tuning<br/>10 epochs · LR 1e-5]
    D --> E[85.3% Val Acc]

    style C fill:#f96,stroke:#333,color:#000
    style D fill:#69f,stroke:#333,color:#000
```

### Phase 1 — Freeze Backbone, Train Head

Frozen all ResNet-50 layers and trained only the custom classification head for 10 epochs.

**Why:** The backbone already carries rich ImageNet representations. Training the full network immediately with a high LR risks **Catastrophic Forgetting** — destroying pretrained features before the head has stabilized.

**Result:** ~72% val accuracy with a stable head, ready for fine-tuning.

### Phase 2 — Full Fine-tuning

Unfrozen the entire backbone and continued training with `LR = 1e-5`.

**Why:** Once the head is stable, a very small LR allows the backbone to subtly adapt its high-level features to food-specific textures and shapes without diverging.

**Result:** Val accuracy reached **85.3%** with strong generalization (84.7% test).

---

## Key Optimizations

| Technique | Detail |
| :--- | :--- |
| **Mixed Precision (AMP)** | `torch.amp` — reduces VRAM, speeds up training |
| **Data Augmentation** | `RandomResizedCrop(224)` + `RandomHorizontalFlip` |
| **Lazy Loading** | Model loaded on first request — faster app startup |
| **Learning Rate Schedule** | `ReduceLROnPlateau` with patience=2 |

---

## Project Structure

```
food-classification/
├── app.py                  # Gradio web demo
├── train.py                # Training pipeline (2-phase)
├── inference.py            # Single-image prediction script
├── requirements.txt
├── outputs/
│   └── best_model.pth      # Trained model weights
├── samples/                # Sample images for testing
│   └── pizza.jpg
└── demo.gif
```

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/lequanganhtuan/PF-Food_Classification.git
cd PF-Food_Classification
pip install -r requirements.txt
```

### 2. Run Inference

```bash
python inference.py --image samples/pizza.jpg
```

```
Prediction: Pizza (98.4%) | Latency: 0.12s
```

> Model weights are loaded from `outputs/best_model.pth` automatically.

### 3. Launch Web Demo

```bash
python app.py
```

```
Running on local URL: http://127.0.0.1:7860
```

### 4. Retrain from Scratch *(optional)*

> Requires a CUDA-enabled GPU.

```bash
python train.py --epochs 20 --batch_size 32
```

---

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) — 101,000 images across 101 food categories (750 train / 250 test per class).

Loaded via `torchvision.datasets.Food101`.

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
gradio>=6.0.0
pillow>=9.0.0
```

---

## License

MIT © [Le Quang Anh Tuan](https://github.com/lequanganhtuan)
