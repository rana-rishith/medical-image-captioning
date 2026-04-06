# 🏥 Medical Image Captioning with ViT-Base and Phi-2 using LoRA

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A parameter-efficient multimodal system for generating natural language descriptions of radiology images. Combines a **frozen ViT-Base** vision encoder with **Microsoft Phi-2 (2.7B)** language model, connected via a learned projection MLP and fine-tuned with **LoRA** on the ROCOv2 radiology dataset.

> **Trained on a single NVIDIA RTX 4090 (24 GB) in ~5.2 hours.**

---

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────────┐
│   ViT-Base       │     │  Projection MLP       │     │  Phi-2 (2.7B) + LoRA     │
│   (86M, frozen)  │────▶│  Linear(768→2560)     │────▶│  LoRA on q_proj, v_proj   │
│   224×224 input  │     │  GELU                 │     │  r=8, α=16               │
│   197 tokens     │     │  Linear(2560→2560)    │     │  BFloat16 + FP32 LoRA    │
│   768-dim output │     │  Zero-init last layer │     │  Flash Attention (SDPA)   │
└─────────────────┘     └──────────────────────┘     └─────────────────────────┘
```

**Design Highlights:**
- **LLaVA-style zero-initialization** — second projection layer starts at zero, preventing NaN from out-of-distribution visual embeddings
- **BFloat16 precision** — eliminates overflow risks inherent to float16 while halving memory vs float32
- **Manual FP32 loss computation** — logits are cast to float32 before cross-entropy to prevent gradient instability
- **Separate learning rates** — 2×10⁻³ for projection (learning from scratch) vs 2×10⁻⁴ for LoRA (fine-tuning)

---

## Results

### Quantitative Evaluation (ROCOv2 Validation Set — 500 samples)

| Metric   | Score  |
|----------|--------|
| BLEU-1   | 0.1481 |
| BLEU-4   | 0.0244 |
| METEOR   | 0.1414 |
| ROUGE-L  | 0.1791 |
| CIDEr-D  | 0.1450 |

### Training Progression

| Epoch | Train Loss | Val Loss | Time/Epoch |
|-------|-----------|----------|------------|
| 1     | 2.6470    | 2.5218   | 62.3 min   |
| 2     | 2.5825    | 2.5061   | 62.4 min   |
| 3     | 2.5628    | 2.4654   | 62.3 min   |
| 4     | 2.4923    | 2.4244   | 62.2 min   |
| 5     | 2.4655    | 2.4167   | 62.4 min   |

Consistent loss decrease with a narrow train-val gap (~0.05) indicates stable learning without overfitting.

### Sample Outputs

| Reference | Generated |
|-----------|-----------|
| CT of abdomen revealing full-thickness pancreatic transection | CT scan of the abdomen and pelvis showing a large mass in the right lower quadrant |
| Lateral X-ray — fracture of the femur | X-ray of the right knee showing large osteophyte in medial condyle |
| Posteroanterior chest radiograph — rounded nodular opacities | Chest X-ray showing a large right pleural effusion |

The model correctly identifies imaging modalities (CT, X-ray, MRI) and generates anatomically plausible descriptions, though it exhibits mode collapse on ambiguous inputs — a known limitation discussed in the paper.

---

## Key Technical Findings

These practical insights emerged during development and may help others building medical MLLMs:

1. **NaN Loss Prevention** — Never pass `labels` directly to the model's forward method in mixed precision. Instead: extract logits → cast to float32 → compute `F.cross_entropy` manually.

2. **Projection Initialization** — Xavier init on layer 1 + zero-init on layer 2 (no LayerNorm) prevents projection-related NaN instability during mixed-precision training.

3. **PEFT-Device Map Conflict** — Apply LoRA *before* moving the model to GPU. Applying LoRA after `device_map="auto"` causes parameter placement conflicts.

4. **BFloat16 > Float16** — On supported GPUs (RTX 30xx/40xx, A100, H100), bfloat16 matches float32's exponent range, eliminating the need for `GradScaler`.

5. **LoRA in FP32** — Promote LoRA adapter weights to float32 after model loading to prevent gradient underflow during training.

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with ≥16 GB VRAM (tested on RTX 4090 24 GB)
- CUDA 11.8+

### Installation

```bash
git clone https://github.com/rana-rishith/medical-image-captioning.git
cd medical-image-captioning

pip install -r requirements.txt
```

### Training

```bash
python train.py
```

The script will automatically:
- Download the ROCOv2-radiology dataset from HuggingFace
- Load ViT-Base and Phi-2 with LoRA configuration
- Train for 5 epochs with checkpointing
- Run inference on 10 samples and evaluate on 500 samples

### Inference (after training)

```python
from PIL import Image
# Load your trained model checkpoint, then:
caption = generate_caption(Image.open("your_xray.png"))
print(caption)
```

---

## Project Structure

```
medical-image-captioning/
├── train.py              # Full training, inference, and evaluation pipeline
├── requirements.txt      # Python dependencies
├── README.md
├── LICENSE
└── paper/
    └── README.md         # Link to accompanying paper/report
```

> **Note:** Model weights and dataset are not included in this repository. The dataset is loaded automatically from HuggingFace (`eltorio/ROCOv2-radiology`), and checkpoints are saved locally during training.

---

## Hardware & Training Details

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 4090 (24 GB VRAM) |
| Platform | RunPod Cloud |
| Training Time | ~5.2 hours (5 epochs) |
| Effective Batch Size | 16 (bs=4 × grad_accum=4) |
| Trainable Parameters | ~1.57M (LoRA) + ~5.90M (Projection) |
| Dataset | ROCOv2-radiology (59,962 train / 9,904 val) |

---

## Acknowledgments

- [ROCOv2 Dataset](https://huggingface.co/datasets/eltorio/ROCOv2-radiology) — Radiology Objects in COntext
- [Microsoft Phi-2](https://huggingface.co/microsoft/phi-2) — Small language model
- [Google ViT-Base](https://huggingface.co/google/vit-base-patch16-224) — Vision Transformer
- [LoRA (Hu et al., 2022)](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation
- [LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485) — Projection design inspiration
- [PFMVG (He et al., 2025)](https://doi.org/10.1109/ISBI56570.2024.10635742) — Comparative reference for medical MLLM approaches

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{musunuri2025medicalcaptioning,
  author    = {Musunuri, Rana Rishith},
  title     = {Medical Image Captioning with ViT-Base and Phi-2 using LoRA Fine-Tuning},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/rana-rishith/medical-image-captioning}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

**Disclaimer:** This system is a research prototype and is **not** intended for clinical use. Generated captions should not be used for medical diagnosis or treatment decisions.
