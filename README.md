# 🏥 Medical Image Captioning with ViT-Base and Phi-2 using LoRA

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A parameter-efficient multimodal system for generating natural language descriptions of radiology images. Combines a **frozen ViT-Base** vision encoder with **Microsoft Phi-2 (2.7B)** language model, connected via a learned projection MLP and fine-tuned with **LoRA** on the ROCOv2 radiology dataset.

> **Trained on a single NVIDIA RTX 4090 (24 GB) in ~7.15 hours.** Captions are decoded greedily with 3-gram repetition blocking, over a 256-token training window chosen so reference captions keep their EOS token.

---

## Architecture

![Architecture and data flow diagram](docs/architecture.png)

The ViT-Base encoder stays frozen and emits 197 patch embeddings. A two-layer projection MLP lifts them to the Phi-2 embedding width and they're concatenated ahead of the tokenized prompt and caption. LoRA adapters sit on the query and value projections of the decoder; every other decoder weight stays frozen. Cross-entropy is computed on caption tokens only, with image tokens, the prompt prefix, and padding masked to −100.

**Design Highlights:**
- **LLaVA-style zero-initialization** — second projection layer starts at zero, preventing NaN from out-of-distribution visual embeddings
- **BFloat16 precision** — eliminates overflow risks inherent to float16 while halving memory vs float32
- **Manual FP32 loss computation** — logits are cast to float32 before cross-entropy to prevent gradient instability
- **Separate learning rates** — 2×10⁻³ for projection (learning from scratch) vs 2×10⁻⁴ for LoRA (fine-tuning)
- **256-token training window + 3-gram repetition blocking at decode time** — the two refinements described below, both reported in the accompanying paper

---

## Results

### Quantitative Evaluation (ROCOv2 Validation Set — 500 samples, 256-token window, 3-gram repetition blocking)

| Metric   | Score  |
|----------|--------|
| BLEU-1   | 0.1652 |
| BLEU-2   | 0.0917 |
| BLEU-3   | 0.0477 |
| BLEU-4   | 0.0270 |
| METEOR   | 0.1586 |
| ROUGE-L  | 0.1867 |
| CIDEr-D  | 0.1721 |

The decay from BLEU-1 to BLEU-4 fits the task: the model recovers the right clinical vocabulary far more reliably than the exact multi-word phrasing of a literature-derived reference.

### Comparison with ImageCLEFmedical 2024 systems

Kaprov, CS_Morgan, and DS@BioMed are systems from the ImageCLEFmedical 2024 caption-prediction task, which draws on the same ROCOv2 data. This is an indicative comparison, not a controlled head-to-head — different eval splits, sample sizes, and metric implementations (see caveats below the table).

| System | BLEU-1 | METEOR | ROUGE | CIDEr |
|---|---|---|---|---|
| Kaprov | 0.1697 | 0.0609 | 0.1905 | 0.1070 |
| CS_Morgan | **0.2093** | 0.0927 | **0.2508** | **0.2450** |
| DS@BioMed | 0.0121 | 0.0353 | 0.1031 | 0.0715 |
| **Proposed (this repo)** | 0.1652 | **0.1586** | 0.1867 | 0.1721 |

**Where this system stands:** it holds the highest METEOR of the four by a wide margin, sits roughly level with Kaprov on BLEU-1/ROUGE, trails the stronger CS_Morgan system on BLEU-1/ROUGE/CIDEr, and beats DS@BioMed on all four. The contribution here is efficiency — a single 24 GB consumer GPU, no multi-GPU cluster — rather than a broad accuracy win.

> **Caveats:** the CIDEr-D here is a from-scratch TF-IDF implementation with a Gaussian length penalty, not the reference `pycocoevalcap` code, so its absolute scale need not match published values. The comparison systems report on the ImageCLEF 2024 evaluation set with that task's own metric tooling and split, so treat this table as indicative rather than a controlled comparison. The ROUGE row is especially loose — this system reports ROUGE-L, the ImageCLEF task reports ROUGE-1.

### Sample Outputs

| Reference | Generated |
|-----------|-----------|
| Chest radiograph showing a recurrent right pneumothorax | Chest X-ray demonstrating right-sided pneumothorax with partial lung collapse |
| Chest X-ray showing bilateral clavicular hypoplasia | Frontal chest X-ray with bilateral shoulder girdle changes |
| Chest X-ray showing bilateral infiltrates worsened in the lower lung fields | Chest X-ray demonstrating bilateral lower lobe infiltrates |
| Control of chest-x-ray after cast rejection revealed complete left lung aeration | Chest radiograph showing left lung re-expansion |

These are selected examples (see the accompanying paper for the corresponding images), not a random draw — they sit beside the aggregate metrics rather than in place of them. The pattern holds across all four: the model names the modality and the broad finding, then hedges on the precise qualifier, and its captions run shorter than the references — which pulls BLEU-4 and CIDEr down relative to BLEU-1 and METEOR.

---

## Configuration Notes

Two refinements sit on top of the original single-stage setup:

| Change | What it does | Why |
|---|---|---|
| `max_len` 128 → 256 | Gives longer captions room to reach the EOS token during training | At 128, the data loader flagged captions truncated before EOS, which trains the decoder without stop supervision on those samples and encourages run-on output. 256 clears this for all but a short tail of very long captions. |
| `no_repeat_ngram=3` at eval/inference | Blocks the decoder from repeating any 3-gram during greedy decoding | Decode-time only, no retraining cost. Earlier settings produced visibly repetitive captions; this directly suppresses that pattern. |

Because `max_len` changes the training inputs, this is a genuinely different configuration from a 128-token run, not a rerun of the same experiment — it's reported as its own result rather than replacing an earlier one.

### Mode collapse

Under single-stage joint training, the decoder can drift toward generic captions for ambiguous inputs, with a subset of outputs repeating a small set of safe descriptions — the decoder minimizes loss through language statistics before the projection learns to carry visual signal. This inflates metrics that reward common word overlap (METEOR, above all) more than metrics that reward specific content, which lines up with this system's METEOR standing relative to its BLEU-1/ROUGE-L/CIDEr standing against the strongest comparison system above. The 256-token window and repetition blocking reduce the symptom; neither touches the underlying cause, which is the gradient race between the fast text-prior pathway and the slow visual-grounding pathway.

A grounding diagnostic (`forward_step_with_zeroed_images` / `HallucinationCounter` in `train.py`) scores every validation batch twice — once normally, once with the projected visual embeddings zeroed out. The gap between the two losses is logged each epoch; a gap near zero or negative means the decoder is largely ignoring the image.

---

## Key Technical Findings

These practical insights emerged during development and may help others building medical MLLMs:

1. **NaN Loss Prevention** — Never pass `labels` directly to the model's forward method in mixed precision. Instead: extract logits → cast to float32 → compute `F.cross_entropy` manually.

2. **Projection Initialization** — Xavier init on layer 1 + zero-init on layer 2 (no LayerNorm) prevents projection-related NaN instability during mixed-precision training.

3. **PEFT-Device Map Conflict** — Apply LoRA *before* moving the model to GPU. Applying LoRA after `device_map="auto"` causes parameter placement conflicts.

4. **BFloat16 > Float16** — On supported GPUs (RTX 30xx/40xx, A100, H100), bfloat16 matches float32's exponent range, eliminating the need for `GradScaler`.

5. **LoRA in FP32** — Promote LoRA adapter weights to float32 after model loading to prevent gradient underflow during training.

6. **Watch for EOS truncation** — If `max_len` is too short, some captions get cut off before their EOS token, which silently removes stop-token supervision from training and shows up later as run-on, repetitive generations. The dataset loader now warns once per worker when this happens.

7. **Decode-time repetition blocking is cheap insurance** — `no_repeat_ngram=3` at generation time costs nothing at training time and directly suppresses the repeated-phrase failure mode without needing to retrain or restructure the model.

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

By default, the HuggingFace dataset cache is written to `./hf_cache`. Set `HF_CACHE_DIR` to point it elsewhere (e.g. a persistent volume on cloud GPU platforms):

```bash
export HF_CACHE_DIR=/workspace/hf_cache
python train.py
```

The script will automatically:
- Download the ROCOv2-radiology dataset from HuggingFace
- Load ViT-Base and Phi-2 with LoRA configuration
- Train for 5 epochs with checkpointing
- Run inference on 10 samples and evaluate on 500 samples (both with and without `no_repeat_ngram` decoding)

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
├── docs/
│   └── architecture.png  # Architecture and data-flow diagram
└── paper/
    ├── README.md                      # Paper summary and citation
    └── MIC_Comparison_Paper_LNCS.docx # Full paper (Springer LNCS format)
```

> **Note:** Model weights and dataset are not included in this repository. The dataset is loaded automatically from HuggingFace (`eltorio/ROCOv2-radiology`), and checkpoints are saved locally during training.

---

## Hardware & Training Details

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 4090 (24 GB VRAM) |
| Platform | RunPod Cloud |
| Training Time | ~7.15 hours (5 epochs, max_len=256, full training split) |
| Effective Batch Size | 16 (bs=4 × grad_accum=4) |
| Trainable Parameters | ~1.57M (LoRA) + ~5.90M (Projection) |
| Dataset | ROCOv2-radiology — 79,793 total (59,962 train / 9,904 val / 9,927 test) |

---

## Acknowledgments

- [ROCOv2 Dataset](https://huggingface.co/datasets/eltorio/ROCOv2-radiology) — Radiology Objects in COntext
- [Microsoft Phi-2](https://huggingface.co/microsoft/phi-2) — Small language model
- [Google ViT-Base](https://huggingface.co/google/vit-base-patch16-224) — Vision Transformer
- [LoRA (Hu et al., 2022)](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation
- [LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485) — Projection design inspiration
- ImageCLEFmedical 2024 caption-prediction task — source of the Kaprov, CS_Morgan, and DS@BioMed comparison systems

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{musunuri2026medicalcaptioning,
  author    = {Musunuri Rana Rishith},
  title     = {Resource-Efficient Medical Image Captioning with a Frozen ViT-Base Encoder and Phi-2 under LoRA Fine-Tuning: A Comparative Study on ROCOv2},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/rana-rishith/medical-image-captioning}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

**Disclaimer:** This system is a research prototype and is **not** intended for clinical use. Generated captions should not be used for medical diagnosis or treatment decisions.
