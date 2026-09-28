# Resource-Efficient Medical Image Captioning with ViT-Base and Phi-2 (LoRA)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A radiology captioning system trained end to end on **one consumer GPU: an RTX 3060 with 12 GB of VRAM**. A frozen ViT-Base encoder feeds Microsoft Phi-2 (2.7B) through a small projection network, and Phi-2 is adapted with LoRA. Only **11.15M parameters train, 0.39% of the 2.88B in the full system**, and training peaks at **8.56 GiB of VRAM**.

The point of the project is the budget. Most published ROCOv2 captioning systems train on multi-GPU clusters or data-centre cards. This one runs on a card a student can buy.

| | |
|---|---|
| GPU | 1× NVIDIA RTX 3060, 12 GB |
| Peak VRAM (training) | 8.56 GiB |
| Trainable parameters | 11.15M (8.53M projection + 2.62M LoRA) |
| Frozen parameters | ViT-Base/16 85.8M, Phi-2 2.78B |
| Training time | ~25.5 h (~26.6 h including per-epoch validation) |
| Encoder cost during training | zero: features precomputed once |
| Test set | full official ROCOv2 test split, 9,927 images |

---

## Architecture

![v5 architecture](docs/architecture_v5.png)

Training runs in two parts.

**Stage 0, offline and one-time.** Every ROCOv2 image goes through a frozen ViT-Base/16 (`google/vit-base-patch16-224`, no pooler) once. The 197 × 768 patch embeddings land in an fp16 memory-mapped cache, 58,026 × 197 × 768 for the training split (about 17.6 GB on disk). From then on the encoder never runs during training: each step reads its features straight from disk through `np.load(mmap_mode="r")`. This one change removes 86M parameters of forward compute from every training step.

**Stage 1, online.** Cached features pass through a trainable projection (Linear 768→2560, GELU, Linear 2560→2560, LayerNorm) and a learned scalar gate. The 197 visual tokens are prepended to the tokenized prompt and caption, then fed to Phi-2. Phi-2's base weights stay frozen; LoRA (r=8, α=16, dropout 0.05) sits on `q_proj` and `v_proj` in all 32 decoder blocks. Image and prompt positions carry label −100, so the cross-entropy covers caption tokens only.

### What keeps it inside 12 GB

| Measure | Effect |
|---|---|
| Offline ViT feature cache | encoder weights and activations never touch the training step |
| Frozen Phi-2 + LoRA on q/v only | 2.62M adapter parameters instead of 2.78B |
| SDPA attention (eager rejected at load time) | no retained (B, H, T, T) score matrix per layer |
| Gradient checkpointing in both stages (`use_reentrant=False`) | activations recomputed instead of stored across 32 layers |
| Loss computed on gathered caption positions before the fp32 upcast | avoids upcasting the full (B, T, 51,200) logit tensor, roughly 250 MB per micro-batch |
| bf16 autocast where supported, fp16 + GradScaler otherwise | half-precision activations |
| Micro-batch 4 × gradient accumulation 8 | effective batch 32 on a 12 GB card |
| `expandable_segments` CUDA allocator | less fragmentation near the memory cap |

### Two-stage training

Joint single-stage training collapsed in earlier versions: the decoder learned to lower the loss from caption statistics alone before the projection learned to carry any visual signal, and the gate drifted toward zero. v5 splits the schedule:

- **Stage A (2 epochs).** Projection only, lr 1e-3. Phi-2 fully frozen, no LoRA. The gate starts at 1.0 and stays frozen, so the model cannot learn to mute the image.
- **Stage B (3 epochs).** LoRA attached (lr 5e-5), projection continues at lr 2e-4, gate becomes trainable. The optimizer and cosine schedule are rebuilt from scratch so Stage A's momentum does not carry into LoRA.

AdamW (β = 0.9, 0.95), weight decay 0.01, 3% warmup, gradient clipping at 1.0, label smoothing 0.05, seed 0.

### Integrity gates

The v4 collapse traced back to cached feature rows that did not match their captions. v5 refuses to train until alignment is proven:

1. The cache and a row-order manifest are written together, atomically, and tagged with a `cache_sig` hash. Any mismatch hard-fails.
2. 32 random images are re-encoded through the ViT and must match their cached rows at cosine > 0.995, with an explicit off-by-one check.
3. The projection must overfit 8 images and recover their captions before the real run starts. (It passed: 7/8 captions recovered, loss 3.90 → 0.29, gate rose from 1.00 to 1.14.)

---

## Results

All numbers below come from `pycocoevalcap` on the **full official ROCOv2 test split (9,927 images)**. Validation (9,903 images, 1,000-image subset per epoch) was used only to pick the checkpoint.

| | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|---|
| **v5 (with image)** | 0.1043 | 0.0561 | 0.0288 | 0.0164 | 0.0579 | 0.1667 | **0.1180** |
| Blind ablation (features zeroed) | 0.1276 | 0.0628 | 0.0257 | 0.0126 | 0.0423 | 0.1408 | 0.0511 |

### Is the model looking at the image?

The blind row runs the same checkpoint on the same 9,927 images with every visual feature set to zero. With the image, CIDEr more than doubles (0.0511 → 0.1180), and METEOR, ROUGE-L, BLEU-3 and BLEU-4 all rise.

BLEU-1 and BLEU-2 go the other way. Without an image, the decoder falls back on the most common radiology phrasing, and single common words score well on BLEU-1. CIDEr down-weights n-grams that appear across many references, so it rewards the image-specific content that the blind model cannot produce. That is why CIDEr is the metric to read here.

The loss-level grounding probe on 256 test images agrees:

| Probe | Gap |
|---|---|
| `zeros_gap` (loss with zeroed features − loss with real features) | +0.284 |
| `mismatch_gap` (loss with another image's features − loss with real features) | +0.120 |

Both are positive, so real images lower the loss, and the *right* image lowers it further than a wrong one. In v4 the zeros gap sat at −1.85.

### 500-image validation vs full test split

Earlier versions of this repo and the paper draft reported numbers on 500 randomly sampled validation images. Those are replaced by the full test split above.

| Evaluation | Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|---|---|
| 500 validation images (earlier) | single-stage, 256-token window | 0.1362 | 0.0733 | 0.0361 | 0.0187 | 0.0636 | 0.1638 | 0.1045 |
| **9,927 test images (current)** | **v5, two-stage** | **0.1043** | **0.0561** | **0.0288** | **0.0164** | **0.0579** | **0.1667** | **0.1180** |

Read the two rows as a change of protocol, not as a gain or a loss. Both the model and the evaluation set differ between them, so no single cause explains the movement. The full test split is the one to cite: it is 20× larger, it is held out from checkpoint selection, and it matches the split other ROCOv2 papers report on.

Both rows use `pycocoevalcap`. The figures in older versions of this README (BLEU-1 0.1652, METEOR 0.1586, CIDEr 0.1721) came from a from-scratch metric implementation and are withdrawn; they are not comparable with anything in this table.

### Comparison with DS@BioMed (ImageCLEFmedical 2024)

| System | BLEU-1 | METEOR | ROUGE | CIDEr | Hardware |
|---|---|---|---|---|---|
| DS@BioMed | 0.0121 | 0.0353 | 0.1031 | 0.0715 | — |
| **This work (v5)** | **0.1043** | **0.0579** | **0.1667** | **0.1180** | 1× RTX 3060, 12 GB |

v5 scores higher on all four. Treat this as indicative. DS@BioMed was scored on the ImageCLEFmedical 2024 test set (17,237 images) with that task's own tooling, not on the ROCOv2 test split, and the task reports ROUGE-1 where this work reports ROUGE-L. Stronger ImageCLEF 2024 systems, such as CS_Morgan, remain ahead of this work on BLEU, ROUGE and CIDEr. The contribution here is reaching this range on a single 12 GB card.

---

## Quick start

### Requirements

- Python 3.10+
- NVIDIA GPU with 12 GB VRAM or more (developed on an RTX 3060 12 GB)
- A Java runtime for `pycocoevalcap` (PTBTokenizer and METEOR need it). On Windows, install Temurin and check that `java -version` works in the shell.
- About 25 GB free disk for the feature cache (train, validation and test)

```bash
git clone https://github.com/rana-rishith/medical-image-captioning.git
cd medical-image-captioning
pip install -r requirements.txt
```

### Run

`train.py` is written in cell format (`# %%` markers), so it opens as a notebook in VS Code or Jupytext and also runs as a plain script.

Smoke test first. It uses 512 rows per split and one epoch per stage, and nothing it prints is reportable:

```bash
SMOKE=1 python train.py          # Linux / macOS
set SMOKE=1 && python train.py   # Windows cmd
```

Then the full run:

```bash
python train.py
```

Paths default to `C:\mic` and `C:\hf_cache` on Windows and `/workspace/mic` and `/workspace/hf_cache` elsewhere; change `root` and `hf_cache` in `Config` to move them. On Windows the DataLoader uses `num_workers=0`.

The script builds the feature cache on first run, runs both integrity gates, trains Stage A and Stage B, then writes `outputs_v5/final_results.json` with test metrics, the blind ablation and the grounding probe.

---

## Repository layout

```
medical-image-captioning/
├── train.py                          # v5 pipeline: cache, integrity gates, two-stage training, evaluation
├── requirements.txt
├── docs/
│   └── architecture_v5.png           # architecture and data-flow figure
├── legacy/
│   ├── train_v3_single_stage.py      # earlier single-stage script (superseded)
│   ├── smoke_test_v3.py              # smoke test for the single-stage script
│   └── architecture_v3.png           # earlier figure (RTX 4090 setup)
├── paper/
│   └── README.md
├── LICENSE
└── README.md
```

Model weights, the feature cache and the dataset are not in the repo. ROCOv2 downloads from HuggingFace (`eltorio/ROCOv2-radiology`) on first run.

---

## Dataset

ROCOv2 (`eltorio/ROCOv2-radiology`): 59,962 train, 9,904 validation and 9,927 test image–caption pairs from the biomedical literature. A caption-length filter (3–60 words) applies to the training split only, leaving 58,026 pairs. Validation keeps 9,903 images (one caption was empty after cleaning). Test is used in full.

---

## Acknowledgments

- [ROCOv2](https://huggingface.co/datasets/eltorio/ROCOv2-radiology): Radiology Objects in COntext, version 2
- [Microsoft Phi-2](https://huggingface.co/microsoft/phi-2)
- [Google ViT-Base/16](https://huggingface.co/google/vit-base-patch16-224)
- [LoRA (Hu et al., 2022)](https://arxiv.org/abs/2106.09685)
- [LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485), for the projection-prefix design
- ImageCLEFmedical 2024 caption-prediction task, source of the DS@BioMed comparison

---

## Citation

```bibtex
@misc{musunuri2026medicalcaptioning,
  author    = {Musunuri, Rana Rishith and Sharma, Nandani},
  title     = {Resource-Efficient Medical Image Captioning with a Frozen ViT-Base Encoder and Phi-2 under LoRA Fine-Tuning},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/rana-rishith/medical-image-captioning}
}
```

---

## License

MIT. See [LICENSE](LICENSE).

**Disclaimer:** this is a research prototype and is **not** for clinical use. Generated captions must not inform diagnosis or treatment.
