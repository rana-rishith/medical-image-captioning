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

