#!/usr/bin/env python3
"""
Medical Image Captioning with ViT-Base + Phi-2 + LoRA
=====================================================

Trains a multimodal medical image captioning system on the ROCOv2
radiology dataset. Combines a frozen ViT-Base vision encoder with
Microsoft Phi-2 (2.7B) language model, connected via a LLaVA-style
projection MLP and fine-tuned with LoRA.

Requirements:
    - NVIDIA GPU with >= 16 GB VRAM (tested on RTX 4090, 24 GB)
    - Python 3.10+
    - See requirements.txt for dependencies

Usage:
    python train.py

Author: Rana Rishith Musunuri
License: MIT
"""

import os
import gc
import math
import random
import warnings
import json
import time
import re
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    ViTModel,
    ViTImageProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from rouge_score import rouge_scorer

warnings.filterwarnings("ignore")
for res in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(res, quiet=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    """Training and model configuration."""

    # Model identifiers
    vit_name: str = "google/vit-base-patch16-224"
    phi2_name: str = "microsoft/phi-2"
    vit_dim: int = 768
    phi2_dim: int = 2560

    # LoRA hyperparameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj", "v_proj")

    # Training hyperparameters
    epochs: int = 5
    batch_size: int = 4
    grad_accum: int = 4          # effective batch size = 16
    lr_lora: float = 2e-4
    lr_proj: float = 2e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_len: int = 128
    prompt: str = "Caption this medical image: "

    # Paths and misc
    save_dir: str = "./checkpoints"
    seed: int = 42
    num_workers: int = 2
    eval_samples: int = 500
    infer_samples: int = 10


cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)

NUM_IMG_TOKENS = (224 // 16) ** 2 + 1  # 197 (196 patches + CLS)


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_mem_info() -> str:
    """Return current GPU memory usage string."""
    try:
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1e9
        return f"{used:.2f} / {total / 1e9:.2f} GB"
    except Exception:
        alloc = torch.cuda.memory_allocated() / 1e9
        return f"{alloc:.2f} GB allocated"


# ──────────────────────────────────────────────────────────────────────
#  Projection Module (LLaVA-style, zero-init last layer)
# ──────────────────────────────────────────────────────────────────────
class ProjectionMLP(nn.Module):
    """
    Two-layer MLP that projects ViT features (768-dim) to Phi-2's
    embedding space (2560-dim).

    The second linear layer is zero-initialized so that at step 0 the
    image tokens contribute nothing to the decoder. This prevents NaN
    instability from out-of-distribution embeddings hitting the LLM's
    attention layers.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x.to(self.fc1.weight.dtype))))


# ──────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────
class ROCOv2Dataset(Dataset):
    """PyTorch dataset wrapper for the ROCOv2-radiology HuggingFace dataset."""

    def __init__(self, hf_split, processor, tokenizer, max_len, prompt,
                 img_col="image", cap_col="caption"):
        self.data = hf_split
        self.proc = processor
        self.tok = tokenizer
        self.max_len = max_len
        self.prompt = prompt
        self.img_col = img_col
        self.cap_col = cap_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            row = self.data[idx]
            img = row[self.img_col]
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            img = img.convert("RGB")
            pixel_values = self.proc(
                images=img, return_tensors="pt"
            ).pixel_values.squeeze(0)
        except Exception:
            pixel_values = torch.zeros(3, 224, 224)
            row = {self.cap_col: "medical image"}

        caption = str(row[self.cap_col]).strip() or "medical image"
        text = self.prompt + caption + self.tok.eos_token

        enc = self.tok(
            text, max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        input_ids = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)

        prompt_enc = self.tok(
            self.prompt, add_special_tokens=False, return_tensors="pt"
        )
        prompt_len = prompt_enc.input_ids.shape[1]
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ──────────────────────────────────────────────────────────────────────
#  Forward Pass
# ──────────────────────────────────────────────────────────────────────
def forward_step(batch, vit, proj, lm, tok):
    """
    Compute the captioning loss for a single batch.

    Loss is computed manually in float32 to prevent numerical overflow
    that occurs when passing labels directly to the model in bfloat16.
    """
    pv = batch["pixel_values"].to(DEVICE, dtype=torch.bfloat16)
    ids = batch["input_ids"].to(DEVICE)
    amsk = batch["attention_mask"].to(DEVICE)
    labs = batch["labels"].to(DEVICE)
    bsz = pv.shape[0]

    with torch.no_grad():
        vis = vit(pixel_values=pv).last_hidden_state
    vis_proj = proj(vis)

    txt_emb = lm.get_input_embeddings()(ids)
    vis_proj = vis_proj.to(dtype=txt_emb.dtype)

    combined_emb = torch.cat([vis_proj, txt_emb], dim=1)
    img_mask = torch.ones(bsz, NUM_IMG_TOKENS, device=DEVICE, dtype=amsk.dtype)
    combined_mask = torch.cat([img_mask, amsk], dim=1)
    img_labels = torch.full(
        (bsz, NUM_IMG_TOKENS), -100, device=DEVICE, dtype=labs.dtype
    )
    combined_labels = torch.cat([img_labels, labs], dim=1)

    out = lm(inputs_embeds=combined_emb, attention_mask=combined_mask)

    # Manual FP32 loss computation for numerical stability
    shift_logits = out.logits[:, :-1, :].contiguous().float()
    shift_labels = combined_labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


# ──────────────────────────────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_caption(
    image: Image.Image, vit, proj, lm, tok, proc, max_new_tokens: int = 128
) -> str:
    """Generate a caption for a single medical image."""
    vit.eval()
    proj.eval()
    lm.eval()

    img = image.convert("RGB")
    pv = proc(images=img, return_tensors="pt").pixel_values.to(
        DEVICE, dtype=torch.bfloat16
    )

    vis = vit(pixel_values=pv).last_hidden_state
    vis_proj = proj(vis)

    prompt_ids = tok(
        cfg.prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(DEVICE)
    prompt_emb = lm.get_input_embeddings()(prompt_ids)
    vis_proj = vis_proj.to(dtype=prompt_emb.dtype)

    embeds = torch.cat([vis_proj, prompt_emb], dim=1)
    past = None
    generated_ids = []
    cur_embeds = embeds

    for _ in range(max_new_tokens):
        out = lm(inputs_embeds=cur_embeds, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        next_id = logits.argmax(dim=-1)

        if next_id.item() == tok.eos_token_id:
            break

        generated_ids.append(next_id.item())
        cur_embeds = lm.get_input_embeddings()(next_id.unsqueeze(0))

    return tok.decode(generated_ids, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────────
#  CIDEr-D Metric (implemented from scratch)
# ──────────────────────────────────────────────────────────────────────
def _tokenize(s):
    return re.sub(r"[^\w\s]", " ", s.lower()).split()


def _count_ngrams(tokens, n):
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _compute_tfidf(ngram_counts, ref_doc_freq, n_docs, n):
    vec = {}
    total = max(sum(ngram_counts.values()), 1)
    for ng, cnt in ngram_counts.items():
        tf = cnt / total
        df = ref_doc_freq[n].get(ng, 0)
        idf = math.log(max(1.0, (n_docs - df) / (1.0 + df))) if df > 0 else 0.0
        vec[ng] = tf * idf
    return vec


def _vec_norm(vec):
    return math.sqrt(sum(v * v for v in vec.values())) if vec else 0.0


def _vec_dot(v1, v2):
    return sum(v1[k] * v2.get(k, 0.0) for k in v1)


def compute_cider(
    references: List[str], hypotheses: List[str], n_range: int = 4
) -> float:
    """Compute CIDEr-D score between reference and hypothesis captions."""
    n_docs = len(references)
    if n_docs == 0:
        return 0.0

    ref_doc_freq = defaultdict(lambda: defaultdict(int))
    ref_tokens_list = [_tokenize(r) for r in references]
    for tokens in ref_tokens_list:
        for n in range(1, n_range + 1):
            seen = set()
            for ng in _count_ngrams(tokens, n):
                if ng not in seen:
                    ref_doc_freq[n][ng] += 1
                    seen.add(ng)

    scores = []
    for ref_tok, hyp in zip(ref_tokens_list, hypotheses):
        hyp_tok = _tokenize(hyp)
        score_n = []
        for n in range(1, n_range + 1):
            ref_ng = _count_ngrams(ref_tok, n)
            hyp_ng = _count_ngrams(hyp_tok, n)
            ref_vec = _compute_tfidf(ref_ng, ref_doc_freq, n_docs, n)
            hyp_vec = _compute_tfidf(hyp_ng, ref_doc_freq, n_docs, n)
            norm_r = _vec_norm(ref_vec)
            norm_h = _vec_norm(hyp_vec)
            cos = (
                _vec_dot(ref_vec, hyp_vec) / (norm_r * norm_h)
                if norm_r > 0 and norm_h > 0
                else 0.0
            )
            delta = len(hyp_tok) - len(ref_tok)
            penalty = math.exp(-(delta**2) / (2 * 6.0**2))
            score_n.append(cos * penalty * 10.0)
        scores.append(sum(score_n) / len(score_n))

    return sum(scores) / len(scores)


# ──────────────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────────────
def evaluate(val_hf, vit, proj, lm, tok, proc, img_col, cap_col, max_samples=500):
    """Evaluate the model on the validation set with standard NLG metrics."""
    vit.eval()
    proj.eval()
    lm.eval()

    n = min(max_samples, len(val_hf))
    indices = random.sample(range(len(val_hf)), n)

    references = []
    hypotheses = []

    print(f"\n  Evaluating on {n} samples ...")
    t0 = time.time()

    for i, idx in enumerate(indices):
        row = val_hf[idx]
        img = row[img_col]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        ref = str(row[cap_col]).strip()
        gen = generate_caption(img, vit, proj, lm, tok, proc)

        references.append(ref)
        hypotheses.append(gen)

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{n} done  ({(time.time() - t0) / 60:.1f}m)")
        torch.cuda.empty_cache()

    # BLEU
    smooth = SmoothingFunction().method1
    ref_tok = [[_tokenize(r)] for r in references]
    hyp_tok = [_tokenize(h) for h in hypotheses]
    bleu1 = corpus_bleu(
        ref_tok, hyp_tok, weights=(1, 0, 0, 0), smoothing_function=smooth
    )
    bleu4 = corpus_bleu(
        ref_tok, hyp_tok, weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth,
    )

    # METEOR
    meteor_scores = []
    for r, h in zip(references, hypotheses):
        try:
            ms = nltk_meteor([nltk.word_tokenize(r)], nltk.word_tokenize(h))
        except Exception:
            ms = 0.0
        meteor_scores.append(ms)
    meteor = np.mean(meteor_scores)

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [
        scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)
    ]
    rougeL = np.mean(rouge_scores)

    # CIDEr-D
    cider = compute_cider(references, hypotheses)

    elapsed = (time.time() - t0) / 60
    return {
        "BLEU-1": round(bleu1, 4),
        "BLEU-4": round(bleu4, 4),
        "METEOR": round(float(meteor), 4),
        "ROUGE-L": round(float(rougeL), 4),
        "CIDEr": round(cider, 4),
        "num_samples": n,
        "eval_time_min": round(elapsed, 1),
    }


# ──────────────────────────────────────────────────────────────────────
#  Model Loading
# ──────────────────────────────────────────────────────────────────────
def load_models(cfg: Config):
    """Load ViT, Phi-2 with LoRA, and the projection module."""
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n  GPU before loading: {gpu_mem_info()}")

    # Vision encoder (frozen)
    print("\n=== Loading ViT ===")
    vit_proc = ViTImageProcessor.from_pretrained(cfg.vit_name)
    vit_model = ViTModel.from_pretrained(
        cfg.vit_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    for p in vit_model.parameters():
        p.requires_grad = False
    print(
        f"  ViT loaded ({sum(p.numel() for p in vit_model.parameters()) / 1e6:.1f}M "
        f"params, frozen)"
    )

    # Language model
    print("\n=== Loading Phi-2 ===")
    phi2_config = AutoConfig.from_pretrained(cfg.phi2_name, trust_remote_code=True)
    if not hasattr(phi2_config, "pad_token_id") or phi2_config.pad_token_id is None:
        phi2_config.pad_token_id = 50256

    phi2_tok = AutoTokenizer.from_pretrained(cfg.phi2_name, trust_remote_code=True)
    if phi2_tok.pad_token is None:
        phi2_tok.pad_token = phi2_tok.eos_token
        phi2_tok.pad_token_id = phi2_tok.eos_token_id

    # Enable all SDPA backends (Flash Attention supported on RTX 30xx/40xx/A100+)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    phi2_model = AutoModelForCausalLM.from_pretrained(
        cfg.phi2_name,
        config=phi2_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    phi2_model.config.pad_token_id = phi2_tok.pad_token_id

    # Apply LoRA BEFORE moving to GPU to avoid PEFT-device_map conflicts
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets),
    )
    phi2_model = get_peft_model(phi2_model, lora_cfg)
    phi2_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    phi2_model = phi2_model.to(DEVICE)

    # Promote LoRA weights to float32 for training stability
    for name, p in phi2_model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    trainable = sum(p.numel() for p in phi2_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in phi2_model.parameters())
    print(
        f"  LoRA applied: {trainable / 1e6:.2f}M trainable / {total / 1e6:.1f}M total "
        f"({100 * trainable / total:.2f}%)"
    )

    # Projection module
    projection = ProjectionMLP(cfg.vit_dim, cfg.phi2_dim).to(DEVICE)
    print(
        f"  Projection: "
        f"{sum(p.numel() for p in projection.parameters()) / 1e6:.2f}M params"
    )
    print(f"  GPU total: {gpu_mem_info()}")

    return vit_model, vit_proc, phi2_model, phi2_tok, projection


# ──────────────────────────────────────────────────────────────────────
#  Training Loop
# ──────────────────────────────────────────────────────────────────────
def train(vit_model, vit_proc, phi2_model, phi2_tok, projection, train_loader,
          val_loader, val_hf, img_col, cap_col):
    """Full training loop with checkpointing and validation."""
    lora_params = [p for _, p in phi2_model.named_parameters() if p.requires_grad]
    proj_params_list = list(projection.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": cfg.lr_lora},
            {"params": proj_params_list, "lr": cfg.lr_proj},
        ],
        weight_decay=cfg.weight_decay,
    )

    total_steps = (len(train_loader) // cfg.grad_accum) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    resume_path = os.path.join(cfg.save_dir, "resume_checkpoint.pt")
    best_path = os.path.join(cfg.save_dir, "best_model.pt")
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    # Resume from checkpoint if available
    if os.path.exists(resume_path):
        print(f"\n  Resuming from {resume_path}")
        phi2_model.gradient_checkpointing_disable()
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        projection.load_state_dict(ckpt["projection"])
        phi2_model.load_state_dict(ckpt["lora_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_val_loss = ckpt["best_val_loss"]
        del ckpt
        gc.collect()
        torch.cuda.empty_cache()
        phi2_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print(
            f"  Resuming at epoch {start_epoch}, step {global_step}, "
            f"best_val={best_val_loss:.4f}\n"
        )

    for epoch in range(start_epoch, cfg.epochs):
        phi2_model.train()
        projection.train()
        epoch_loss = 0.0
        batch_count = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            loss = forward_step(batch, vit_model, projection, phi2_model, phi2_tok)
            loss = loss / cfg.grad_accum

            if torch.isnan(loss):
                print(f"  [Epoch {epoch + 1}] NaN at step {step} — skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            epoch_loss += loss.item() * cfg.grad_accum
            batch_count += 1

            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(phi2_model.parameters()) + list(projection.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 200 == 0:
                    avg = epoch_loss / max(1, batch_count)
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    print(
                        f"  [E{epoch + 1}] step {global_step:>6d}  "
                        f"loss={avg:.4f}  lr={lr_now:.2e}  "
                        f"time={elapsed / 60:.1f}m"
                    )

                if global_step % 1000 == 0:
                    torch.save(
                        {
                            "projection": projection.state_dict(),
                            "lora_state": phi2_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_val_loss": best_val_loss,
                        },
                        resume_path,
                    )
                    print(f"    Checkpoint saved (step {global_step})")

        avg_train_loss = epoch_loss / max(1, batch_count)
        train_time = (time.time() - t0) / 60
        print(
            f"\n  ── Epoch {epoch + 1}/{cfg.epochs} ──  "
            f"train_loss={avg_train_loss:.4f}  time={train_time:.1f}m"
        )

        # Validation
        phi2_model.eval()
        projection.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                loss = forward_step(
                    batch, vit_model, projection, phi2_model, phi2_tok
                )
                if not torch.isnan(loss):
                    val_loss_sum += loss.item()
                    val_count += 1

        avg_val_loss = val_loss_sum / max(1, val_count)
        print(f"  val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "projection": projection.state_dict(),
                    "lora_state": phi2_model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": avg_val_loss,
                },
                best_path,
            )
            print(f"  ★ Best model saved (val_loss={avg_val_loss:.4f})")

        torch.save(
            {
                "projection": projection.state_dict(),
                "lora_state": phi2_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
            },
            resume_path,
        )
        print(f"  Resume checkpoint saved (next start: epoch {epoch + 1})")
        torch.cuda.empty_cache()
        gc.collect()

    print("\n  Training complete.\n")
    return best_path


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def main():
    seed_everything(cfg.seed)

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
            if hasattr(torch.cuda.get_device_properties(0), "total_mem")
            else f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Load models
    vit_model, vit_proc, phi2_model, phi2_tok, projection = load_models(cfg)

    # Load dataset
    print("\n=== Loading ROCOv2 ===")
    ds = load_dataset("eltorio/ROCOv2-radiology")
    img_col, cap_col = "image", "caption"

    train_hf = ds["train"]
    val_hf = ds["validation"] if "validation" in ds else ds["test"]

    train_ds = ROCOv2Dataset(
        train_hf, vit_proc, phi2_tok, cfg.max_len, cfg.prompt, img_col, cap_col
    )
    val_ds = ROCOv2Dataset(
        val_hf, vit_proc, phi2_tok, cfg.max_len, cfg.prompt, img_col, cap_col
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    print(f"  Train: {len(train_ds)} samples → {len(train_loader)} batches/epoch")
    print(f"  Val  : {len(val_ds)} samples → {len(val_loader)} batches/epoch")

    # Train
    best_path = train(
        vit_model, vit_proc, phi2_model, phi2_tok, projection,
        train_loader, val_loader, val_hf, img_col, cap_col,
    )

    # Load best checkpoint
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        projection.load_state_dict(ckpt["projection"])
        phi2_model.load_state_dict(ckpt["lora_state"], strict=False)
        print(
            f"Loaded best checkpoint "
            f"(epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})"
        )

    # Sample inference
    print("\n" + "=" * 70)
    print("  SAMPLE INFERENCE OUTPUTS")
    print("=" * 70)
    sample_indices = random.sample(
        range(len(val_hf)), min(cfg.infer_samples, len(val_hf))
    )
    for i, idx in enumerate(sample_indices):
        row = val_hf[idx]
        img = row[img_col]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        ref = str(row[cap_col]).strip()
        gen = generate_caption(img, vit_model, projection, phi2_model, phi2_tok, vit_proc)
        print(f"\n  Sample {i + 1}")
        print(f"  REF: {ref}")
        print(f"  GEN: {gen}")
        torch.cuda.empty_cache()

    # Evaluation
    results = evaluate(
        val_hf, vit_model, projection, phi2_model, phi2_tok, vit_proc,
        img_col, cap_col, max_samples=cfg.eval_samples,
    )

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    for k, v in results.items():
        print(f"    {k:>15s} : {v}")
    print("=" * 50)

    results_path = os.path.join(cfg.save_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")
    print("\n  ✓ ALL DONE — training, inference, and evaluation complete.\n")


if __name__ == "__main__":
    main()
