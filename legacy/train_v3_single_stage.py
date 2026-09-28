#!/usr/bin/env python3
"""
Medical Image Captioning with ViT-Base + Phi-2 + LoRA (Improved Single-Stage)
=============================================================================

Same architecture and training strategy as your original train_unified.py,
with two low-risk changes aimed at improving metrics WITHOUT changing the
training approach you already trust:

  CHANGE 1: max_len 128 -> 256
      Your run logged: "[WARN] max_len=128 truncated at least one caption
      before its EOS token". Truncated captions train the model with no
      stop-token supervision, which hurts BLEU/CIDEr and causes run-on
      repetition. 256 gives long captions room to reach EOS.

  CHANGE 2: no_repeat_ngram=3 at evaluation & inference
      Blocks the decoder from repeating any 3-gram during greedy decoding.
      This is a pure decode-time change (no retraining cost) that directly
      counters the repetitive-output symptom of mode collapse. Your original
      generate_caption() already supported this flag; here it is turned on
      for eval and sample inference.

Everything else is IDENTICAL to your original single-stage script, so the
comparison to your paper's numbers stays fair: same model, same data, same
optimizer, same epochs. Only max_len and decode-time repetition control
change.

NOTE ON HONESTY: raising max_len changes the training inputs, so these are
genuinely new results, not the same run. Report them as a separate
configuration. Neither change is guaranteed to beat your single-stage
numbers -- they address specific, observed failure signals (EOS truncation,
repetition), which is the principled reason to expect improvement.

Requirements:
    pip install datasets transformers peft torch torchvision pillow nltk rouge-score

Usage:
    python train_unified_improved.py

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
from typing import List, Dict, Any
from datetime import datetime

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
#  TrainingLogger
# ──────────────────────────────────────────────────────────────────────
class TrainingLogger:
    """Logs training runs to JSON files."""

    def __init__(self, save_dir: str = "./checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.log_file = os.path.join(save_dir, "training_log.json")
        self.error_file = os.path.join(save_dir, "errors.json")
        self.eval_file = os.path.join(save_dir, "evaluations.json")
        self.inference_file = os.path.join(save_dir, "inferences.json")

        self.logs = []
        self.errors = []
        self.evaluations = []
        self.inferences = []
        self.start_time = time.time()

    def log(self, message: str, level: str = "INFO"):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
        }
        self.logs.append(entry)
        print(f"[{level}] {message}")
        self._save_logs()

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  metrics: Dict[str, Any], nan_count: int = 0, skipped: int = 0):
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "metrics": metrics,
            "nan_count": nan_count,
            "skipped_batches": skipped,
            "timestamp": datetime.now().isoformat(),
        }
        self.logs.append(entry)
        self._save_logs()

    def log_evaluation(self, metrics: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        self.evaluations.append(entry)
        self._save_evals()

    def log_inference_sample(self, sample_idx: int, reference: str, hypothesis: str):
        entry = {
            "sample": sample_idx,
            "reference": reference,
            "hypothesis": hypothesis,
            "timestamp": datetime.now().isoformat(),
        }
        self.inferences.append(entry)
        self._save_inferences()

    def log_error(self, context: str, error_msg: str, context_info: str = ""):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error": error_msg,
            "context_info": context_info,
        }
        self.errors.append(entry)
        self._save_errors()

    def generate_final_report(self) -> str:
        elapsed = (time.time() - self.start_time) / 3600
        report = {
            "total_time_hours": round(elapsed, 2),
            "total_logs": len(self.logs),
            "total_errors": len(self.errors),
            "total_evaluations": len(self.evaluations),
            "total_inferences": len(self.inferences),
            "completed_at": datetime.now().isoformat(),
        }
        report_file = os.path.join(self.save_dir, "final_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        return json.dumps(report, indent=2)

    def _save_logs(self):
        with open(self.log_file, "w") as f:
            json.dump(self.logs, f, indent=2)

    def _save_errors(self):
        with open(self.error_file, "w") as f:
            json.dump(self.errors, f, indent=2)

    def _save_evals(self):
        with open(self.eval_file, "w") as f:
            json.dump(self.evaluations, f, indent=2)

    def _save_inferences(self):
        with open(self.inference_file, "w") as f:
            json.dump(self.inferences, f, indent=2)


# ──────────────────────────────────────────────────────────────────────
#  HallucinationCounter + zeroed-image forward
# ──────────────────────────────────────────────────────────────────────
def forward_step_with_zeroed_images(batch, vit, proj, lm, tok, device):
    """Compute loss with image embeddings zeroed out (grounding diagnostic)."""
    pv = batch["pixel_values"].to(device, dtype=torch.bfloat16)
    ids = batch["input_ids"].to(device)
    amsk = batch["attention_mask"].to(device)
    labs = batch["labels"].to(device)
    bsz = pv.shape[0]

    NUM_IMG_TOKENS = (224 // 16) ** 2 + 1  # 197

    with torch.no_grad():
        vis = vit(pixel_values=pv).last_hidden_state

    vis_proj = torch.zeros_like(proj(vis))

    txt_emb = lm.get_input_embeddings()(ids)
    vis_proj = vis_proj.to(dtype=txt_emb.dtype)

    combined_emb = torch.cat([vis_proj, txt_emb], dim=1)
    img_mask = torch.ones(bsz, NUM_IMG_TOKENS, device=device, dtype=amsk.dtype)
    combined_mask = torch.cat([img_mask, amsk], dim=1)
    img_labels = torch.full((bsz, NUM_IMG_TOKENS), -100, device=device, dtype=labs.dtype)
    combined_labels = torch.cat([img_labels, labs], dim=1)

    out = lm(inputs_embeds=combined_emb, attention_mask=combined_mask)

    shift_logits = out.logits[:, :-1, :].contiguous().float()
    shift_labels = combined_labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


class HallucinationCounter:
    """Tracks mode collapse diagnostics via zeros-gap."""

    def __init__(self, threshold_gap: float = 0.5, threshold_repetition: float = 0.7):
        self.threshold_gap = threshold_gap
        self.threshold_repetition = threshold_repetition
        self.clear()

    def clear(self):
        self.gaps = []
        self.repetition_rates = []
        self.total_samples = 0

    def update(self, loss_real: float, loss_zeros: float):
        gap = loss_real - loss_zeros
        self.gaps.append(gap)
        self.total_samples += 1

    def add_repetition_rate(self, text: str):
        tokens = re.findall(r"\b\w+\b", text.lower())
        if len(tokens) == 0:
            return
        unique_ratio = len(set(tokens)) / len(tokens)
        self.repetition_rates.append(unique_ratio)

    def report(self) -> Dict[str, Any]:
        if len(self.gaps) == 0:
            return {
                "total_samples": 0,
                "hallucination_rate": 0.0,
                "zeros_gap_mean": 0.0,
                "zeros_gap_std": 0.0,
            }
        gaps_arr = np.array(self.gaps)
        halluc_rate = np.mean(gaps_arr < self.threshold_gap) * 100
        return {
            "total_samples": len(self.gaps),
            "hallucination_rate": round(halluc_rate, 1),
            "zeros_gap_mean": round(float(np.mean(gaps_arr)), 4),
            "zeros_gap_std": round(float(np.std(gaps_arr)), 4),
            "zeros_gap_min": round(float(np.min(gaps_arr)), 4),
            "zeros_gap_max": round(float(np.max(gaps_arr)), 4),
        }


# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    """Training and model configuration."""

    vit_name: str = "google/vit-base-patch16-224"
    phi2_name: str = "microsoft/phi-2"
    vit_dim: int = 768
    phi2_dim: int = 2560

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj", "v_proj")

    epochs: int = 5
    batch_size: int = 4
    grad_accum: int = 4          # effective batch size = 16
    lr_lora: float = 2e-4
    lr_proj: float = 2e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06

    # ── CHANGE 1: max_len 128 -> 256 (fixes EOS-truncation warning) ──
    max_len: int = 256

    # ── CHANGE 2: repetition control at decode time (0 = off) ──
    # Applied ONLY at eval/inference; training is unaffected. Blocks any
    # repeated 3-gram during greedy decoding.
    eval_no_repeat_ngram: int = 3

    prompt: str = "Caption this medical image: "

    save_dir: str = "./checkpoints_improved"
    seed: int = 42
    num_workers: int = 2
    eval_samples: int = 500
    infer_samples: int = 10


cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)

NUM_IMG_TOKENS = (224 // 16) ** 2 + 1  # 197


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_mem_info() -> str:
    try:
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1e9
        return f"{used:.2f} / {total / 1e9:.2f} GB"
    except Exception:
        alloc = torch.cuda.memory_allocated() / 1e9
        return f"{alloc:.2f} GB allocated"


# ──────────────────────────────────────────────────────────────────────
#  Projection Module
# ──────────────────────────────────────────────────────────────────────
class ProjectionMLP(nn.Module):
    """Two-layer MLP: ViT 768-dim -> Phi-2 2560-dim, zero-init last layer."""

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

    _warned_boundary = False
    _warned_eos_truncation = False

    def __init__(self, hf_split, processor, tokenizer, max_len, prompt,
                 img_col="image", cap_col="caption"):
        self.data = hf_split
        self.proc = processor
        self.tok = tokenizer
        self.max_len = max_len
        self.prompt = prompt
        self.img_col = img_col
        self.cap_col = cap_col

        self._prompt_ids = self.tok(
            self.prompt, add_special_tokens=False
        ).input_ids

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

        prompt_len = len(self._prompt_ids)
        labels = input_ids.clone()

        if not ROCOv2Dataset._warned_boundary:
            actual_prefix = input_ids[:prompt_len].tolist()
            if actual_prefix != self._prompt_ids:
                print(
                    "[WARN] Prompt/caption tokenization boundary mismatch "
                    "detected. Label masking may be off by a token or two "
                    "for affected samples. (printed once per worker)"
                )
                ROCOv2Dataset._warned_boundary = True

        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        if not ROCOv2Dataset._warned_eos_truncation:
            if attention_mask.sum().item() == self.max_len:
                last_id = input_ids[self.max_len - 1].item()
                if last_id != self.tok.eos_token_id:
                    print(
                        f"[WARN] max_len={self.max_len} truncated at least "
                        f"one caption before its EOS token. If this is still "
                        f"common at 256, a few very long captions remain; "
                        f"that is expected and low-impact. (printed once "
                        f"per worker)"
                    )
                    ROCOv2Dataset._warned_eos_truncation = True

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
    """Compute the captioning loss for a single batch (FP32 loss)."""
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
    img_labels = torch.full((bsz, NUM_IMG_TOKENS), -100, device=DEVICE, dtype=labs.dtype)
    combined_labels = torch.cat([img_labels, labs], dim=1)

    out = lm(inputs_embeds=combined_emb, attention_mask=combined_mask)

    shift_logits = out.logits[:, :-1, :].contiguous().float()
    shift_labels = combined_labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


# ──────────────────────────────────────────────────────────────────────
#  Inference (with optional n-gram repetition blocking)
# ──────────────────────────────────────────────────────────────────────
def _banned_ngram_tokens(generated_ids, no_repeat_ngram):
    n = no_repeat_ngram
    if len(generated_ids) < n - 1:
        return set()
    prefix = tuple(generated_ids[-(n - 1):]) if n > 1 else tuple()
    banned = set()
    for i in range(len(generated_ids) - n + 1):
        if tuple(generated_ids[i:i + n - 1]) == prefix:
            banned.add(generated_ids[i + n - 1])
    return banned


@torch.no_grad()
def generate_caption(image, vit, proj, lm, tok, proc, max_new_tokens: int = 128,
                     no_repeat_ngram: int = 0) -> str:
    """Greedy caption generation, optional n-gram repetition blocking."""
    vit.eval()
    proj.eval()
    lm.eval()

    img = image.convert("RGB")
    pv = proc(images=img, return_tensors="pt").pixel_values.to(DEVICE, dtype=torch.bfloat16)

    vis = vit(pixel_values=pv).last_hidden_state
    vis_proj = proj(vis)

    prompt_ids = tok(cfg.prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)
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

        if no_repeat_ngram > 0:
            banned = _banned_ngram_tokens(generated_ids, no_repeat_ngram)
            for t in banned:
                logits[0, t] = float("-inf")

        next_id = logits.argmax(dim=-1)
        if next_id.item() == tok.eos_token_id:
            break
        generated_ids.append(next_id.item())
        cur_embeds = lm.get_input_embeddings()(next_id.unsqueeze(0))

    return tok.decode(generated_ids, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────────
#  CIDEr-D Metric
# ──────────────────────────────────────────────────────────────────────
def _tokenize(s):
    return re.sub(r"[^\w\s]", " ", s.lower()).split()


def _count_ngrams(tokens, n):
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


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


def compute_cider(references, hypotheses, n_range: int = 4) -> float:
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
            cos = (_vec_dot(ref_vec, hyp_vec) / (norm_r * norm_h)
                   if norm_r > 0 and norm_h > 0 else 0.0)
            delta = len(hyp_tok) - len(ref_tok)
            penalty = math.exp(-(delta ** 2) / (2 * 6.0 ** 2))
            score_n.append(cos * penalty * 10.0)
        scores.append(sum(score_n) / len(score_n))

    return sum(scores) / len(scores)


# ──────────────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────────────
def evaluate(val_hf, vit, proj, lm, tok, proc, img_col, cap_col,
             max_samples=500, logger=None, no_repeat_ngram=0):
    vit.eval()
    proj.eval()
    lm.eval()

    n = min(max_samples, len(val_hf))
    indices = random.sample(range(len(val_hf)), n)

    references = []
    hypotheses = []

    print(f"\n  Evaluating on {n} samples (no_repeat_ngram={no_repeat_ngram}) ...")
    t0 = time.time()

    for i, idx in enumerate(indices):
        try:
            row = val_hf[idx]
            img = row[img_col]
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            ref = str(row[cap_col]).strip()
            gen = generate_caption(img, vit, proj, lm, tok, proc,
                                   no_repeat_ngram=no_repeat_ngram)
            references.append(ref)
            hypotheses.append(gen)
        except Exception as e:
            if logger:
                logger.log_error("evaluate", str(e)[:80], context_info=f"sample_idx={idx}")

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{n} done  ({(time.time() - t0) / 60:.1f}m)")
        torch.cuda.empty_cache()

    smooth = SmoothingFunction().method1
    ref_tok = [[_tokenize(r)] for r in references]
    hyp_tok = [_tokenize(h) for h in hypotheses]
    bleu1 = corpus_bleu(ref_tok, hyp_tok, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(ref_tok, hyp_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(ref_tok, hyp_tok, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(ref_tok, hyp_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    meteor_scores = []
    for r, h in zip(references, hypotheses):
        try:
            ms = nltk_meteor([nltk.word_tokenize(r)], nltk.word_tokenize(h))
        except Exception:
            ms = 0.0
        meteor_scores.append(ms)
    meteor = np.mean(meteor_scores)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)]
    rougeL = np.mean(rouge_scores)

    cider = compute_cider(references, hypotheses)

    elapsed = (time.time() - t0) / 60

    results = {
        "BLEU-1": round(bleu1, 4),
        "BLEU-2": round(bleu2, 4),
        "BLEU-3": round(bleu3, 4),
        "BLEU-4": round(bleu4, 4),
        "METEOR": round(float(meteor), 4),
        "ROUGE-L": round(float(rougeL), 4),
        "CIDEr": round(cider, 4),
        "num_samples": n,
        "eval_time_min": round(elapsed, 1),
        "no_repeat_ngram": no_repeat_ngram,
        "references": references,
        "hypotheses": hypotheses,
    }

    if logger:
        eval_metrics = {k: v for k, v in results.items() if k not in ["references", "hypotheses"]}
        logger.log_evaluation(eval_metrics)

    return results


# ──────────────────────────────────────────────────────────────────────
#  Model Loading
# ──────────────────────────────────────────────────────────────────────
def load_models(cfg: Config):
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n  GPU before loading: {gpu_mem_info()}")

    print("\n=== Loading ViT ===")
    vit_proc = ViTImageProcessor.from_pretrained(cfg.vit_name)
    vit_model = ViTModel.from_pretrained(
        cfg.vit_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    for p in vit_model.parameters():
        p.requires_grad = False
    print(f"  ViT loaded ({sum(p.numel() for p in vit_model.parameters()) / 1e6:.1f}M params, frozen)")

    print("\n=== Loading Phi-2 ===")
    phi2_config = AutoConfig.from_pretrained(cfg.phi2_name, trust_remote_code=True)
    if not hasattr(phi2_config, "pad_token_id") or phi2_config.pad_token_id is None:
        phi2_config.pad_token_id = 50256

    phi2_tok = AutoTokenizer.from_pretrained(cfg.phi2_name, trust_remote_code=True)
    if phi2_tok.pad_token is None:
        phi2_tok.pad_token = phi2_tok.eos_token
        phi2_tok.pad_token_id = phi2_tok.eos_token_id

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

    for name, p in phi2_model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    trainable = sum(p.numel() for p in phi2_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in phi2_model.parameters())
    print(f"  LoRA applied: {trainable / 1e6:.2f}M trainable / {total / 1e6:.1f}M total "
          f"({100 * trainable / total:.2f}%)")

    projection = ProjectionMLP(cfg.vit_dim, cfg.phi2_dim).to(DEVICE)
    print(f"  Projection: {sum(p.numel() for p in projection.parameters()) / 1e6:.2f}M params")
    print(f"  GPU total: {gpu_mem_info()}")

    return vit_model, vit_proc, phi2_model, phi2_tok, projection


# ──────────────────────────────────────────────────────────────────────
#  Training Loop (single-stage, identical to original)
# ──────────────────────────────────────────────────────────────────────
def train(vit_model, vit_proc, phi2_model, phi2_tok, projection, train_loader,
          val_loader, val_hf, img_col, cap_col, hallucination_counter, logger):
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
        print(f"  Resuming at epoch {start_epoch}, step {global_step}, "
              f"best_val={best_val_loss:.4f}\n")

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

                # FIX: log by global_step so it actually prints (the original
                # gated on a condition that could never fire with grad_accum=4).
                if global_step % 200 == 0:
                    avg = epoch_loss / max(1, batch_count)
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    print(f"  [E{epoch + 1}] step {global_step:>6d}  "
                          f"loss={avg:.4f}  lr={lr_now:.2e}  time={elapsed / 60:.1f}m")

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
        print(f"\n  ── Epoch {epoch + 1}/{cfg.epochs} ──  "
              f"train_loss={avg_train_loss:.4f}  time={train_time:.1f}m")

        # Validation
        phi2_model.eval()
        projection.eval()
        val_loss_sum = 0.0
        val_count = 0
        hallucination_counter.clear()
        nan_count = 0
        zeros_gaps = []

        with torch.no_grad():
            for batch in val_loader:
                try:
                    loss_real = forward_step(batch, vit_model, projection, phi2_model, phi2_tok)
                    loss_zeros = forward_step_with_zeroed_images(
                        batch, vit_model, projection, phi2_model, phi2_tok, DEVICE
                    )
                    if not torch.isnan(loss_real):
                        val_loss_sum += loss_real.item()
                        val_count += 1
                    if not torch.isnan(loss_real) and not torch.isnan(loss_zeros):
                        zeros_gaps.append(loss_real.item() - loss_zeros.item())
                    if hasattr(hallucination_counter, "update"):
                        try:
                            hallucination_counter.update(loss_real.item(), loss_zeros.item())
                        except TypeError:
                            pass
                except Exception as e:
                    logger.log_error("validation_batch", str(e)[:80])
                    nan_count += 1

        if val_count == 0:
            print(f"  ⚠ Validation produced 0 usable batches out of "
                  f"{len(val_loader)} ({nan_count} failed) — skipping val_loss "
                  f"and best-checkpoint update this epoch.")
            avg_val_loss = float("inf")
        else:
            avg_val_loss = val_loss_sum / val_count
            suffix = f", {nan_count} failed)" if nan_count else ")"
            print(f"  val_loss={avg_val_loss:.4f}  "
                  f"({val_count}/{len(val_loader)} batches usable" + suffix)

        if zeros_gaps:
            gap_mean = float(np.mean(zeros_gaps))
            gap_std = float(np.std(zeros_gaps))
            print(f"  zeros_gap: mean={gap_mean:.4f} std={gap_std:.4f} "
                  f"(n={len(zeros_gaps)}) — near 0 or negative means the model "
                  f"is ignoring the image")

        hal_report = hallucination_counter.report()
        if hal_report.get("total_samples", 0) > 0:
            print(f"  hallucination_rate={hal_report['hallucination_rate']:.1f}% "
                  f"(gap={hal_report['zeros_gap_mean']:.3f}±{hal_report['zeros_gap_std']:.3f})")

        logger.log_epoch(epoch=epoch + 1, train_loss=avg_train_loss,
                         val_loss=avg_val_loss, metrics={}, nan_count=nan_count)

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
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"\n  max_len = {cfg.max_len}  (was 128 in original)")
    print(f"  eval_no_repeat_ngram = {cfg.eval_no_repeat_ngram}  (0 in original)\n")

    vit_model, vit_proc, phi2_model, phi2_tok, projection = load_models(cfg)

    logger = TrainingLogger(save_dir=cfg.save_dir)
    logger.log("Model loading complete", level="SUCCESS")

    hallucination_counter = HallucinationCounter(threshold_gap=0.5, threshold_repetition=0.7)

    print("\n=== Loading ROCOv2 ===")
    ds = load_dataset("eltorio/ROCOv2-radiology", cache_dir="/workspace/hf_cache")
    img_col, cap_col = "image", "caption"

    train_hf = ds["train"]
    val_hf = ds["validation"] if "validation" in ds else ds["test"]

    train_ds = ROCOv2Dataset(train_hf, vit_proc, phi2_tok, cfg.max_len, cfg.prompt, img_col, cap_col)
    val_ds = ROCOv2Dataset(val_hf, vit_proc, phi2_tok, cfg.max_len, cfg.prompt, img_col, cap_col)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    print(f"  Train: {len(train_ds)} samples → {len(train_loader)} batches/epoch")
    print(f"  Val  : {len(val_ds)} samples → {len(val_loader)} batches/epoch")

    best_path = train(vit_model, vit_proc, phi2_model, phi2_tok, projection,
                      train_loader, val_loader, val_hf, img_col, cap_col,
                      hallucination_counter, logger)

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        projection.load_state_dict(ckpt["projection"])
        phi2_model.load_state_dict(ckpt["lora_state"], strict=False)
        print(f"Loaded best checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    # Sample inference (with repetition blocking on)
    print("\n" + "=" * 70)
    print("  SAMPLE INFERENCE OUTPUTS")
    print("=" * 70)
    inference_results = []
    sample_indices = random.sample(range(len(val_hf)), min(cfg.infer_samples, len(val_hf)))
    for i, idx in enumerate(sample_indices):
        try:
            row = val_hf[idx]
            img = row[img_col]
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            ref = str(row[cap_col]).strip()
            gen = generate_caption(img, vit_model, projection, phi2_model, phi2_tok, vit_proc,
                                   no_repeat_ngram=cfg.eval_no_repeat_ngram)
            inference_results.append({"sample": i + 1, "idx": idx,
                                      "reference": ref, "hypothesis": gen, "error": None})
            logger.log_inference_sample(i + 1, ref, gen)
            print(f"\n  Sample {i + 1} (idx {idx})")
            print(f"  REF: {ref}")
            print(f"  GEN: {gen}")
        except Exception as e:
            error_msg = str(e)[:100]
            inference_results.append({"sample": i + 1, "idx": idx,
                                      "reference": None, "hypothesis": None, "error": error_msg})
            logger.log_error("sample_inference", error_msg, context_info=f"sample_idx={idx}")
            print(f"\n  Sample {i + 1}: ERROR — {error_msg}")
        torch.cuda.empty_cache()

    infer_path = os.path.join(cfg.save_dir, "sample_inferences.json")
    with open(infer_path, "w") as f:
        json.dump(inference_results, f, indent=2)
    logger.log(f"Inference results saved to {infer_path}", level="SUCCESS")
    print("=" * 70)

    # ── Evaluation: run BOTH with and without repetition blocking so you can
    #    report the honest delta and pick the config for your paper. ──
    print("\n=== Evaluation WITHOUT repetition blocking (matches original setup) ===")
    results_base = evaluate(val_hf, vit_model, projection, phi2_model, phi2_tok, vit_proc,
                            img_col, cap_col, max_samples=cfg.eval_samples,
                            logger=logger, no_repeat_ngram=0)

    print("\n=== Evaluation WITH no_repeat_ngram=3 ===")
    results_nr = evaluate(val_hf, vit_model, projection, phi2_model, phi2_tok, vit_proc,
                          img_col, cap_col, max_samples=cfg.eval_samples,
                          logger=logger, no_repeat_ngram=cfg.eval_no_repeat_ngram)

    def _print_results(tag, results):
        print("\n" + "=" * 50)
        print(f"  EVALUATION RESULTS — {tag}")
        print("=" * 50)
        for k, v in results.items():
            if k not in ["references", "hypotheses"]:
                print(f"    {k:>15s} : {v}")
        print("=" * 50)

    _print_results("max_len=256, no_repeat=0", results_base)
    _print_results("max_len=256, no_repeat=3", results_nr)

    # Honest side-by-side vs the paper's original single-stage numbers
    orig = {"BLEU-1": 0.1481, "BLEU-4": 0.0244, "METEOR": 0.1414,
            "ROUGE-L": 0.1791, "CIDEr": 0.1450}
    print("\n" + "=" * 70)
    print("  COMPARISON vs ORIGINAL PAPER (single-stage, max_len=128)")
    print("=" * 70)
    print(f"  {'Metric':>10s} | {'Original':>9s} | {'256/nr0':>9s} | {'256/nr3':>9s}")
    print("  " + "-" * 48)
    for m in ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]:
        o = orig[m]
        a = results_base.get(m, float("nan"))
        b = results_nr.get(m, float("nan"))
        print(f"  {m:>10s} | {o:>9.4f} | {a:>9.4f} | {b:>9.4f}")
    print("=" * 70)
    print("  NOTE: pick whichever config you report, and state max_len=256 +")
    print("        the no_repeat setting explicitly. These are new results,")
    print("        not the original run — report them as a new configuration.\n")

    results_path = os.path.join(cfg.save_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({"no_repeat_0": {k: v for k, v in results_base.items()
                                   if k not in ["references", "hypotheses"]},
                   "no_repeat_3": {k: v for k, v in results_nr.items()
                                   if k not in ["references", "hypotheses"]}},
                  f, indent=2)
    print(f"  Results saved to {results_path}")

    logger.generate_final_report()
    print("\n  ✓ ALL DONE.\n")


if __name__ == "__main__":
    main()
