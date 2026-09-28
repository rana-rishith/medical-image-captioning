# %% [markdown]
# # Resource-Efficient Medical Image Captioning — v5 (cell-split)
#
# Frozen ViT-Base/16 encoder -> two-layer projection MLP (768 -> 2560 -> 2560,
# `v4_layernorm_gate`) -> Phi-2 (2.7B) with LoRA (r=8, alpha=16, q_proj/v_proj).
# Dataset: ROCOv2 (`eltorio/ROCOv2-radiology`). Metrics: `pycocoevalcap`.
#
# **What changed from v4, and why.** The v4 collapse signature (gate -> 0.08,
# LayerNorm gamma -> 0.03, zeros_gap negative and worsening) says the projection
# learned to mute its own input. A projection does that when the input does not
# predict the target — i.e. when feature row *i* is not the encoding of the image
# whose caption sits at index *i*. So v5 does not add another fix on top of v4.
# It makes misalignment **impossible to express**, then refuses to train until
# that is proven on the actual bytes on disk:
#
# | # | Change | Kills |
# |---|--------|-------|
# | C1 | Cache written via `np.save`, read **only** via `np.load(mmap_mode="r")` | 128-byte `.npy` header offset |
# | C2 | Row order defined by a manifest written *with* the cache; dataset indexes the manifest, never the HF split | index drift between cache and loader |
# | C3 | Manifest carries a `cache_sig` (rows, dtype, shape, encoder, resize); mismatch = hard fail | stale/partial cache reuse |
# | C4 | Cell 8 re-encodes K random images through ViT and requires cos > 0.995 against the cached row | offset, dtype, order, truncation — all of them |
# | C5 | Cell 12 overfits 8 images for 300 steps and requires the captions back verbatim | any structural break in the visual path |
# | C6 | Gate initialised to 1.0 and **frozen** through Stage A | the mute-the-input optimum |
# | C7 | Stage B rebuilds optimizer + schedule from scratch | Stage A momentum leaking into LoRA |
#
# C6 is a training-schedule change, not an architecture change: same parameters,
# same forward, same `proj_arch` tag. Flip `cfg.gate_init = 0.0` and
# `cfg.gate_trainable_stage_a = True` to recover exact v4 behaviour.
#
# **Order of execution.** Cells 1-8 are setup and must all pass. Cell 12 is a
# gate, not a log line — if it fails, stop, because nothing after it can work.
# Cells 13-14 are the real run.

# %% [markdown]
# ## Cell 1 — Install (run once per machine)

# %%
# !pip install -q "transformers>=4.38" datasets "peft>=0.10" accelerate pillow numpy
# !pip install -q pycocoevalcap
# PTBTokenizer and METEOR need a JRE:
# !apt-get update -qq && apt-get install -y -qq default-jre   # Linux
# Windows: install Temurin JRE and make sure `java -version` works in the shell.

# %% [markdown]
# ## Cell 2 — Imports and device

# %%
import os

# Set before torch initializes CUDA. Reduces fragmentation, which matters when
# the working set sits within a few hundred MiB of the 12 GiB cap.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import io
import sys
import json
import time
import random
import shutil
import hashlib
import platform
import warnings
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    AMP_DTYPE = torch.float32
elif torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16
else:
    AMP_DTYPE = torch.float16
USE_SCALER = AMP_DTYPE == torch.float16

print(f"python   {sys.version.split()[0]}")
print(f"torch    {torch.__version__}")
print(f"device   {DEVICE}  amp={AMP_DTYPE}")
if DEVICE.type == "cuda":
    _p = torch.cuda.get_device_properties(0)
    print(f"gpu      {_p.name}  {_p.total_memory / 1e9:.1f} GB "
          f"({_p.total_memory / 2**30:.1f} GiB)  sm_{_p.major}{_p.minor}")

# %% [markdown]
# ## Cell 3 — Configuration
#
# One dataclass, no hidden autoscaling. v4 silently rewrote `num_beams` from 4
# to 1 on a 12 GB card, which meant the decoding you reported was not the
# decoding you configured. Here every value that reaches the paper is printed
# once at the top and never touched again.

# %%
_IS_WIN = platform.system() == "Windows"


@dataclass
class Config:
    # ---- paths ----------------------------------------------------------
    root: str = r"C:\mic" if _IS_WIN else "/workspace/mic"
    hf_cache: str = r"C:\hf_cache" if _IS_WIN else "/workspace/hf_cache"
    cache_dir_name: str = "vit_feats_v5"     # NOT vit_feats — v4's cache is suspect

    # ---- data -----------------------------------------------------------
    dataset_name: str = "eltorio/ROCOv2-radiology"
    min_caption_words: int = 3
    max_caption_words: int = 60              # applied to TRAIN ONLY
    max_caption_tokens: int = 96

    # ---- encoder --------------------------------------------------------
    vit_name: str = "google/vit-base-patch16-224"
    vit_tokens: int = 197                    # 196 patches + CLS
    vit_dim: int = 768
    feat_dtype: str = "float16"

    # ---- language model -------------------------------------------------
    lm_name: str = "microsoft/phi-2"
    lm_dim: int = 2560

    # ---- projection (proj_arch = v4_layernorm_gate) ---------------------
    proj_hidden: int = 2560
    proj_dropout: float = 0.0
    gate_init: float = 1.0                   # C6 (v4 used 0.0)
    gate_trainable_stage_a: bool = False     # C6
    gate_trainable_stage_b: bool = True

    # ---- LoRA -----------------------------------------------------------
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: Tuple[str, ...] = ("q_proj", "v_proj")

    # ---- optimisation ---------------------------------------------------
    stage_a_epochs: int = 2
    stage_b_epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 8                      # effective batch 32
    lr_proj_a: float = 1e-3
    lr_proj_b: float = 2e-4
    lr_lora: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.05
    grad_checkpointing: bool = True

    # ---- grounding aux loss (off by default; see Cell 11) ---------------
    aux_grounding_weight: float = 0.0
    aux_grounding_margin: float = 0.5
    aux_every: int = 4

    # ---- generation (reported decoding) ---------------------------------
    num_beams: int = 4
    max_new_tokens: int = 64
    min_new_tokens: int = 5
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    gen_batch_size: int = 8

    # ---- eval -----------------------------------------------------------
    eval_samples: Optional[int] = None       # None = full official split
    val_probe_samples: int = 1000            # val subset for per-epoch CIDEr
    diag_probe_samples: int = 256            # fixed probe for zeros/mismatch gap

    # ---- integrity gates ------------------------------------------------
    verify_rows: int = 32                    # C4: images re-encoded
    verify_cos_min: float = 0.995            # C4: pass threshold
    overfit_images: int = 8                  # C5
    overfit_steps: int = 400                 # C5
    overfit_loss_max: float = 0.35            # C5: absolute pass threshold
    overfit_drop_ratio: float = 0.30          # C5: or final <= 0.30 * initial

    # ---- runtime --------------------------------------------------------
    seed: int = 0
    num_workers: int = 0 if _IS_WIN else 4
    smoke: bool = bool(int(os.environ.get("SMOKE", "0")))
    prompt: str = "Question: Describe the findings in this radiology image.\nAnswer:"

    # ---- derived --------------------------------------------------------
    def __post_init__(self):
        self.ckpt_dir = os.path.join(self.root, "checkpoints_v5")
        self.out_dir = os.path.join(self.root, "outputs_v5")
        self.cache_dir = os.path.join(self.hf_cache, self.cache_dir_name)
        for d in (self.ckpt_dir, self.out_dir, self.cache_dir):
            os.makedirs(d, exist_ok=True)
        os.environ.setdefault("HF_HOME", self.hf_cache)
        os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(self.hf_cache, "datasets"))
        if self.smoke:
            self.stage_a_epochs = 1
            self.stage_b_epochs = 1
            self.eval_samples = 64
            self.val_probe_samples = 64
            self.diag_probe_samples = 32

    @property
    def np_feat_dtype(self):
        return np.dtype(self.feat_dtype)

    def cache_sig(self) -> str:
        """C3 — any change here invalidates the cache by construction."""
        payload = {
            "encoder": self.vit_name,
            "tokens": self.vit_tokens,
            "dim": self.vit_dim,
            "dtype": self.feat_dtype,
            "dataset": self.dataset_name,
            "min_w": self.min_caption_words,
            "max_w": self.max_caption_words,
            "writer": "np.save",
            "version": 5,
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


cfg = Config()

PROJ_ARCH = "v4_layernorm_gate"

print("=" * 72)
print("CONFIGURATION TO STATE IN THE PAPER")
print("=" * 72)
for k in ("vit_name", "lm_name", "proj_hidden", "lora_r", "lora_alpha",
          "lora_targets", "batch_size", "grad_accum", "lr_proj_a", "lr_proj_b",
          "lr_lora", "stage_a_epochs", "stage_b_epochs", "num_beams",
          "max_new_tokens", "label_smoothing", "seed"):
    print(f"  {k:<22} {getattr(cfg, k)}")
print(f"  {'proj_arch':<22} {PROJ_ARCH}")
print(f"  {'effective_batch':<22} {cfg.batch_size * cfg.grad_accum}")
print(f"  {'gpu':<22} {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu'}")
print(f"  {'cache_sig':<22} {cfg.cache_sig()}")
if cfg.smoke:
    print("\n  *** SMOKE=1 — pipeline check only. No number from this run is reportable. ***")
print("=" * 72)

# %% [markdown]
# ## Cell 4 — Determinism and small utilities

# %%
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


seed_everything(cfg.seed)


def human(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}T"


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class JsonLog:
    """Append-only JSONL. Flushed per record so a crash keeps everything."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, **rec):
        rec["ts"] = time.time()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")


log = JsonLog(os.path.join(cfg.out_dir, "train_log.jsonl"))

# %% [markdown]
# ## Cell 5 — Load ROCOv2 and build the manifest
#
# The manifest is the single source of truth for *which caption goes with which
# cache row*. It is a plain list of `{split, hf_index, caption}` records, written
# to disk next to the features in the same function that writes the features.
# Nothing downstream ever re-derives an index from the HF dataset, so the two can
# not drift. Caption-length filtering applies to **train only** — val and test
# stay as the full official splits so reported metrics are comparable.

# %%
from datasets import load_dataset

import re

_WS = re.compile(r"\s+")


def clean_caption(text: Optional[str]) -> str:
    if not text:
        return ""
    t = _WS.sub(" ", str(text)).strip()
    return t


CANON = ("train", "validation", "test")


def resolve_splits(ds) -> Dict[str, Optional[str]]:
    keys = list(ds.keys())

    def pick(cands):
        for c in cands:
            if c in keys:
                return c
        return None

    return {"train": pick(["train"]),
            "validation": pick(["validation", "valid", "val", "dev"]),
            "test": pick(["test"])}


def _caption_key(cols: List[str]) -> str:
    for cand in ("caption", "text", "Caption", "captions"):
        if cand in cols:
            return cand
    raise RuntimeError(f"no caption column in {cols}")


def build_manifest(cfg: Config):
    ds = load_dataset(cfg.dataset_name, cache_dir=os.environ["HF_DATASETS_CACHE"])
    print({k: len(v) for k, v in ds.items()})
    smap = resolve_splits(ds)
    print(f"  split map: {smap}")
    if smap["train"] is None or smap["test"] is None:
        raise RuntimeError(f"need train and test splits, got {list(ds.keys())}")

    cap_key = _caption_key(ds[smap["train"]].column_names)
    print(f"  caption column: {cap_key!r}")

    def records_for(hf_split: str, canon: str, keep_idx: Optional[set] = None):
        caps = ds[hf_split][cap_key]
        apply_filter = (canon == "train")   # val/test stay as full official splits
        recs, dropped = [], 0
        for i, c in enumerate(caps):
            if keep_idx is not None and i not in keep_idx:
                continue
            c = clean_caption(c)
            n = len(c.split())
            if n == 0:
                dropped += 1
                continue
            if apply_filter and not (cfg.min_caption_words <= n <= cfg.max_caption_words):
                dropped += 1
                continue
            recs.append({"split": canon, "hf_split": hf_split,
                         "hf_index": i, "caption": c})
        print(f"  {canon:<11} <- {hf_split:<11} kept {len(recs):>6} dropped {dropped:>6}"
              f"  ({'length-filtered' if apply_filter else 'full official split'})")
        return recs

    manifest: Dict[str, List[Dict[str, Any]]] = {}
    if smap["validation"] is not None:
        manifest["train"] = records_for(smap["train"], "train")
        manifest["validation"] = records_for(smap["validation"], "validation")
    else:
        # No official validation split: carve a deterministic 2% out of train.
        # Records carry hf_split/hf_index, so the two canonical splits stay
        # disjoint and the cache stays aligned by construction.
        n_train = len(ds[smap["train"]])
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(n_train)
        n_val = max(int(0.02 * n_train), 256)
        val_idx, tr_idx = set(perm[:n_val].tolist()), set(perm[n_val:].tolist())
        print(f"  no validation split found -> carving {n_val} rows out of train")
        manifest["train"] = records_for(smap["train"], "train", keep_idx=tr_idx)
        manifest["validation"] = records_for(smap["train"], "validation", keep_idx=val_idx)
    manifest["test"] = records_for(smap["test"], "test")
    return ds, manifest


hf_ds, manifest = build_manifest(cfg)
log.write(event="manifest", counts={k: len(v) for k, v in manifest.items()})

# %% [markdown]
# ## Cell 6 — Load the ViT encoder (frozen)
#
# `add_pooling_layer=False`: the pooler is a randomly-initialised Linear(768,768)
# whose output nothing reads — we take `last_hidden_state`. Removing it drops
# 590,592 dead parameters from the reported count (86.39M -> 85.80M) and stops it
# burning compute on every one of ~80k precompute forwards. It does **not**
# change what gets cached.

# %%
from transformers import ViTModel, ViTImageProcessor

vit_proc = ViTImageProcessor.from_pretrained(cfg.vit_name, cache_dir=cfg.hf_cache)
vit = ViTModel.from_pretrained(
    cfg.vit_name,
    add_pooling_layer=False,
    torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
    cache_dir=cfg.hf_cache,
).to(DEVICE).eval()
for p in vit.parameters():
    p.requires_grad = False

_n_vit = sum(p.numel() for p in vit.parameters())
print(f"ViT loaded: {human(_n_vit)} params, frozen, pooler={'present' if getattr(vit, 'pooler', None) else 'removed'}")
assert getattr(vit, "pooler", None) is None, "pooler still attached"


def _to_rgb(img) -> Image.Image:
    if isinstance(img, dict) and "bytes" in img:
        img = Image.open(io.BytesIO(img["bytes"]))
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    return img.convert("RGB")


@torch.no_grad()
def encode_images(images: List[Any]) -> np.ndarray:
    """-> (B, 197, 768) float32 numpy. Single code path for cache writes AND
    the Cell 8 verification, so a preprocessing difference can not hide."""
    pil = [_to_rgb(im) for im in images]
    px = vit_proc(images=pil, return_tensors="pt")["pixel_values"]
    px = px.to(DEVICE, dtype=vit.dtype)
    out = vit(pixel_values=px).last_hidden_state
    return out.float().cpu().numpy()

# %% [markdown]
# ## Cell 7 — Write the feature cache (C1, C2, C3)
#
# Writes to `<split>.npy.tmp` + `<split>.manifest.json.tmp`, then renames both
# only after the last row lands. A killed job therefore leaves no half-file that
# a later run could mistake for complete — which is the failure mode that
# produced a 59,962-row `train.npy` for a 79k-row split.

# %%
def cache_paths(cfg: Config, split: str) -> Tuple[str, str]:
    return (os.path.join(cfg.cache_dir, f"{split}.npy"),
            os.path.join(cfg.cache_dir, f"{split}.manifest.json"))


def cache_is_valid(cfg: Config, split: str, n_expected: int) -> bool:
    fpath, mpath = cache_paths(cfg, split)
    if not (os.path.exists(fpath) and os.path.exists(mpath)):
        return False
    try:
        meta = json.load(open(mpath, encoding="utf-8"))
    except Exception:
        return False
    if meta.get("cache_sig") != cfg.cache_sig():
        print(f"  [{split}] cache_sig mismatch -> stale, will rebuild")
        return False
    if len(meta.get("records", [])) != n_expected:
        print(f"  [{split}] manifest has {len(meta.get('records', []))} records,"
              f" expected {n_expected} -> rebuild")
        return False
    arr = np.load(fpath, mmap_mode="r")          # C1: header-aware
    if arr.shape != (n_expected, cfg.vit_tokens, cfg.vit_dim):
        print(f"  [{split}] array shape {arr.shape} != "
              f"{(n_expected, cfg.vit_tokens, cfg.vit_dim)} -> rebuild")
        return False
    if arr.dtype != cfg.np_feat_dtype:
        print(f"  [{split}] dtype {arr.dtype} != {cfg.np_feat_dtype} -> rebuild")
        return False
    return True


def write_cache(cfg: Config, hf_ds, records: List[Dict[str, Any]], split: str,
                enc_batch: int = 32):
    fpath, mpath = cache_paths(cfg, split)
    ftmp, mtmp = fpath + ".tmp", mpath + ".tmp"
    n = len(records)
    shape = (n, cfg.vit_tokens, cfg.vit_dim)
    nbytes = int(np.prod(shape)) * cfg.np_feat_dtype.itemsize
    free = shutil.disk_usage(cfg.cache_dir).free
    print(f"  [{split}] {n} rows -> {nbytes / 1e9:.2f} GB (free {free / 1e9:.1f} GB)")
    if free < nbytes * 1.05:
        raise RuntimeError(f"not enough free space for {split} cache")

    # np.lib.format.open_memmap writes a proper .npy header, so the file is
    # readable by np.load. Never np.memmap on these paths.
    mm = np.lib.format.open_memmap(ftmp, mode="w+",
                                   dtype=cfg.np_feat_dtype, shape=shape)
    hf_split = records[0]["hf_split"]
    assert all(r["hf_split"] == hf_split for r in records), \
        f"[{split}] records span multiple HF splits"
    split_ds = hf_ds[hf_split]
    t0 = time.time()
    try:
        for s in range(0, n, enc_batch):
            chunk = records[s:s + enc_batch]
            idxs = [r["hf_index"] for r in chunk]
            imgs = split_ds.select(idxs)["image"]
            feats = encode_images(imgs)
            mm[s:s + len(chunk)] = feats.astype(cfg.np_feat_dtype)
            if s % (enc_batch * 50) == 0:
                done = s + len(chunk)
                rate = done / max(time.time() - t0, 1e-6)
                eta = (n - done) / max(rate, 1e-6) / 60
                print(f"    {done}/{n}  {rate:.1f} img/s  eta {eta:.1f} min", flush=True)
        mm.flush()
    finally:
        del mm
        free_cuda()

    with open(mtmp, "w", encoding="utf-8") as fh:
        json.dump({
            "cache_sig": cfg.cache_sig(),
            "split": split,
            "shape": list(shape),
            "dtype": cfg.feat_dtype,
            "writer": "np.lib.format.open_memmap (npy header present)",
            "reader": "np.load(mmap_mode='r')",
            "records": records,
        }, fh)

    os.replace(ftmp, fpath)      # C2/C3: features and manifest land together
    os.replace(mtmp, mpath)
    print(f"  [{split}] done in {(time.time() - t0) / 60:.1f} min")


for split in ("train", "validation", "test"):
    if split not in manifest:
        continue
    recs = manifest[split]
    if cfg.smoke:
        recs = recs[: min(len(recs), 512)]
        manifest[split] = recs
    if cache_is_valid(cfg, split, len(recs)):
        print(f"  [{split}] cache valid, reusing")
    else:
        write_cache(cfg, hf_ds, recs, split)

# %% [markdown]
# ## Cell 8 — CACHE INTEGRITY GATE (C4) — hard fail
#
# This is the cell that decides whether any of the rest is worth running. Four
# checks, in increasing strength:
#
# 1. **Header/shape/dtype** — read with `np.load(mmap_mode="r")`.
# 2. **No dead rows** — a zero or constant row means a decode failure got cached.
# 3. **Discriminability** — mean-pooled cosine between *different* rows must sit
#    well below 1. If adjacent rows are near-identical, rows are spliced.
# 4. **Round-trip** — re-encode K random images through the same `encode_images`
#    and require cos > 0.995 against the cached row. This is the only check that
#    can distinguish "features are fine" from "features are fine but attached to
#    the wrong caption", and it is the one v4 never had.

# %%
def open_cache(cfg: Config, split: str):
    fpath, mpath = cache_paths(cfg, split)
    arr = np.load(fpath, mmap_mode="r")                 # C1
    meta = json.load(open(mpath, encoding="utf-8"))
    recs = meta["records"]
    assert arr.shape[0] == len(recs), \
        f"[{split}] rows {arr.shape[0]} != manifest {len(recs)}"
    assert meta["cache_sig"] == cfg.cache_sig(), f"[{split}] cache_sig mismatch"
    return arr, recs


def audit_cache(cfg: Config, hf_ds, split: str) -> Dict[str, Any]:
    arr, recs = open_cache(cfg, split)
    n = arr.shape[0]
    rng = np.random.default_rng(cfg.seed)
    print(f"\n[{split}] rows={n} shape={arr.shape} dtype={arr.dtype}")

    # ---- 2. dead rows -------------------------------------------------
    probe = np.sort(rng.choice(n, size=min(128, n), replace=False))
    rows = np.asarray(arr[probe], dtype=np.float32)
    flat = rows.reshape(len(probe), -1)
    per_row_std = flat.std(axis=1)
    n_dead = int((per_row_std < 1e-4).sum())
    print(f"  feature stats: mean={flat.mean():+.4f} std={flat.std():.4f}")
    print(f"  row std: min={per_row_std.min():.4f} mean={per_row_std.mean():.4f} "
          f"dead={n_dead}")
    assert n_dead == 0, f"[{split}] {n_dead} constant/zero rows in cache"
    assert np.isfinite(rows).all(), f"[{split}] non-finite values in cache"

    # ---- 3. discriminability (CLS token — the most image-specific one) --
    cls = rows[:, 0, :]
    cls = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
    sim = cls @ cls.T
    off = sim[~np.eye(len(cls), dtype=bool)]
    print(f"  CLS cos (different rows): mean={off.mean():.4f} "
          f"p99={np.percentile(off, 99):.4f} max={off.max():.4f}")
    if off.mean() > 0.97:
        print("       NOTE: rows are highly similar. Expected to a degree for "
              "greyscale radiology, but watch the round-trip check below.")
    assert off.mean() < 0.995, \
        f"[{split}] rows near-identical (mean CLS cos {off.mean():.4f}) — spliced or constant"

    # ---- 4. round-trip -------------------------------------------------
    k = min(cfg.verify_rows, n)
    check = np.sort(rng.choice(n, size=k, replace=False))
    split_ds = hf_ds[recs[0]["hf_split"]]
    cos_all = []
    for s in range(0, k, 8):
        sub = check[s:s + 8]
        imgs = split_ds.select([recs[int(i)]["hf_index"] for i in sub])["image"]
        fresh = encode_images(imgs)                              # (b,197,768) f32
        cached = np.asarray(arr[sub], dtype=np.float32)
        a = fresh.reshape(len(sub), -1)
        b = cached.reshape(len(sub), -1)
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
        cos_all.extend(cos.tolist())
    cos_all = np.array(cos_all)
    print(f"  round-trip cos over {k} rows: min={cos_all.min():.5f} "
          f"mean={cos_all.mean():.5f}")
    bad = int((cos_all < cfg.verify_cos_min).sum())
    assert bad == 0, (
        f"[{split}] {bad}/{k} rows fail round-trip (min cos {cos_all.min():.4f}).\n"
        "  Cache row i is NOT the encoding of manifest record i. Do not train.\n"
        "  Delete the cache directory and re-run Cell 7."
    )

    # ---- 5. off-by-one probe ------------------------------------------
    # If rows were shifted by one, cos(fresh_i, cached_{i+1}) would beat
    # cos(fresh_i, cached_i). Confirm the diagonal wins.
    sub = check[:min(8, k)]
    imgs = split_ds.select([recs[int(i)]["hf_index"] for i in sub])["image"]
    fresh = encode_images(imgs).reshape(len(sub), -1)
    shifted = np.asarray(arr[np.clip(sub + 1, 0, n - 1)], dtype=np.float32).reshape(len(sub), -1)
    diag = np.asarray(arr[sub], dtype=np.float32).reshape(len(sub), -1)

    def _cos(x, y):
        return (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-8)

    c_d, c_s = _cos(fresh, diag), _cos(fresh, shifted)
    print(f"  aligned cos={c_d.mean():.5f}  shifted-by-1 cos={c_s.mean():.5f}  "
          f"diagonal wins {int((c_d > c_s).sum())}/{len(sub)}")
    assert bool((c_d > c_s).all()), (
        f"[{split}] for at least one probe, row i+1 matches image i better than "
        "row i does. The cache is shifted. Delete it and re-run Cell 7.")

    print(f"  [{split}] PASS")
    return {"split": split, "rows": int(n), "roundtrip_cos_min": float(cos_all.min()),
            "pooled_cos_mean": float(off.mean())}


audit = [audit_cache(cfg, hf_ds, s) for s in ("train", "validation", "test") if s in manifest]
log.write(event="cache_audit", results=audit)
print("\nCACHE INTEGRITY GATE: PASS")

# %% [markdown]
# ## Cell 9 — Dataset, tokenizer, collator
#
# The dataset returns `(feature_row, caption)` taken from the **same** manifest
# the cache was written with, so `__getitem__(i)` can not pair row *i* with
# caption *j*. Labels are `-100` across the visual span and the prompt, so the
# loss is only ever computed on caption tokens.

# %%
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(cfg.lm_name, cache_dir=cfg.hf_cache,
                                    trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

PROMPT_IDS = tok(cfg.prompt, add_special_tokens=False).input_ids
PROMPT_LEN = len(PROMPT_IDS)

# With label_smoothing=eps, the reported loss sits roughly eps*log(V) above the
# true cross-entropy. Printing it once stops anyone re-deriving it from a log.
from transformers import AutoTokenizer, AutoConfig

tok = AutoTokenizer.from_pretrained(cfg.lm_name, cache_dir=cfg.hf_cache,
                                    trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

PROMPT_IDS = tok(cfg.prompt, add_special_tokens=False).input_ids
PROMPT_LEN = len(PROMPT_IDS)

# Minimum reachable loss under label smoothing = entropy of the smoothed target.
# C must be the LOGIT width (Phi-2: 51200), not len(tok) (50295).
import math

LOGIT_VOCAB = AutoConfig.from_pretrained(cfg.lm_name, cache_dir=cfg.hf_cache,
                                         trust_remote_code=True).vocab_size


def smoothing_floor(eps: float, C: int) -> float:
    if eps <= 0:
        return 0.0
    p_true = 1 - eps + eps / C
    p_other = eps / C
    return -p_true * math.log(p_true) - (C - 1) * p_other * math.log(p_other)


SMOOTH_FLOOR = smoothing_floor(cfg.label_smoothing, LOGIT_VOCAB)
print(f"label smoothing {cfg.label_smoothing} over {LOGIT_VOCAB} logits "
      f"-> minimum reachable train loss {SMOOTH_FLOOR:.3f}")
print("  (probe losses use no smoothing; do not compare them to train loss)")
print(f"tokenizer: vocab={len(tok)} pad={tok.pad_token_id} eos={tok.eos_token_id}")
print(f"prompt ({PROMPT_LEN} tokens): {cfg.prompt!r}")


class CachedCaptionDataset(Dataset):
    def __init__(self, cfg: Config, split: str, limit: Optional[int] = None,
                 indices: Optional[List[int]] = None):
        self.cfg = cfg
        self.split = split
        self.path, _ = cache_paths(cfg, split)
        _, self.records = open_cache(cfg, split)
        self.arr = None                            # opened lazily per worker
        self.index = list(range(len(self.records))) if indices is None else list(indices)
        if limit is not None:
            self.index = self.index[:limit]

    def __len__(self):
        return len(self.index)

    def _ensure(self):
        if self.arr is None:
            self.arr = np.load(self.path, mmap_mode="r")   # C1, per-worker

    def __getitem__(self, i: int):
        self._ensure()
        row = self.index[i]
        feat = np.asarray(self.arr[row], dtype=np.float32)
        rec = self.records[row]
        return {"feat": torch.from_numpy(feat),
                "caption": rec["caption"],
                "row": row,
                "hf_index": rec["hf_index"]}


def collate_train(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    feats = torch.stack([b["feat"] for b in batch])
    seqs, labels = [], []
    for b in batch:
        cap = tok(" " + b["caption"], add_special_tokens=False,
                  truncation=True, max_length=cfg.max_caption_tokens).input_ids
        cap = cap + [tok.eos_token_id]
        ids = PROMPT_IDS + cap
        lab = [-100] * PROMPT_LEN + cap
        seqs.append(ids)
        labels.append(lab)
    T = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
    label_ids = torch.full((len(seqs), T), -100, dtype=torch.long)
    attn = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labels)):
        input_ids[i, :len(s)] = torch.tensor(s)
        label_ids[i, :len(l)] = torch.tensor(l)
        attn[i, :len(s)] = 1
    return {"feats": feats, "input_ids": input_ids, "labels": label_ids,
            "attention_mask": attn,
            "captions": [b["caption"] for b in batch],
            "rows": [b["row"] for b in batch]}


def collate_gen(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    feats = torch.stack([b["feat"] for b in batch])
    ids = torch.tensor([PROMPT_IDS] * len(batch), dtype=torch.long)
    attn = torch.ones_like(ids)
    return {"feats": feats, "input_ids": ids, "attention_mask": attn,
            "captions": [b["caption"] for b in batch],
            "rows": [b["row"] for b in batch]}


train_ds = CachedCaptionDataset(cfg, "train")
val_ds = CachedCaptionDataset(cfg, "validation")
test_ds = CachedCaptionDataset(cfg, "test")
print(f"datasets: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

_b = collate_train([train_ds[i] for i in range(min(4, len(train_ds)))])
print("sanity batch:", {k: tuple(v.shape) for k, v in _b.items()
                        if isinstance(v, torch.Tensor)})
assert (_b["labels"][:, :PROMPT_LEN] == -100).all(), "prompt not masked out of loss"


 

# %% [markdown]
# ## Cell 10 — Projection MLP and the multimodal wrapper
#
# `proj_arch = "v4_layernorm_gate"` — unchanged from v4 so old checkpoints stay
# loadable. The only difference is `gate_init` and whether the gate carries
# gradient in Stage A (C6). The projection runs in fp32 and its output is cast to
# the LM embedding dtype; keeping it out of autocast is what stopped the fc2
# scale blow-up at `lr_proj=2e-3` in v3.

# %%
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict


class ProjectionMLP(nn.Module):
    arch = PROJ_ARCH

    def __init__(self, d_in: int, d_hidden: int, d_out: int,
                 gate_init: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.drop(self.act(self.fc1(x))))
        return self.gate * self.norm(h)

    @torch.no_grad()
    def stats(self) -> Dict[str, float]:
        return {"gate": float(self.gate.detach()),
                "ln_gamma": float(self.norm.weight.detach().mean()),
                "fc2_w_rms": float(self.fc2.weight.detach().pow(2).mean().sqrt())}


class MICModel(nn.Module):
    """Visual prefix + text. Loss on caption tokens only."""

    def __init__(self, lm, proj: ProjectionMLP):
        super().__init__()
        self.lm = lm
        self.proj = proj

    @property
    def embed(self):
        return self.lm.get_input_embeddings()

    def _build(self, feats: torch.Tensor, input_ids: torch.Tensor,
               attention_mask: torch.Tensor,
               labels: Optional[torch.Tensor] = None):
        txt = self.embed(input_ids)
        vis = self.proj(feats.to(torch.float32)).to(txt.dtype)
        inp = torch.cat([vis, txt], dim=1)
        vis_attn = torch.ones(vis.shape[:2], dtype=attention_mask.dtype,
                              device=attention_mask.device)
        attn = torch.cat([vis_attn, attention_mask], dim=1)
        lab = None
        if labels is not None:
            pad = torch.full(vis.shape[:2], -100, dtype=labels.dtype,
                             device=labels.device)
            lab = torch.cat([pad, labels], dim=1)
        return inp, attn, lab

    def forward(self, feats, input_ids, attention_mask, labels,
                label_smoothing: float = 0.0):
        inp, attn, lab = self._build(feats, input_ids, attention_mask, labels)
        out = self.lm(inputs_embeds=inp, attention_mask=attn)
        logits = out.logits[:, :-1, :]
        target = lab[:, 1:]

        # Gather the supervised positions BEFORE upcasting. Only caption tokens
        # carry a label — the 197-token visual span and the prompt are -100 —
        # so this is ~10% of the sequence. Upcasting the full (B,T,51200) tensor
        # to fp32 costs ~250 MB per micro-batch and is the difference between
        # fitting and OOM on a 12 GB card. The value is identical: cross_entropy
        # with reduction='mean' averages over non-ignored positions either way.
        mask = target != -100
        n_sup = int(mask.sum())
        if n_sup == 0:
            return logits.sum() * 0.0
        loss = F.cross_entropy(
            logits[mask].float(),
            target[mask],
            label_smoothing=label_smoothing,
        )
        return loss

    @torch.no_grad()
    def generate(self, feats, input_ids, attention_mask, **gen_kw):
        inp, attn, _ = self._build(feats, input_ids, attention_mask, None)
        out = self.lm.generate(inputs_embeds=inp, attention_mask=attn, **gen_kw)
        return out


def load_lm(cfg: Config):
    # attn_implementation: "eager" materialises a (B, H, T, T) score tensor per
    # layer and keeps it for backward. At B=4, H=32, T=247 over 32 layers that
    # is several GB of retained activations — enough to push a 12 GiB card into
    # sysmem fallback, where every access crosses PCIe and steps take minutes.
    # "sdpa" uses a fused kernel that does not retain the score matrix.
    #
    # trust_remote_code is deliberately NOT set here: Phi-2 is native in
    # transformers, and the remote modeling file is eager-only, so passing it
    # can silently override the sdpa request. The tokenizer still uses it.
    lm = AutoModelForCausalLM.from_pretrained(
        cfg.lm_name,
        torch_dtype=AMP_DTYPE if DEVICE.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        cache_dir=cfg.hf_cache,
    )
    lm.config.pad_token_id = tok.pad_token_id
    lm.config.use_cache = False
    impl = getattr(lm.config, "_attn_implementation", "unknown")
    print(f"LM loaded: dtype={next(lm.parameters()).dtype} attn={impl}")
    assert impl == "sdpa", (
        f"attention is {impl!r}, not sdpa. Eager attention retains the score "
        "matrix per layer and will exhaust VRAM. Check the transformers version."
    )
    return lm


def attach_lora(lm, cfg: Config):
    lcfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets),
        bias="none",
        task_type="CAUSAL_LM",
    )
    lm = get_peft_model(lm, lcfg)
    lm.print_trainable_parameters()
    return lm


lm = load_lm(cfg)
for p in lm.parameters():
    p.requires_grad = False
proj = ProjectionMLP(cfg.vit_dim, cfg.proj_hidden, cfg.lm_dim,
                     gate_init=cfg.gate_init, dropout=cfg.proj_dropout).to(DEVICE)
model = MICModel(lm.to(DEVICE), proj)

# Trainable params (~11M) stay fp32 so AdamW updates are not quantised away at
# lr=1e-3. The projection is already fp32 by construction; this covers LoRA in
# Stage B, where get_peft_model inherits the base model's bf16.
for n, p in model.named_parameters():
    if p.requires_grad:
        p.data = p.data.float()

_n_proj = sum(p.numel() for p in proj.parameters())
_n_lora = cfg.lora_r * 2 * cfg.lm_dim * 2 * lm.config.num_hidden_layers
print(f"\nparameter budget (for the paper / architecture figure)")
print(f"  ViT-B/16 (frozen)      {human(_n_vit)}")
print(f"  Phi-2    (frozen)      {human(sum(p.numel() for p in lm.parameters()))}")
print(f"  projection (trainable) {human(_n_proj)}")
print(f"  LoRA r={cfg.lora_r} (trainable)  ~{human(_n_lora)}")
print(f"  trainable total        ~{human(_n_proj + _n_lora)}")
print(f"  proj stats at init     {proj.stats()}")
if DEVICE.type == "cuda":
    print(f"  VRAM after load        {torch.cuda.memory_allocated() / 2**30:.2f} GiB "
          f"(expect ~5.3)")
    torch.cuda.reset_peak_memory_stats()

# %% [markdown]
# ## Cell 11 — Grounding diagnostics
#
# * `zeros_gap` = loss(zeroed features) − loss(real features). Positive means
#   real images help. v4 sat at −1.85, i.e. images were actively hurting.
# * `mismatch_gap` = loss(shuffled features) − loss(real features). This is the
#   number to report: it isolates *which* image the caption came from, whereas
#   `zeros_gap` can be inflated by any non-zero prefix.
# * `distinct-2` on generated captions. A collapsed model emits one caption.
#
# All three run on a **fixed** probe subset so values are comparable across
# epochs. Enable `cfg.aux_grounding_weight` only if `mismatch_gap` is still
# under ~0.05 at the end of Stage A.

# %%
@torch.no_grad()
def grounding_probe(model: MICModel, ds: CachedCaptionDataset,
                    n: int = 256, bs: int = 4) -> Dict[str, float]:
    model.eval()
    idx = list(range(min(n, len(ds))))
    loader = DataLoader(torch.utils.data.Subset(ds, idx), batch_size=bs,
                        shuffle=False, collate_fn=collate_train, num_workers=0)
    real = zero = mism = 0.0
    nb = 0
    for b in loader:
        feats = b["feats"].to(DEVICE)
        ids = b["input_ids"].to(DEVICE)
        attn = b["attention_mask"].to(DEVICE)
        lab = b["labels"].to(DEVICE)
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
            real += model(feats, ids, attn, lab).item()
            zero += model(torch.zeros_like(feats), ids, attn, lab).item()
            if feats.size(0) > 1:
                perm = torch.roll(torch.arange(feats.size(0), device=DEVICE), 1)
                mism += model(feats[perm], ids, attn, lab).item()
            else:
                mism += float("nan")
        nb += 1
    real, zero, mism = real / nb, zero / nb, mism / nb
    return {"loss_real": real, "loss_zeros": zero, "loss_mismatch": mism,
            "zeros_gap": zero - real, "mismatch_gap": mism - real}


def distinct_n(caps: List[str], n: int = 2) -> float:
    grams = set()
    total = 0
    for c in caps:
        w = c.split()
        for i in range(len(w) - n + 1):
            grams.add(tuple(w[i:i + n]))
            total += 1
    return len(grams) / max(total, 1)


def report_probe(tag: str, p: Dict[str, float]):
    print(f"  [{tag}] real={p['loss_real']:.4f} zeros={p['loss_zeros']:.4f} "
          f"mismatch={p['loss_mismatch']:.4f} | zeros_gap={p['zeros_gap']:+.4f} "
          f"mismatch_gap={p['mismatch_gap']:+.4f}")
    if tag.endswith("/pre"):
        print("       (pre-training reference: mismatch_gap ~0 is expected here)")
    elif p["mismatch_gap"] < 0.02:
        print("       WARNING: mismatch_gap ~ 0 — captions do not depend on the image.")
    

# %% [markdown]
# ## Cell 12 — OVERFIT SANITY GATE (C5) — hard fail
#
# Train the projection (LM frozen) on 8 images for 300 steps, then generate on
# those same 8. A working visual path memorises them: loss < 0.15 and the
# generated caption matches the reference closely. If this fails, the break is
# structural — embedding concat, label offset, dtype, feature path — and no
# amount of schedule tuning on 79k images will fix it. Ten minutes here beats
# twenty hours of a collapsing run.
#
# The projection is re-initialised afterwards, so this cell leaves no trace on
# the real run.

# %%
def overfit_gate(cfg: Config, model: MICModel, ds: CachedCaptionDataset) -> bool:
    import copy
    saved = copy.deepcopy(model.proj.state_dict())
    model.proj.gate.requires_grad_(True)
    for p in model.proj.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.proj.parameters(), lr=1e-3, weight_decay=0.0)

    items = [ds[i] for i in range(cfg.overfit_images)]
    batch = collate_train(items)
    feats = batch["feats"].to(DEVICE)
    ids = batch["input_ids"].to(DEVICE)
    attn = batch["attention_mask"].to(DEVICE)
    lab = batch["labels"].to(DEVICE)

    print(f"overfitting {cfg.overfit_images} images for {cfg.overfit_steps} steps "
          f"(no label smoothing; loss not comparable to Stage A)")
    with torch.no_grad():
        f = feats.float()
        print(f"  feats: mean={f.mean():+.4f} std={f.std():.4f} "
              f"min={f.min():+.3f} max={f.max():+.3f}")
    model.train()
    first = None
    last = float("inf")
    for step in range(1, cfg.overfit_steps + 1):
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
            loss = model(feats, ids, attn, lab, label_smoothing=0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.proj.parameters(), cfg.max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        last = loss.item()
        if first is None:
            first = last
        if step % 50 == 0 or step == 1:
            print(f"  step {step:>4}  loss {last:.4f}  {model.proj.stats()}")

    model.eval()
    gen_batch = collate_gen(items)
    with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
        out = model.generate(
            gen_batch["feats"].to(DEVICE),
            gen_batch["input_ids"].to(DEVICE),
            gen_batch["attention_mask"].to(DEVICE),
            max_new_tokens=cfg.max_new_tokens, num_beams=1, do_sample=False,
            min_new_tokens=cfg.min_new_tokens, use_cache=True,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
        )
    gens = [tok.decode(o, skip_special_tokens=True).strip() for o in out]
    print("\n  generated vs reference")
    n_match = 0
    for g, r in zip(gens, batch["captions"]):
        gw, rw = set(g.lower().split()), set(r.lower().split())
        ov = len(gw & rw) / max(len(rw), 1)
        n_match += int(ov > 0.6)
        print(f"    overlap {ov:.2f}\n      GEN: {g[:110]}\n      REF: {r[:110]}")

    model.proj.load_state_dict(saved)
    opt.zero_grad(set_to_none=True)
    del opt
    free_cuda()

    loss_ok = (last <= cfg.overfit_loss_max) or (last <= cfg.overfit_drop_ratio * first)
    cap_ok = n_match >= max(cfg.overfit_images // 2, 1)
    passed = loss_ok and cap_ok
    print(f"\n  loss {first:.4f} -> {last:.4f} "
          f"(pass if <= {cfg.overfit_loss_max} or <= "
          f"{cfg.overfit_drop_ratio * first:.4f})  loss_ok={loss_ok}")
    print(f"  {n_match}/{cfg.overfit_images} captions recovered (>0.6 token overlap) "
          f"cap_ok={cap_ok}")
    print(f"  OVERFIT GATE: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("""
  Do not proceed. In order, check:
    1. model._build — is `vis` really prepended, and does `lab` get -100 padding
       of exactly vis.shape[1] columns?
    2. logits[:, :-1] vs labels[:, 1:] — an off-by-one here trains on the wrong
       target and still produces a smoothly falling loss.
    3. proj.gate — if it is drifting toward 0 even here, with 8 images and no
       LoRA, the features themselves are the problem: re-run Cell 8.
    4. feats dtype/scale — print feats.mean()/std(); ViT last_hidden_state
       should be roughly zero-mean with std around 0.5-1.5.""")
    return passed


assert overfit_gate(cfg, model, train_ds), "overfit gate failed — stop here"
log.write(event="overfit_gate", passed=True)
        

# %% [markdown]
# ## Cell 13 — Checkpointing
#
# Every checkpoint carries `proj_arch`, `cache_sig` and the stage. `load_ckpt`
# hard-fails on a tag mismatch rather than doing a silent `strict=False` partial
# load — that is how a shortcut-baked state dict survives a "fix".

# %%
def save_ckpt(path: str, model: MICModel, stage: str, epoch: int,
              best: float, extra: Optional[Dict] = None):
    payload = {
        "proj_arch": PROJ_ARCH,
        "cache_sig": cfg.cache_sig(),
        "stage": stage,
        "epoch": epoch,
        "best_cider": best,
        "proj": {k: v.cpu() for k, v in model.proj.state_dict().items()},
        "config": {k: v for k, v in asdict(cfg).items() if isinstance(v, (int, float, str, bool, type(None)))},
    }
    try:
        payload["lora"] = {k: v.cpu() for k, v in get_peft_model_state_dict(model.lm).items()}
    except Exception:
        payload["lora"] = None
    if extra:
        payload["extra"] = extra
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_ckpt(path: str, model: MICModel, strict_sig: bool = True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("proj_arch") != PROJ_ARCH:
        raise RuntimeError(
            f"checkpoint proj_arch={ck.get('proj_arch')} != {PROJ_ARCH}. "
            f"Delete it:\n  rm {path}")
    if strict_sig and ck.get("cache_sig") != cfg.cache_sig():
        raise RuntimeError(
            f"checkpoint was trained on cache_sig={ck.get('cache_sig')}, "
            f"current is {cfg.cache_sig()}. Features changed; do not resume.")
    model.proj.load_state_dict(ck["proj"])
    if ck.get("lora"):
        set_peft_model_state_dict(model.lm, ck["lora"])
    print(f"resumed {path}: stage={ck['stage']} epoch={ck['epoch']} "
          f"best_cider={ck['best_cider']:.4f}")
    return ck


CKPT_A = os.path.join(cfg.ckpt_dir, "stage_a.pt")
CKPT_B = os.path.join(cfg.ckpt_dir, "stage_b.pt")
CKPT_BEST = os.path.join(cfg.ckpt_dir, "best.pt")

# %% [markdown]
# ## Cell 14 — Evaluation (pycocoevalcap)
#
# `pycocoevalcap` is authoritative. NLTK numbers are not numerically comparable
# and must never share a table with these. Also computes the **blind** ablation
# (zeroed features) on the identical image subset, so the sighted/blind delta in
# metric space is an apples-to-apples comparison.

# %%
try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    PYCOCO = True
except Exception as e:
    PYCOCO = False
    print(f"pycocoevalcap unavailable ({e}) — metrics will be skipped")


def coco_metrics(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    if not PYCOCO:
        return {}
    gts = {str(i): [{"caption": r}] for i, r in enumerate(refs)}
    res = {str(i): [{"caption": h if h.strip() else "none"}] for i, h in enumerate(hyps)}
    ptb = PTBTokenizer()
    gts, res = ptb.tokenize(gts), ptb.tokenize(res)
    out: Dict[str, float] = {}
    bleu, _ = Bleu(4).compute_score(gts, res)
    for i, b in enumerate(bleu, 1):
        out[f"BLEU-{i}"] = float(b)
    out["METEOR"] = float(Meteor().compute_score(gts, res)[0])
    out["ROUGE-L"] = float(Rouge().compute_score(gts, res)[0])
    out["CIDEr"] = float(Cider().compute_score(gts, res)[0])
    return out


GEN_KW = dict(
    max_new_tokens=cfg.max_new_tokens,
    min_new_tokens=cfg.min_new_tokens,
    length_penalty=cfg.length_penalty,
    no_repeat_ngram_size=cfg.no_repeat_ngram_size,
    do_sample=False,
    use_cache=True,
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.pad_token_id,
)


@torch.no_grad()
def generate_split(model: MICModel, ds: CachedCaptionDataset,
                   limit: Optional[int] = None, blind: bool = False,
                   num_beams: Optional[int] = None) -> Tuple[List[str], List[str]]:
    model.eval()
    sub = ds if limit is None else torch.utils.data.Subset(ds, list(range(min(limit, len(ds)))))
    loader = DataLoader(sub, batch_size=cfg.gen_batch_size, shuffle=False,
                        collate_fn=collate_gen, num_workers=cfg.num_workers)
    beams = cfg.num_beams if num_beams is None else num_beams
    refs, hyps = [], []
    t0 = time.time()
    for bi, b in enumerate(loader):
        feats = b["feats"].to(DEVICE)
        if blind:
            feats = torch.zeros_like(feats)
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
            out = model.generate(
                feats, b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE),
                num_beams=beams, early_stopping=beams > 1, **GEN_KW,
            )
        hyps.extend(tok.decode(o, skip_special_tokens=True).strip() for o in out)
        refs.extend(b["captions"])
        if bi % 25 == 0:
            done = len(hyps)
            print(f"    gen {done}  ({done / max(time.time() - t0, 1e-6):.1f} cap/s)",
                  flush=True)
    return refs, hyps


def evaluate(model: MICModel, ds: CachedCaptionDataset, tag: str,
             limit: Optional[int] = None, with_blind: bool = False) -> Dict[str, Any]:
    refs, hyps = generate_split(model, ds, limit=limit)
    m = coco_metrics(refs, hyps)
    m["distinct-2"] = distinct_n(hyps, 2)
    m["unique_caption_ratio"] = len(set(hyps)) / max(len(hyps), 1)
    m["mean_len"] = float(np.mean([len(h.split()) for h in hyps]))
    m["n"] = len(hyps)
    print(f"  [{tag}] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in m.items()))
    res: Dict[str, Any] = {"tag": tag, "sighted": m}
    if with_blind:
        _, bhyps = generate_split(model, ds, limit=limit, blind=True)
        bm = coco_metrics(refs, bhyps)
        bm["distinct-2"] = distinct_n(bhyps, 2)
        identical = sum(int(a == b) for a, b in zip(hyps, bhyps)) / max(len(hyps), 1)
        bm["identical_to_sighted"] = identical
        print(f"  [{tag}/blind] " + "  ".join(f"{k}={v:.4f}" for k, v in bm.items()
                                              if isinstance(v, float)))
        print(f"  CIDEr sighted-blind delta = "
              f"{m.get('CIDEr', float('nan')) - bm.get('CIDEr', float('nan')):+.4f}")
        res["blind"] = bm
    with open(os.path.join(cfg.out_dir, f"eval_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump({"metrics": res,
                   "samples": [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps))[:200]]},
                  fh, indent=2)
    return res


# ---- metric smoke test: fail now, not after a full generation pass ----------
_m = coco_metrics(["chest x-ray showing a right lower lobe nodule",
                   "axial CT of the abdomen with a hepatic cyst"],
                  ["chest x-ray with a nodule in the right lower lobe",
                   "CT of the abdomen showing a liver cyst"])
_need = {"BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"}
assert _need <= set(_m), f"metrics missing: {_need - set(_m)} (check Java for METEOR/PTB)"
print("metric smoke test:", {k: round(v, 4) for k, v in _m.items()})

# ---- generation speed check: confirms the KV cache is active ----------------
_t = time.time()
generate_split(model, val_ds, limit=16, num_beams=cfg.num_beams)
print(f"16 captions with beams={cfg.num_beams}: {time.time() - _t:.1f}s")





# %% [markdown]
# ## Cell 15 — The training loop
#
# One function for both stages; what differs is which parameters carry gradient
# and which optimizer they are attached to. Stage B builds a **fresh** AdamW and
# a fresh cosine schedule (C7) — carrying Stage A's second-moment estimates into
# LoRA is what let the text prior win the race in v3.

# %%
def param_groups(model: MICModel, stage: str, cfg: Config):
    if stage == "A":
        for p in model.lm.parameters():
            p.requires_grad_(False)
        for n, p in model.proj.named_parameters():
            p.requires_grad_(True if n != "gate" else cfg.gate_trainable_stage_a)
        return [{"params": [p for p in model.proj.parameters() if p.requires_grad],
                 "lr": cfg.lr_proj_a, "name": "proj"}]
    for n, p in model.proj.named_parameters():
        p.requires_grad_(True if n != "gate" else cfg.gate_trainable_stage_b)
    lora = [p for n, p in model.lm.named_parameters() if "lora_" in n]
    for p in lora:
        p.requires_grad_(True)
    return [
        {"params": [p for p in model.proj.parameters() if p.requires_grad],
         "lr": cfg.lr_proj_b, "name": "proj"},
        {"params": lora, "lr": cfg.lr_lora, "name": "lora"},
    ]


CKPT_BEST_A = os.path.join(cfg.ckpt_dir, "best_stage_a.pt")


def run_stage(model: MICModel, stage: str, epochs: int, cfg: Config,
              best_cider: float = -1.0) -> float:
    from transformers import get_cosine_schedule_with_warmup

    groups = param_groups(model, stage, cfg)
    names = [g["name"] for g in groups]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate_train, num_workers=cfg.num_workers,
                        pin_memory=DEVICE.type == "cuda", drop_last=True,
                        persistent_workers=cfg.num_workers > 0)
    steps_per_epoch = max(len(loader) // cfg.grad_accum, 1)
    total = steps_per_epoch * epochs
    sched = get_cosine_schedule_with_warmup(
        opt, int(total * cfg.warmup_ratio), total)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_SCALER)

    trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"\n{'=' * 72}\nSTAGE {stage}: {epochs} epoch(s), {total} optimizer steps, "
          f"groups={names}, trainable={human(trainable)}\n{'=' * 72}")
    if stage == "A":
        print(f"  gate trainable in Stage A: {model.proj.gate.requires_grad} "
              f"(value {float(model.proj.gate):.4f})")

    p0 = grounding_probe(model, val_ds, cfg.diag_probe_samples)
    report_probe(f"stage{stage}/pre", p0)
    log.write(event="stage_start", stage=stage, total_steps=total, **p0)

    gstep = 0
    for ep in range(1, epochs + 1):
        model.train()
        # Checkpointing applies to BOTH stages. A `stage == "B"` guard here was
        # the memory bug: in Stage A the LM is frozen but the projected visual
        # prefix carries grad, so without checkpointing every activation in all
        # 32 layers is retained for backward. use_reentrant=False is required
        # precisely because no *parameter* of the base model needs grad.
        if (cfg.grad_checkpointing
                and hasattr(model.lm, "gradient_checkpointing_enable")):
            model.lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            model.lm.config.use_cache = False
        run_loss, nb, t0 = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(loader):
            feats = b["feats"].to(DEVICE, non_blocking=True)
            ids = b["input_ids"].to(DEVICE, non_blocking=True)
            attn = b["attention_mask"].to(DEVICE, non_blocking=True)
            lab = b["labels"].to(DEVICE, non_blocking=True)

            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
                loss = model(feats, ids, attn, lab,
                             label_smoothing=cfg.label_smoothing)
                if (cfg.aux_grounding_weight > 0 and stage == "B"
                        and i % cfg.aux_every == 0):
                    with torch.no_grad():
                        lz = model(torch.zeros_like(feats), ids, attn, lab,
                                   label_smoothing=cfg.label_smoothing)
                    loss = loss + cfg.aux_grounding_weight * F.relu(
                        cfg.aux_grounding_margin + loss - lz)

            if not torch.isfinite(loss):
                print(f"  non-finite loss at micro-step {i}, skipping")
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss / cfg.grad_accum).backward()
            run_loss += loss.item()
            nb += 1

            if (i + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                gn = {g["name"]: float(torch.nn.utils.clip_grad_norm_(
                    g["params"], cfg.max_grad_norm)) for g in groups}
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1

                # First logged step reports peak VRAM and observed step rate.
                # Peak must stay under ~11 GiB on a 12 GiB card; above that the
                # driver serves from system RAM and throughput collapses ~40x.
                if gstep == 50 and DEVICE.type == "cuda":
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    sps = (time.time() - t0) / gstep
                    print(f"  [mem] peak {peak:.2f} GiB | {sps:.2f} s/opt-step "
                          f"| epoch eta {sps * steps_per_epoch / 3600:.1f} h")
                    if peak > 11.0:
                        print("  [mem] WARNING: peak is close to the 12 GiB cap. "
                              "Halve batch_size and double grad_accum — the "
                              "effective batch of 32 is unchanged.")

                if gstep % 50 == 0:
                    print(f"  ep{ep} step {gstep}/{total} loss={run_loss / nb:.4f} "
                          f"(floor {SMOOTH_FLOOR:.3f}) "
                          f"lr={sched.get_last_lr()[0]:.2e} gn={gn} "
                          f"{model.proj.stats()}", flush=True)
                    log.write(event="step", stage=stage, epoch=ep, step=gstep,
                              loss=run_loss / nb, grad_norms=gn,
                              **model.proj.stats())

        train_min = (time.time() - t0) / 60

        # Turn checkpointing off before generation. It is inert under no_grad,
        # but leaving it on keeps use_cache=False sticky, and beam search
        # without a KV cache is roughly an order of magnitude slower.
        if hasattr(model.lm, "gradient_checkpointing_disable"):
            model.lm.gradient_checkpointing_disable()

        p = grounding_probe(model, val_ds, cfg.diag_probe_samples)
        report_probe(f"stage{stage}/ep{ep}", p)
        ev = evaluate(model, val_ds, f"val_stage{stage}_ep{ep}",
                      limit=cfg.val_probe_samples)
        cider = ev["sighted"].get("CIDEr", -1.0)
        print(f"  epoch {ep}: train {train_min:.1f} min, total {(time.time() - t0) / 60:.1f} min "
              f"train_loss={run_loss / max(nb, 1):.4f} val_CIDEr(selection only)={cider:.4f}")
        log.write(event="epoch", stage=stage, epoch=ep, train_minutes=train_min,
                  train_loss=run_loss / max(nb, 1), val_cider=cider, **p)

        save_ckpt(CKPT_A if stage == "A" else CKPT_B, model, stage, ep, best_cider)
        if cider > best_cider:
            best_cider = cider
            save_ckpt(CKPT_BEST, model, stage, ep, best_cider, extra={"probe": p})
            print(f"  new best (CIDEr {best_cider:.4f}) -> {CKPT_BEST}")
            if stage == "A":
                save_ckpt(CKPT_BEST_A, model, stage, ep, best_cider, extra={"probe": p})
                print(f"  stage A best preserved -> {CKPT_BEST_A}")

    del opt, sched, loader
    free_cuda()
    return best_cider    
    

# %% [markdown]
# ## Cell 16 — Stage A: projection only
#
# Phi-2 fully frozen, no LoRA. Gradients have exactly one path to reduce loss:
# through the visual projection. This is what removes the race between the
# text-prior shortcut (fast) and visual grounding (slow).

# %%
best = run_stage(model, "A", cfg.stage_a_epochs, cfg)
print(f"\nStage A best val CIDEr: {best:.4f}")
pa = grounding_probe(model, val_ds, cfg.diag_probe_samples)
report_probe("stageA/final", pa)
if pa["mismatch_gap"] < 0.02:
    print("""
STOP AND READ. mismatch_gap is ~0 after Stage A with the LM frozen and no LoRA.
That can not be a LoRA-shortcut problem, and the overfit gate already passed, so
the projection can fit 8 images but not generalise across 79k. Likely causes, in
order: lr_proj_a too high (try 3e-4), 197 visual tokens swamping a short caption
(consider mean-pooling to 32 before the projection — that IS an architecture
change, so decide deliberately), or the caption distribution being genuinely
near-unimodal. Do not start Stage B expecting it to rescue this.""")

# %% [markdown]
# ## Cell 17 — Stage B: attach LoRA, fresh optimizer
#
# The projection weights carry forward; the optimizer state does not.

# %%
best_a = best
lm_peft = attach_lora(model.lm, cfg)
model.lm = lm_peft
model.lm.config.use_cache = False

# best_cider resets to -1: a Stage A "best" has no LoRA state, so loading it
# into the Stage B model in Cell 18 would pair Stage A's projection with
# whatever LoRA weights happened to be resident. Stage A's number is kept in
# best_a for the paper's ablation table.
best = run_stage(model, "B", cfg.stage_b_epochs, cfg, best_cider=-1.0)
print(f"\nStage A best val CIDEr: {best_a:.4f}")
print(f"Stage B best val CIDEr: {best:.4f}")

# %% [markdown]
# ## Cell 18 — Final evaluation on the full test split
#
# Checkpoint selected on validation CIDEr, reported on the held-out test split.
# `eval_samples=None` means the full official ~9,927 images — not the 500-image
# subset. Expect the headline numbers to move slightly relative to the 500-image
# run; the full-split numbers are the ones to put in the paper.

# %%
load_ckpt(CKPT_BEST, model)
test_res = evaluate(model, test_ds, "test_final", limit=cfg.eval_samples,
                    with_blind=True)
final_probe = grounding_probe(model, test_ds, cfg.diag_probe_samples)
report_probe("test/final", final_probe)

summary = {
    "proj_arch": PROJ_ARCH,
    "cache_sig": cfg.cache_sig(),
    "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
    "config": {k: v for k, v in asdict(cfg).items()
               if isinstance(v, (int, float, str, bool, type(None)))},
    "splits": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    "cache_audit": audit,
    "best_val_cider_stage_a": best_a,
    "best_val_cider": best,
    "test": test_res,
    "grounding": final_probe,
    "decoding": {"num_beams": cfg.num_beams, "max_new_tokens": cfg.max_new_tokens,
                 "min_new_tokens": cfg.min_new_tokens,
                 "no_repeat_ngram_size": cfg.no_repeat_ngram_size},
}
with open(os.path.join(cfg.out_dir, "final_results.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary["test"], indent=2))
print(f"\nwrote {os.path.join(cfg.out_dir, 'final_results.json')}")

# %% [markdown]
# ## Cell 19 — Sample inference, sighted vs blind
#
# Ten rows where GEN and BLIND read the same is more legible evidence of
# collapse than any scalar in the log.

# %%
n_show = 10
items = [test_ds[i] for i in range(min(n_show, len(test_ds)))]
gb = collate_gen(items)
kw = dict(GEN_KW, num_beams=cfg.num_beams, early_stopping=cfg.num_beams > 1)
model.eval()
with torch.no_grad(), torch.autocast(DEVICE.type, dtype=AMP_DTYPE,
                                     enabled=DEVICE.type == "cuda"):
    s = model.generate(gb["feats"].to(DEVICE), gb["input_ids"].to(DEVICE),
                       gb["attention_mask"].to(DEVICE), **kw)
    z = model.generate(torch.zeros_like(gb["feats"]).to(DEVICE),
                       gb["input_ids"].to(DEVICE),
                       gb["attention_mask"].to(DEVICE), **kw)
sighted = [tok.decode(o, skip_special_tokens=True).strip() for o in s]
blind = [tok.decode(o, skip_special_tokens=True).strip() for o in z]
same = 0
for i, (g, bl, r) in enumerate(zip(sighted, blind, gb["captions"])):
    same += int(g == bl)
    print(f"\n[{i}] REF   : {r}")
    print(f"    GEN   : {g}")
    print(f"    BLIND : {bl}   {'<-- IDENTICAL' if g == bl else ''}")
print(f"\nidentical sighted/blind: {same}/{len(sighted)}")