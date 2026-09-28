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

