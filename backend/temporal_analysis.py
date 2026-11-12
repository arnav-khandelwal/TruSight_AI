# -*- coding: utf-8 -*-
"""
Temporal reuse / recycled-footage detection.

Two signals, both using shared CLIP image embeddings (no torchvision):

  1. Reference matching - if an operator has configured a reference database of
     known footage (models/reference_embeddings.pkl), each sampled frame is
     matched against it; a high match ratio flags footage recycled from that
     reference set. When NO reference database is configured we say so honestly
     (previously this module silently compared against random vectors, which was
     a bug that produced meaningless similarities).

  2. Intra-clip near-duplication - always available: if non-adjacent frames are
     near-identical, the clip may contain looped or spliced-in repeated segments.

The reference database must be built by the operator from genuine reference
footage via build_reference_database(); we never ship synthetic/random vectors.
"""
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image

import config
from clip_backbone import image_embeddings

log = logging.getLogger("trusight.temporal")
MODELS_DIR = Path(__file__).parent / "models"
REFERENCE_PATH = MODELS_DIR / "reference_embeddings.pkl"


def _load_reference() -> np.ndarray:
    """Load reference embeddings, or return an empty array if none are configured.
    A dimension mismatch (e.g. an old ResNet-2048 or random file) is treated as
    'not configured' rather than silently used."""
    if not REFERENCE_PATH.exists():
        return np.zeros((0, 512), dtype=np.float32)
    try:
        with open(REFERENCE_PATH, "rb") as f:
            embs = np.asarray(pickle.load(f), dtype=np.float32)
        if embs.ndim != 2 or embs.shape[1] != 512:
            log.warning("Reference DB has wrong shape %s; ignoring.", embs.shape)
            return np.zeros((0, 512), dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.clip(norms, 1e-8, None)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not load reference DB: %s", e)
        return np.zeros((0, 512), dtype=np.float32)


def _intra_clip_duplication(embs: np.ndarray) -> float:
    """Max cosine similarity between non-adjacent frame pairs (0 if <3 frames)."""
    n = len(embs)
    if n < 3:
        return 0.0
    sims = embs @ embs.T
    best = 0.0
    for i in range(n):
        for j in range(i + 2, n):        # skip adjacent frames (naturally similar)
            best = max(best, float(sims[i, j]))
    return best


def analyze_temporal_reuse(frame_paths: List[str]) -> Dict[str, Any]:
    if not frame_paths:
        return {"detected": False, "confidence": 0.0, "details": "No frames to analyze",
                "matched_frames": 0, "total_frames": 0}

    imgs = []
    for p in frame_paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:  # noqa: BLE001
            continue
    if not imgs:
        return {"detected": False, "confidence": 0.0, "details": "Frames could not be read",
                "matched_frames": 0, "total_frames": len(frame_paths)}

    embs = image_embeddings(imgs)
    total = len(embs)
    reference = _load_reference()

    # ---- signal 1: reference matching --------------------------------------------------
    if reference.shape[0] > 0:
        sims = embs @ reference.T                 # (frames, refs), both normalized
        max_sim = sims.max(axis=1)
        matched = int((max_sim >= config.REUSE_SIMILARITY_THRESHOLD).sum())
        match_ratio = matched / total
        avg_top = float(max_sim.mean())
        ref_detected = match_ratio >= config.REUSE_MATCH_RATIO
        ref_msg = (f"{matched}/{total} frames match configured reference footage "
                   f"(mean top-similarity {avg_top:.2f}).")
    else:
        matched, match_ratio, avg_top, ref_detected = 0, 0.0, 0.0, False
        ref_msg = ("No reference database configured, so recycling from known footage "
                   "could not be checked (an operator can build one from reference clips).")

    # ---- signal 2: intra-clip near-duplication ----------------------------------------
    dup = _intra_clip_duplication(embs)
    dup_flag = dup >= 0.98
    dup_msg = (f"Repeated or looped segments suspected (max non-adjacent frame similarity "
               f"{dup:.2f})." if dup_flag else
               f"No looped segments (max non-adjacent frame similarity {dup:.2f}).")

    detected = bool(ref_detected or dup_flag)
    confidence = float(max(avg_top if ref_detected else 0.0, dup if dup_flag else 0.0))
    details = ref_msg + " " + dup_msg

    return {
        "detected": detected,
        "confidence": confidence,
        "details": details,
        "matched_frames": matched,
        "total_frames": total,
        "reference_configured": bool(reference.shape[0] > 0),
        "intra_clip_duplication": dup,
    }


def build_reference_database(image_dir: str, output_path: str = None) -> int:
    """Build a reference embedding database from genuine reference images.
    Returns the number of embeddings stored."""
    out = Path(output_path) if output_path else REFERENCE_PATH
    MODELS_DIR.mkdir(exist_ok=True)
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        files.extend(Path(image_dir).glob(ext))
    imgs = []
    for f in files:
        try:
            imgs.append(Image.open(f).convert("RGB"))
        except Exception:  # noqa: BLE001
            continue
    embs = image_embeddings(imgs)
    with open(out, "wb") as f:
        pickle.dump(embs, f)
    log.info("Saved %d reference embeddings to %s", len(embs), out)
    return len(embs)
