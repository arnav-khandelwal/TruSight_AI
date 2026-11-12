# -*- coding: utf-8 -*-
"""
Single shared CLIP backbone (transformers, no torchvision dependency), used for
BOTH near-duplicate/reuse embeddings (temporal_analysis) and scene-vs-claim
zero-shot classification (contextual_analysis). Loading one model instead of two
keeps the 4 GB GPU within budget. Lazy-loaded so importing the module is cheap.
"""
import logging
from functools import lru_cache
from typing import List, Dict

import numpy as np
import torch
from PIL import Image

import config

log = logging.getLogger("trusight.clip")
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_proc = None


def _ensure():
    global _model, _proc
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor
        log.info("Loading CLIP backbone %s on %s", config.CLIP_MODEL, _DEVICE)
        _model = CLIPModel.from_pretrained(config.CLIP_MODEL).to(_DEVICE).eval()
        _proc = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
    return _model, _proc


def image_embeddings(imgs: List[Image.Image]) -> np.ndarray:
    """L2-normalized CLIP image embeddings, shape (N, D)."""
    if not imgs:
        return np.zeros((0, 512), dtype=np.float32)
    model, proc = _ensure()
    with torch.no_grad():
        inp = proc(images=imgs, return_tensors="pt").to(_DEVICE)
        feat = model.get_image_features(**inp)
        # In transformers >= 4.40 / 5.x, get_image_features returns BaseModelOutputWithPooling
        if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
            feat = feat.pooler_output
        elif isinstance(feat, (list, tuple)):
            feat = feat[0]
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return feat.cpu().numpy().astype(np.float32)


def zero_shot(img: Image.Image, labels: List[str]) -> Dict[str, float]:
    """Softmax probabilities over `labels` for one image (CLIP zero-shot)."""
    model, proc = _ensure()
    with torch.no_grad():
        inp = proc(text=labels, images=img, return_tensors="pt", padding=True).to(_DEVICE)
        probs = model(**inp).logits_per_image.softmax(dim=-1)[0].cpu().numpy()
    return {labels[i]: float(probs[i]) for i in range(len(labels))}
