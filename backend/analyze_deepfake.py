# -*- coding: utf-8 -*-
"""
Deepfake (face-manipulation) detection.

Pipeline: detect faces per frame with OpenCV YuNet, crop them, and fuse two
training-free-to-deploy signals per face:
  * a pretrained face-forgery ViT (dima806/deepfake_vs_real_image_detection), and
  * a frequency-domain artifact score (forensic_signals), which captures spectral
    fingerprints that generative upsampling tends to leave.
Across frames we also measure temporal coherence (score flicker), since face
swaps are often unstable frame to frame. Weights and bands come from config.py.

Honest defaults: a clip with no detectable face returns LOW risk (nothing to
assess), and every output is a probabilistic indicator, not a verdict.
"""
import logging
import os
from typing import Dict, Any, List

import cv2
import numpy as np
from PIL import Image
import torch

import config
from forensic_signals import frequency_artifact_score

log = logging.getLogger("trusight.deepfake")
_DEVICE = 0 if torch.cuda.is_available() else -1
_HERE = os.path.dirname(__file__)
YUNET_PATH = os.path.join(_HERE, "models", "face_yunet.onnx")

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        _classifier = pipeline("image-classification", model=config.DEEPFAKE_MODEL, device=_DEVICE)
        log.info("Deepfake face classifier loaded on %s", "cuda" if _DEVICE == 0 else "cpu")
    return _classifier


def _fake_prob(preds: List[Dict[str, Any]]) -> float:
    for p in preds:
        if "fake" in p["label"].lower():
            return float(p["score"])
    return 0.0


def _detect_faces(frame_paths: List[str]) -> List[List[Image.Image]]:
    """Return, per frame, a list of up to MAX_FACES_PER_FRAME padded face crops."""
    if not os.path.exists(YUNET_PATH):
        return [[] for _ in frame_paths]
    detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320),
                                          score_threshold=config.FACE_SCORE_THRESHOLD)
    per_frame = []
    for path in frame_paths:
        frame = cv2.imread(path)
        crops: List[Image.Image] = []
        if frame is not None:
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            if faces is not None:
                for f in faces[: config.MAX_FACES_PER_FRAME]:
                    x, y, fw, fh = [int(v) for v in f[:4]]
                    pad = int(0.25 * max(fw, 1))
                    x0, y0 = max(0, x - pad), max(0, y - pad)
                    x1, y1 = min(w, x + fw + pad), min(h, y + fh + pad)
                    crop = frame[y0:y1, x0:x1]
                    if crop.size:
                        crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        per_frame.append(crops)
    return per_frame


def compute_deepfake_score(frame_paths: List[str]) -> Dict[str, Any]:
    """Return fused face score plus its components (used by eval and the endpoint)."""
    per_frame = _detect_faces(frame_paths)
    classifier = _get_classifier()

    frame_means, model_scores, freq_scores = [], [], []
    for crops in per_frame:
        if not crops:
            continue
        fused_this_frame = []
        for crop in crops:
            try:
                mp = _fake_prob(classifier(crop))
            except Exception as e:  # noqa: BLE001
                log.debug("classifier error: %s", e)
                continue
            fq, _ = frequency_artifact_score(crop)
            fused = config.DEEPFAKE_FUSION["model"] * mp + config.DEEPFAKE_FUSION["frequency"] * fq
            model_scores.append(mp)
            freq_scores.append(fq)
            fused_this_frame.append(fused)
        if fused_this_frame:
            frame_means.append(float(np.mean(fused_this_frame)))

    face_count = len(model_scores)
    if face_count == 0:
        return {"fake_probability": 0.10, "model_prob": 0.10, "freq_score": 0.0,
                "confidence": 0.4, "face_count": 0, "flicker": 0.0,
                "frame_scores": [], "no_faces": True}

    fused_prob = float(np.mean(frame_means))
    flicker = float(np.std(frame_means)) if len(frame_means) > 1 else 0.0
    return {
        "fake_probability": fused_prob,
        "model_prob": float(np.mean(model_scores)),
        "freq_score": float(np.mean(freq_scores)),
        "confidence": 0.8,
        "face_count": face_count,
        "flicker": flicker,
        "frame_scores": [round(s, 3) for s in frame_means],
        "no_faces": False,
    }


def _band(score: float) -> str:
    if score < config.DEEPFAKE_BANDS["low"]:
        return "LOW"
    return "MEDIUM" if score < config.DEEPFAKE_BANDS["high"] else "HIGH"


DISCLAIMER = ("This analysis provides probabilistic risk indicators, not factual "
              "determinations.")


def analyze_deepfake_risk(frame_paths: List[str]) -> Dict[str, Any]:
    if not frame_paths:
        return _empty("No frames to analyze.")

    r = compute_deepfake_score(frame_paths)
    fake_prob = r["fake_probability"]
    risk_score = float(fake_prob * 100)
    risk_level = _band(risk_score)
    no_faces = r.get("no_faces", False)
    face_count = r["face_count"]
    flicker = r["flicker"]

    forensic_status = "fail" if fake_prob > 0.65 else ("warn" if fake_prob > 0.35 else "pass")
    # High flicker (unstable face scores across frames) is itself a manipulation cue.
    temporal_status = "warn" if flicker > 0.15 else "pass"

    if no_faces:
        forensic_details = ("No faces were detected in the sampled frames, so face-based "
                            "deepfake manipulation could not be assessed. Other forms of "
                            "editing are not ruled out.")
        temporal_details = "No faces to evaluate for frame-to-frame stability."
        summary = (f"LOW RISK: No faces detected, so no face manipulation could be assessed. {DISCLAIMER}")
    else:
        forensic_details = (
            f"Fused {face_count} detected face(s): pretrained forgery classifier "
            f"{r['model_prob']*100:.1f}% and frequency-artifact score {r['freq_score']*100:.1f}%, "
            f"combined manipulation likelihood {fake_prob*100:.1f}%.")
        temporal_details = (
            f"Frame-to-frame score flicker {flicker*100:.1f}% "
            f"({'unstable, a possible face-swap cue' if flicker > 0.15 else 'stable across frames'}).")
        if risk_level == "HIGH":
            summary = (f"HIGH RISK: {fake_prob*100:.1f}% manipulation likelihood across {face_count} "
                       f"face(s); strong indicators of deepfake or face-swap editing. {DISCLAIMER}")
        elif risk_level == "MEDIUM":
            summary = (f"MEDIUM RISK: {fake_prob*100:.1f}% manipulation likelihood across {face_count} "
                       f"face(s). Further verification recommended. {DISCLAIMER}")
        else:
            summary = (f"LOW RISK: {fake_prob*100:.1f}% manipulation likelihood, consistent with "
                       f"authentic capture. {DISCLAIMER}")

    return {
        "riskScore": risk_score,
        "riskLevel": risk_level,
        "summary": summary,
        "signals": {
            "forensic": {"name": "Deepfake Manipulation Likelihood", "status": forensic_status,
                         "confidence": r["confidence"], "details": forensic_details},
            "temporal": {"name": "Temporal Face Coherence", "status": temporal_status,
                         "confidence": float(1.0 - min(flicker, 1.0)), "details": temporal_details},
        },
    }


def _empty(msg: str) -> Dict[str, Any]:
    return {
        "riskScore": 0, "riskLevel": "LOW", "summary": msg,
        "signals": {
            "forensic": {"name": "Deepfake Manipulation Likelihood", "status": "pass",
                         "confidence": 0.0, "details": "Insufficient data"},
            "temporal": {"name": "Temporal Face Coherence", "status": "pass",
                         "confidence": 0.0, "details": "No frames analyzed"},
        },
    }
