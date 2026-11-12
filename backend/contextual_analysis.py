# -*- coding: utf-8 -*-
"""
Contextual consistency: does the scene in the video match the user's claim?

Two complementary CLIP signals (shared backbone, no second model load):

  1. Category match - the frames are zero-shot classified over a set of event
     scene categories, and the dominant category is compared with the category
     implied by the claim's keywords.
  2. Direct claim alignment - a continuous CLIP score of how well the frames match
     a short description built from the claim versus a generic 'unrelated ordinary
     scene', which generalises beyond the fixed category list.

A mismatch (claim implies a specific event the footage does not show) flags a
contextual inconsistency, with a confidence driven by the alignment gap.
"""
import re
from typing import List, Dict, Any

from PIL import Image

from clip_backbone import zero_shot

# Scene category -> natural-language prompt, plus claim keywords.
SCENE_PROMPTS = {
    "protest": "a protest, demonstration, rally, or large crowd marching",
    "fire": "a fire with flames, smoke, or a burning building",
    "flood": "a flood with water covering streets or land",
    "accident": "a vehicle accident, crash, or wreck",
    "conflict": "armed conflict, soldiers, tanks, or an explosion in a war zone",
    "disaster": "a natural disaster such as an earthquake, storm, or collapsed building",
    "celebration": "a festival, parade, concert, or celebration",
    "normal": "an ordinary, everyday scene with nothing unusual",
}
SCENE_KEYWORDS = {
    "protest": ["protest", "demonstration", "rally", "march", "crowd", "riot"],
    "fire": ["fire", "burning", "flame", "smoke", "blaze", "wildfire"],
    "flood": ["flood", "flooding", "inundation", "submerged", "deluge"],
    "accident": ["accident", "crash", "collision", "wreck"],
    "conflict": ["war", "conflict", "soldier", "military", "tank", "airstrike", "explosion", "bomb"],
    "disaster": ["earthquake", "storm", "hurricane", "tornado", "landslide", "collapse", "tsunami"],
    "celebration": ["festival", "parade", "concert", "celebration", "carnival"],
}
_PROMPT_TO_SCENE = {v: k for k, v in SCENE_PROMPTS.items()}


def _classify_scene(img: Image.Image) -> str:
    probs = zero_shot(img, list(SCENE_PROMPTS.values()))
    top = max(probs, key=probs.get)
    return _PROMPT_TO_SCENE.get(top, "normal")


def determine_expected_scene(claim: str) -> str:
    cl = claim.lower()
    for category, keywords in SCENE_KEYWORDS.items():
        if any(k in cl for k in keywords):
            return category
    return "normal"


def _claim_alignment(imgs: List[Image.Image], claim: str) -> float:
    """Mean CLIP probability that frames show the claim vs an unrelated scene."""
    short = " ".join(re.findall(r"[a-zA-Z]+", claim))[:180] or claim
    labels = [f"a photo showing {short}", "an unrelated, ordinary, everyday scene"]
    vals = [zero_shot(im, labels)[labels[0]] for im in imgs]
    return float(sum(vals) / len(vals)) if vals else 0.0


def analyze_contextual_consistency(frame_paths: List[str], claim: str) -> Dict[str, Any]:
    if not frame_paths:
        return {"inconsistent": False, "confidence": 0.0, "details": "No frames to analyze",
                "detected_scenes": [], "expected_scene": "unknown"}
    if not claim or len(claim.strip()) < 5:
        return {"inconsistent": False, "confidence": 0.0, "details": "No claim provided for comparison",
                "detected_scenes": [], "expected_scene": "unknown"}

    imgs = []
    for p in frame_paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:  # noqa: BLE001
            continue
    if not imgs:
        return {"inconsistent": False, "confidence": 0.0, "details": "Frames could not be read",
                "detected_scenes": [], "expected_scene": "unknown"}

    expected = determine_expected_scene(claim)
    detected = [_classify_scene(im) for im in imgs]
    counts: Dict[str, int] = {}
    for s in detected:
        counts[s] = counts.get(s, 0) + 1
    dominant = max(counts, key=counts.get)
    dominant_ratio = counts[dominant] / len(detected)
    alignment = _claim_alignment(imgs, claim)

    # Inconsistent when the claim implies a specific (non-normal) event the footage
    # does not predominantly show, OR when direct claim alignment is weak.
    category_mismatch = (expected != "normal") and (dominant != expected)
    weak_alignment = (expected != "normal") and (alignment < 0.5)
    inconsistent = bool(category_mismatch or weak_alignment)
    confidence = float(round(min(1.0, dominant_ratio * (1.0 - alignment) + (0.3 if category_mismatch else 0.0)), 3)) if inconsistent else float(round(alignment, 3))

    if inconsistent:
        details = (f"The claim implies '{expected}', but the footage predominantly shows "
                   f"'{dominant}' (claim-alignment {alignment:.2f}). Scene distribution: {counts}.")
    else:
        details = (f"Footage is consistent with the claim (expected '{expected}', "
                   f"claim-alignment {alignment:.2f}). Scene distribution: {counts}.")

    return {
        "inconsistent": inconsistent,
        "confidence": confidence,
        "details": details,
        "detected_scenes": detected,
        "expected_scene": expected,
        "dominant_scene": dominant,
        "claim_alignment": alignment,
    }
