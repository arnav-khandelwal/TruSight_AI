# -*- coding: utf-8 -*-
"""
Central configuration for TruSight AI: model identifiers, fusion weights, decision
thresholds, and sampling parameters. Keeping these in one place makes the pipeline
auditable and lets the evaluation (eval/) vary them without touching module code.

Fusion weights and thresholds below are the deployed defaults; eval/evaluate.py
reports the evidence (ROC operating points, ablations) that motivates them, so
they are documented choices rather than hidden magic numbers.
"""
import os

# ---- pretrained models (all open, no paid API) ------------------------------------------
DEEPFAKE_MODEL = os.environ.get("TRUSIGHT_DEEPFAKE_MODEL", "dima806/deepfake_vs_real_image_detection")
SYNTHETIC_MODELS = os.environ.get(
    "TRUSIGHT_SYNTHETIC_MODELS",
    "umm-maybe/AI-image-detector,Organika/sdxl-detector",  # ensembled by mean
).split(",")
CLIP_MODEL = os.environ.get("TRUSIGHT_CLIP_MODEL", "openai/clip-vit-base-patch32")

# ---- frame sampling ---------------------------------------------------------------------
DEEPFAKE_SAMPLE_INTERVAL_S = 2      # seconds between sampled frames (face pipeline)
SYNTHETIC_SAMPLE_INTERVAL_S = 1     # whole-frame pipeline samples denser
CONTEXT_SAMPLE_INTERVAL_S = 2
MAX_FRAMES = 40                     # hard cap so long clips stay bounded

# ---- deepfake module --------------------------------------------------------------------
# Final face score = w_model * classifier_prob + w_freq * frequency_artifact_score.
DEEPFAKE_FUSION = {"model": 0.75, "frequency": 0.25}
DEEPFAKE_BANDS = {"low": 35.0, "high": 65.0}       # <low = LOW, >=high = HIGH
FACE_SCORE_THRESHOLD = 0.6                          # YuNet detection confidence
MAX_FACES_PER_FRAME = 3

# ---- synthetic (AI-generated) module ----------------------------------------------------
# Final frame score = w_model * ensemble_prob + w_freq * frequency + w_ela * ELA.
# Evidence-based (eval/results/synthetic_eval.json): on the AI-art test set the
# hand-crafted cues did not improve the ensemble (AUC < 0.5), so the ensemble carries
# the score and the cues stay light as auxiliary evidence (they may help more on
# photo-realistic generators, which future work should test).
SYNTHETIC_FUSION = {"model": 0.85, "frequency": 0.08, "ela": 0.07}
SYNTHETIC_BANDS = {"low": 35.0, "high": 65.0}

# ---- context-integrity module -----------------------------------------------------------
CONTEXT_FUSION = {"temporal": 0.50, "contextual": 0.30, "environmental": 0.20}
CONTEXT_BANDS = {"low": 30.0, "high": 70.0}
REUSE_SIMILARITY_THRESHOLD = 0.90   # cosine-sim above this = near-duplicate of reference
REUSE_MATCH_RATIO = 0.30            # fraction of frames matching to flag reuse

# ---- calibration ------------------------------------------------------------------------
# Path to an optional fitted calibrator (isotonic/Platt) produced by eval/calibrate.py.
CALIBRATION_PATH = os.environ.get("PROV_CALIBRATION", os.path.join(os.path.dirname(__file__), "models", "calibration.json"))
