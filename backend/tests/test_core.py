# -*- coding: utf-8 -*-
"""
Unit tests for the pure logic of the TruSight AI backend: forensic signals, band
decisions, label parsing, claim parsing, and the honest 'no reference database'
behaviour. These do not download any model, so CI runs in seconds.
"""
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---- forensic signals -------------------------------------------------------------------
def _img(seed=0):
    rng = np.random.RandomState(seed)
    return Image.fromarray((rng.rand(128, 128, 3) * 255).astype("uint8"))


def test_frequency_score_range_and_determinism():
    import forensic_signals as fs
    s1, feat = fs.frequency_artifact_score(_img(1))
    s2, _ = fs.frequency_artifact_score(_img(1))
    assert 0.0 <= s1 <= 1.0
    assert s1 == s2                      # deterministic
    assert "hf_ratio" in feat and "peakiness" in feat


def test_ela_score_range():
    import forensic_signals as fs
    s, feat = fs.ela_artifact_score(_img(2))
    assert 0.0 <= s <= 1.0
    assert feat["ela_mean"] >= 0.0


def test_flat_image_has_low_highfreq():
    import forensic_signals as fs
    flat = Image.fromarray(np.full((128, 128, 3), 128, dtype="uint8"))
    noisy = _img(3)
    assert fs.radial_highfreq_ratio(flat) <= fs.radial_highfreq_ratio(noisy)


# ---- band / risk decisions --------------------------------------------------------------
def test_deepfake_bands():
    import analyze_deepfake as d
    assert d._band(10) == "LOW"
    assert d._band(50) == "MEDIUM"
    assert d._band(80) == "HIGH"


def test_synthetic_bands():
    import analyze_synthetic as s
    assert s._band(10) == "LOW"
    assert s._band(50) == "MEDIUM"
    assert s._band(90) == "HIGH"


def test_label_parsers():
    import analyze_deepfake as d
    import analyze_synthetic as s
    assert d._fake_prob([{"label": "FAKE", "score": 0.9}, {"label": "REAL", "score": 0.1}]) == 0.9
    assert s._artificial_prob([{"label": "artificial", "score": 0.7}, {"label": "human", "score": 0.3}]) == 0.7


# ---- claim / scene parsing --------------------------------------------------------------
def test_expected_scene_keywords():
    import contextual_analysis as c
    assert c.determine_expected_scene("a massive protest downtown") == "protest"
    assert c.determine_expected_scene("wildfire smoke everywhere") == "fire"
    assert c.determine_expected_scene("a cat on a sofa") == "normal"


def test_time_extraction():
    import environmental_analysis as e
    assert e.extract_time_from_claim("filmed at night") == "night"
    assert e.determine_expected_lighting("night") == "night"
    assert e.classify_lighting(200) == "day"
    assert e.classify_lighting(20) == "night"


# ---- temporal reuse honesty -------------------------------------------------------------
def test_no_reference_db_is_honest_not_random():
    import temporal_analysis as t
    ref = t._load_reference()
    assert ref.shape[0] == 0            # no random vectors when DB is absent


def test_intra_clip_duplication_detects_repeat():
    import temporal_analysis as t
    base = np.random.RandomState(0).randn(512).astype("float32")
    base /= np.linalg.norm(base)
    other = np.random.RandomState(1).randn(512).astype("float32")
    other /= np.linalg.norm(other)
    embs = np.stack([base, other, base])          # frame 0 and 2 identical, non-adjacent
    assert t._intra_clip_duplication(embs) > 0.99


# ---- config sanity ----------------------------------------------------------------------
def test_fusion_weights_sum_to_one():
    import config
    for fusion in (config.DEEPFAKE_FUSION, config.SYNTHETIC_FUSION, config.CONTEXT_FUSION):
        assert abs(sum(fusion.values()) - 1.0) < 1e-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
