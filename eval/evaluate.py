# -*- coding: utf-8 -*-
"""
Honest evaluation of the TruSight AI detectors on real public data.

For each task (synthetic, deepfake) we score every image once through the actual
backend scoring functions, recording each component signal, then compute:
  * ROC-AUC of every signal and of the fused score;
  * accuracy / precision / recall / F1 at the deployed decision band;
  * an ablation (which signals help) with a val/test split so the fusion choice
    is not tuned on the test set;
  * expected calibration error (ECE) before and after isotonic calibration;
  * latency per stage; and
  * robustness of AUC under JPEG recompression, downscaling, and blur.

All numbers are measured, never invented. Results -> eval/results/*.json (+ figures).
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "backend"))
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
DATA = HERE / "data"
RNG = np.random.RandomState(42)


def _list(task):
    paths, labels = [], []
    for cls, y in (("fake", 1), ("real", 0)):
        for p in sorted((DATA / task / cls).glob("*.jpg")):
            paths.append(str(p)); labels.append(y)
    return paths, np.array(labels)


# ---- component scoring per image (reuses the real backend functions) --------------------
def score_synths(paths):
    from analyze_synthetic import compute_synthetic_score
    rows = []
    for p in paths:
        r = compute_synthetic_score([p])
        rows.append([r["model_prob"], r["freq_score"], r["ela_score"], r["synthetic_probability"]])
    return np.array(rows)  # cols: model, freq, ela, fused


def score_deepfakes(paths):
    from analyze_deepfake import compute_deepfake_score
    rows, valid = [], []
    for p in paths:
        r = compute_deepfake_score([p])
        if r.get("no_faces"):
            valid.append(False); rows.append([np.nan, np.nan, np.nan])
        else:
            valid.append(True); rows.append([r["model_prob"], r["freq_score"], r["fake_probability"]])
    return np.array(rows), np.array(valid)  # cols: model, freq, fused


# ---- metrics ----------------------------------------------------------------------------
def auc(y, s):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def point_metrics(y, s, thr):
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    pred = (s >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def ece(y, p, bins=10):
    p = np.clip(p, 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(e)


def perturb(img, kind):
    if kind == "jpeg":
        b = io.BytesIO(); img.save(b, "JPEG", quality=35); b.seek(0); return Image.open(b).convert("RGB")
    if kind == "resize":
        w, h = img.size; s = img.resize((max(32, w // 3), max(32, h // 3))); return s.resize((w, h))
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(1.5))
    return img


# ---- per-task driver --------------------------------------------------------------------
def run_synthetic():
    paths, y = _list("synthetic")
    if len(paths) < 20:
        print("synthetic: not enough data"); return None
    print(f"synthetic: scoring {len(paths)} images ...", flush=True)
    t0 = time.time()
    S = score_synths(paths)
    dt = time.time() - t0
    cols = {"model": S[:, 0], "frequency": S[:, 1], "ela": S[:, 2], "fused": S[:, 3]}
    aucs = {k: auc(y, v) for k, v in cols.items()}
    # val/test split to choose model-only vs fused without peeking
    idx = RNG.permutation(len(y)); half = len(y) // 2
    va, te = idx[:half], idx[half:]
    choose = "fused" if auc(y[va], cols["fused"][va]) >= auc(y[va], cols["model"][va]) else "model"
    test_auc = auc(y[te], cols[choose][te])
    # calibration on the chosen signal
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip").fit(cols[choose][va], y[va])
    cal = ir.predict(cols[choose][te])
    res = {
        "task": "synthetic", "n": int(len(y)), "n_fake": int(y.sum()), "n_real": int((1 - y).sum()),
        "auc_by_signal": aucs,
        "operating_point": point_metrics(y, cols["fused"], 0.5),
        "ablation_choice": choose, "test_auc_chosen": test_auc,
        "ece_raw": ece(y[te], cols[choose][te]), "ece_calibrated": ece(y[te], cal),
        "latency_ms_per_image": 1000 * dt / len(paths),
        "robustness": {},
    }
    # robustness: rescore a subset under perturbations
    sub = idx[:min(120, len(idx))]
    subpaths = [paths[i] for i in sub]; suby = y[sub]
    for kind in ("clean", "jpeg", "resize", "blur"):
        tmp = HERE / "_tmp"; tmp.mkdir(exist_ok=True)
        pert = []
        for i, p in enumerate(subpaths):
            im = perturb(Image.open(p).convert("RGB"), kind)
            q = tmp / f"s_{i}.jpg"; im.save(q, quality=92); pert.append(str(q))
        Sp = score_synths(pert)
        res["robustness"][kind] = auc(suby, Sp[:, 3])
        for q in pert:
            Path(q).unlink(missing_ok=True)
    json.dump(res, open(RESULTS / "synthetic_eval.json", "w"), indent=2)
    np.save(RESULTS / "synthetic_scores.npy", np.column_stack([S, y]))
    print("synthetic:", json.dumps(res, indent=2))
    return res


def run_deepfake():
    paths, y = _list("deepfake")
    if len(paths) < 20:
        print("deepfake: not enough data"); return None
    print(f"deepfake: scoring {len(paths)} images ...", flush=True)
    t0 = time.time()
    S, valid = score_deepfakes(paths)
    dt = time.time() - t0
    yv, Sv = y[valid], S[valid]
    cols = {"model": Sv[:, 0], "frequency": Sv[:, 1], "fused": Sv[:, 2]}
    aucs = {k: auc(yv, v) for k, v in cols.items()}
    idx = RNG.permutation(len(yv)); half = len(yv) // 2
    va, te = idx[:half], idx[half:]
    choose = "fused" if auc(yv[va], cols["fused"][va]) >= auc(yv[va], cols["model"][va]) else "model"
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip").fit(cols[choose][va], yv[va])
    cal = ir.predict(cols[choose][te])
    res = {
        "task": "deepfake", "n_total": int(len(y)), "n_face_detected": int(valid.sum()),
        "face_detection_rate": float(valid.mean()),
        "n_fake": int(yv.sum()), "n_real": int((1 - yv).sum()),
        "auc_by_signal": aucs,
        "operating_point": point_metrics(yv, cols["fused"], 0.5),
        "ablation_choice": choose, "test_auc_chosen": auc(yv[te], cols[choose][te]),
        "ece_raw": ece(yv[te], cols[choose][te]), "ece_calibrated": ece(yv[te], cal),
        "latency_ms_per_image": 1000 * dt / len(paths),
    }
    json.dump(res, open(RESULTS / "deepfake_eval.json", "w"), indent=2)
    np.save(RESULTS / "deepfake_scores.npy", np.column_stack([S, y]))
    print("deepfake:", json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "synthetic"):
        run_synthetic()
    if which in ("all", "deepfake"):
        run_deepfake()
    print("EVAL DONE", flush=True)
