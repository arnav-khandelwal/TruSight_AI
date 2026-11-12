# -*- coding: utf-8 -*-
"""
Cross-dataset generalization: score the SAME detectors (no re-tuning, same fusion
weights and operating band) on a second, independently sourced dataset per task and
report ROC-AUC and the operating-point metrics. This tests whether the training-free
signals transfer beyond the primary evaluation set. All numbers measured.

  deepfake_x  -> analyze_deepfake on JamieWithofs/Deepfake-and-real-images
  synthetic_x -> analyze_synthetic on CIFAKE (small-image, different generator)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "backend"))
RESULTS = HERE / "results"
DATA = HERE / "data"

import evaluate as ev  # reuse _list-like logic, scorers, metrics


def _list(task):
    paths, labels = [], []
    for cls, y in (("fake", 1), ("real", 0)):
        for p in sorted((DATA / task / cls).glob("*.jpg")):
            paths.append(str(p)); labels.append(y)
    return paths, np.array(labels)


def run_synthetic_x():
    paths, y = _list("synthetic_x")
    if len(paths) < 20:
        print("synthetic_x: not enough data"); return None
    print(f"synthetic_x: scoring {len(paths)} images ...", flush=True)
    t0 = time.time()
    S = ev.score_synths(paths)
    dt = time.time() - t0
    cols = {"model": S[:, 0], "frequency": S[:, 1], "ela": S[:, 2], "fused": S[:, 3]}
    res = {
        "task": "synthetic_x", "dataset": "CIFAKE (32x32, Stable Diffusion vs CIFAR-10)",
        "n": int(len(y)), "n_fake": int(y.sum()), "n_real": int((1 - y).sum()),
        "auc_by_signal": {k: ev.auc(y, v) for k, v in cols.items()},
        "operating_point": ev.point_metrics(y, cols["fused"], 0.5),
        "latency_ms_per_image": 1000 * dt / len(paths),
    }
    json.dump(res, open(RESULTS / "synthetic_x_eval.json", "w"), indent=2)
    np.save(RESULTS / "synthetic_x_scores.npy", np.column_stack([S, y]))
    print("synthetic_x:", json.dumps(res, indent=2))
    return res


def run_deepfake_x():
    paths, y = _list("deepfake_x")
    if len(paths) < 20:
        print("deepfake_x: not enough data"); return None
    print(f"deepfake_x: scoring {len(paths)} images ...", flush=True)
    t0 = time.time()
    S, valid = ev.score_deepfakes(paths)
    dt = time.time() - t0
    yv, Sv = y[valid], S[valid]
    cols = {"model": Sv[:, 0], "frequency": Sv[:, 1], "fused": Sv[:, 2]}
    res = {
        "task": "deepfake_x", "dataset": "JamieWithofs/Deepfake-and-real-images (256x256)",
        "n_total": int(len(y)), "n_face_detected": int(valid.sum()),
        "face_detection_rate": float(valid.mean()),
        "n_fake": int(yv.sum()), "n_real": int((1 - yv).sum()),
        "auc_by_signal": {k: ev.auc(yv, v) for k, v in cols.items()},
        "operating_point": ev.point_metrics(yv, cols["fused"], 0.5),
        "latency_ms_per_image": 1000 * dt / len(paths),
    }
    json.dump(res, open(RESULTS / "deepfake_x_eval.json", "w"), indent=2)
    np.save(RESULTS / "deepfake_x_scores.npy", np.column_stack([S, y]))
    print("deepfake_x:", json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "deepfake"):
        run_deepfake_x()
    if which in ("all", "synthetic"):
        run_synthetic_x()
    print("XDATASET EVAL DONE", flush=True)
