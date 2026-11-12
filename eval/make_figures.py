# -*- coding: utf-8 -*-
"""
Publication figures from the saved evaluation scores (eval/results/*.npy, *.json).
Produces ROC curves, a per-signal AUC bar chart, a calibration reliability diagram,
and a robustness bar chart. All from measured scores; nothing invented.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.isotonic import IsotonicRegression

R = Path(__file__).parent / "results"
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def _roc(ax, y, s, label):
    if len(np.unique(y)) < 2:
        return
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, label=f"{label} (AUC {roc_auc_score(y, s):.2f})")


def roc_combined():
    """Both detectors' ROC in a single side-by-side image (one file, so it renders
    cleanly in Word as well as LaTeX)."""
    ps = R / "synthetic_scores.npy"
    pd_ = R / "deepfake_scores.npy"
    if not (ps.exists() and pd_.exists()):
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.2))
    A = np.load(ps); y = A[:, -1].astype(int)
    for j, name in [(0, "Ensemble classifier"), (1, "Frequency"), (2, "ELA"), (3, "Fused")]:
        _roc(a1, y, A[:, j], name)
    a1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate")
    a1.set_title("AI-generated detection"); a1.legend(loc="lower right", fontsize=8)
    B = np.load(pd_); m = ~np.isnan(B[:, 0]); B = B[m]; yb = B[:, -1].astype(int)
    for j, name in [(0, "Forgery classifier"), (1, "Frequency"), (2, "Fused")]:
        _roc(a2, yb, B[:, j], name)
    a2.plot([0, 1], [0, 1], "k--", alpha=0.4)
    a2.set_xlabel("False positive rate"); a2.set_ylabel("True positive rate")
    a2.set_title("Deepfake detection"); a2.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(R / "roc_both.png"); plt.close(fig)


def roc_figures():
    # synthetic: cols model, freq, ela, fused, label
    p = R / "synthetic_scores.npy"
    if p.exists():
        A = np.load(p); y = A[:, -1].astype(int)
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        for j, name in [(0, "Ensemble classifier"), (1, "Frequency"), (2, "ELA"), (3, "Fused")]:
            _roc(ax, y, A[:, j], name)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("AI-generated detection (ROC)"); ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout(); fig.savefig(R / "roc_synthetic.png"); plt.close(fig)

    # deepfake: cols model, freq, fused, label (NaN rows = no face)
    p = R / "deepfake_scores.npy"
    if p.exists():
        A = np.load(p); m = ~np.isnan(A[:, 0]); A = A[m]; y = A[:, -1].astype(int)
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        for j, name in [(0, "Forgery classifier"), (1, "Frequency"), (2, "Fused")]:
            _roc(ax, y, A[:, j], name)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("Deepfake detection (ROC)"); ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout(); fig.savefig(R / "roc_deepfake.png"); plt.close(fig)


def auc_bar():
    rows = []
    for task, f in (("AI-generated", "synthetic_eval.json"), ("Deepfake", "deepfake_eval.json")):
        jp = R / f
        if jp.exists():
            d = json.load(open(jp))
            for sig, a in d["auc_by_signal"].items():
                rows.append((task, sig, a))
    if not rows:
        return
    tasks = sorted({r[0] for r in rows})
    sigs = ["model", "frequency", "ela", "fused"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    width = 0.8 / max(1, len(tasks))
    for ti, task in enumerate(tasks):
        vals = [next((a for (t, s, a) in rows if t == task and s == sig), np.nan) for sig in sigs]
        xs = [i + ti * width for i in range(len(sigs))]
        ax.bar(xs, vals, width=width, label=task)
    ax.set_xticks([i + width * (len(tasks) - 1) / 2 for i in range(len(sigs))])
    ax.set_xticklabels(["Classifier", "Frequency", "ELA", "Fused"])
    ax.axhline(0.5, color="k", ls="--", alpha=0.4)
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Per-signal discrimination"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(R / "auc_bars.png"); plt.close(fig)


def calibration_fig():
    p = R / "synthetic_scores.npy"
    if not p.exists():
        return
    A = np.load(p); y = A[:, -1].astype(int); s = A[:, 3]
    rng = np.random.RandomState(0); idx = rng.permutation(len(y)); h = len(y) // 2
    va, te = idx[:h], idx[h:]
    ir = IsotonicRegression(out_of_bounds="clip").fit(s[va], y[va])
    cal = ir.predict(s[te])
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for probs, name in ((s[te], "raw fused"), (cal, "calibrated")):
        edges = np.linspace(0, 1, 11); xs, ys = [], []
        for i in range(10):
            m = (probs >= edges[i]) & (probs < edges[i + 1] if i < 9 else probs <= edges[i + 1])
            if m.sum():
                xs.append(probs[m].mean()); ys.append(y[te][m].mean())
        ax.plot(xs, ys, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Empirical fraction fake")
    ax.set_title("Calibration (AI-generated)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(R / "calibration.png"); plt.close(fig)


def _ooc_roc(ax, A, title):
    s_pos, s_rand, s_hard = A[:, 0], A[:, 1], A[:, 2]
    n = len(s_pos)
    for neg, name in ((s_rand, "Random mismatch"), (s_hard, "Hard mismatch")):
        y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
        _roc(ax, y, np.concatenate([s_pos, neg]), name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title); ax.legend(loc="lower right", fontsize=8)


def context_fig():
    """Out-of-context detection ROC for the context-integrity module, on the still-image
    benchmark (left) and, when available, the real MSR-VTT video benchmark (right)."""
    pc = R / "context_scores.npy"
    pv = R / "video_scores.npy"
    if not pc.exists():
        return
    if pv.exists():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
        _ooc_roc(a1, np.load(pc), "Image-claim benchmark")
        _ooc_roc(a2, np.load(pv), "Video-claim benchmark (MSR-VTT)")
    else:
        fig, a1 = plt.subplots(1, 1, figsize=(5.0, 4.0))
        _ooc_roc(a1, np.load(pc), "Out-of-context detection")
    fig.tight_layout(); fig.savefig(R / "context_bench.png"); plt.close(fig)


def robustness_fig():
    jp = R / "synthetic_eval.json"
    if not jp.exists():
        return
    d = json.load(open(jp)); rob = d.get("robustness", {})
    if not rob:
        return
    order = ["clean", "jpeg", "resize", "blur"]
    labels = ["Clean", "JPEG q35", "Down x3", "Blur"]
    vals = [rob.get(k, np.nan) for k in order]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.bar(labels, vals, color="#4C72B0")
    ax.axhline(0.5, color="k", ls="--", alpha=0.4)
    ax.set_ylabel("Fused ROC-AUC"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Robustness to perturbations (AI-generated)")
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(R / "robustness.png"); plt.close(fig)


if __name__ == "__main__":
    roc_figures(); roc_combined(); auc_bar(); calibration_fig(); robustness_fig(); context_fig()
    print("figures written to", R)
