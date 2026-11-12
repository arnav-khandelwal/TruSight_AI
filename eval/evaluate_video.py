# -*- coding: utf-8 -*-
"""
Real VIDEO-claim out-of-context benchmark for the context-integrity module.

Unlike evaluate_context.py (single-image proxy), this runs the deployed multi-frame
path, analyze_contextual_consistency(frame_paths, claim), over genuine video clips
from MSR-VTT (in-the-wild YouTube footage with human captions). Each clip paired with
its own caption is in-context; paired with a caption reassigned from another clip it is
out-of-context, mined both at random and hard (from the most visually similar other
clip whose caption is lexically different). We report ROC-AUC of the module's
claim-alignment signal and operating-point metrics. All numbers measured.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "backend"))
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
DATA = HERE / "data" / "video"
FR = DATA / "frames"
RNG = np.random.RandomState(42)


def _load():
    caps = json.load(open(DATA / "captions.json"))
    data = []
    for vid in sorted(caps.keys()):
        fps = sorted((FR / vid).glob("f*.jpg"))
        if len(fps) >= 2:
            data.append((vid, [str(p) for p in fps], str(caps[vid])))
    return data


def _tokens(s):
    return set(w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 2)


def _video_embeddings(data):
    from clip_backbone import image_embeddings
    embs = []
    for k, (vid, fps, _) in enumerate(data):
        imgs = [Image.open(p).convert("RGB") for p in fps]
        e = image_embeddings(imgs).mean(0)
        embs.append(e / (np.linalg.norm(e) + 1e-8))
        if (k + 1) % 200 == 0:
            print(f"  embedded {k + 1}/{len(data)}", flush=True)
    return np.vstack(embs).astype(np.float32)


def _hard_negatives(emb, texts):
    sims = emb @ emb.T
    np.fill_diagonal(sims, -1.0)
    tok = [_tokens(t) for t in texts]
    hard = np.empty(len(texts), dtype=int)
    for i in range(len(texts)):
        order = np.argsort(-sims[i])
        pick = -1
        for j in order:
            if sims[i, j] > 0.98:
                continue
            inter = len(tok[i] & tok[j]); uni = len(tok[i] | tok[j]) or 1
            if inter / uni < 0.40:
                pick = j; break
        hard[i] = pick if pick >= 0 else int(order[0])
    return hard


def _score(data, captions_for_each):
    """Out-of-context score = 1 - claim alignment, from the deployed multi-frame module."""
    from contextual_analysis import analyze_contextual_consistency
    out = np.empty(len(data), dtype=np.float32)
    for k, (vid, fps, _) in enumerate(data):
        r = analyze_contextual_consistency(fps, captions_for_each[k])
        out[k] = 1.0 - float(r.get("claim_alignment", 0.0))
        if (k + 1) % 100 == 0:
            print(f"  scored {k + 1}/{len(data)}", flush=True)
    return out


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def point_metrics(y, s, thr):
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    pred = (s >= thr).astype(int)
    return {"threshold": float(thr), "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0))}


def _best_threshold(y, s):
    best_t, best_a = 0.5, -1.0
    for t in np.unique(s):
        a = ((s >= t).astype(int) == y).mean()
        if a > best_a:
            best_a, best_t = a, t
    return float(best_t)


def main():
    data = _load()
    n = len(data)
    if n < 40:
        print(f"video: not enough data ({n})"); return None
    texts = [d[2] for d in data]
    n_frames = int(np.mean([len(d[1]) for d in data]))
    print(f"video: {n} clips, ~{n_frames} frames each", flush=True)

    emb = _video_embeddings(data)
    rand_idx = (np.arange(n) + RNG.randint(1, n, size=n)) % n
    hard_idx = _hard_negatives(emb, texts)

    t0 = time.time()
    print("scoring in-context clips ...", flush=True)
    s_pos = _score(data, texts)
    print("scoring random out-of-context clips ...", flush=True)
    s_rand = _score(data, [texts[j] for j in rand_idx])
    print("scoring hard out-of-context clips ...", flush=True)
    s_hard = _score(data, [texts[j] for j in hard_idx])
    dt = time.time() - t0

    def evaluate(neg):
        y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
        s = np.concatenate([s_pos, neg])
        idx = RNG.permutation(len(y)); h = len(y) // 2
        thr = _best_threshold(y[idx[:h]], s[idx[:h]])
        return {"auc": auc(y, s), "operating_point": point_metrics(y[idx[h:]], s[idx[h:]], thr),
                "mean_score_in_context": float(s_pos.mean()),
                "mean_score_out_of_context": float(neg.mean())}

    res = {
        "task": "video_context",
        "n_clips": int(n), "n_pairs": int(3 * n), "frames_per_clip": n_frames,
        "dataset": "MSR-VTT 1k-A test split (friedrichor/MSR-VTT), in-the-wild YouTube clips",
        "signal": "claim_alignment from analyze_contextual_consistency (deployed multi-frame path)",
        "random_negatives": evaluate(s_rand),
        "hard_negatives": evaluate(s_hard),
        "latency_ms_per_clip_scoring": 1000 * dt / (3 * n),
    }
    json.dump(res, open(RESULTS / "video_eval.json", "w"), indent=2)
    np.save(RESULTS / "video_scores.npy", np.column_stack([s_pos, s_rand, s_hard]))
    print("video:", json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
    print("VIDEO EVAL DONE", flush=True)
