# -*- coding: utf-8 -*-
"""
Quantitative benchmark for the Context Integrity module (scene-vs-claim).

The module's distinctive signal is a continuous CLIP alignment between the frames
and the claim: analyze_contextual_consistency exposes it as `claim_alignment`, and
the internal _claim_alignment is exactly the code path exercised here. We turn a
captioned image set (COCO val) into an out-of-context detection task the way the
out-of-context media literature does: an image paired with its own caption is
in-context (label 0); the same image paired with a caption reassigned from a
different image is out-of-context (label 1). Two negative-mining strategies are
reported:

  * random   - the swapped caption is drawn uniformly from other images (easy);
  * hard      - the swapped caption is taken from the most visually similar other
                image whose caption is lexically different (a same-scene,
                different-event miscontextualisation, which is the realistic case).

We measure ROC-AUC (threshold free), plus accuracy / precision / recall at an
operating point chosen on a held-out split so nothing is tuned on the test pairs.
Every number is measured from the real module; nothing is invented.
"""
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "backend"))
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
DATA = HERE / "data" / "context"
RNG = np.random.RandomState(42)


def _load():
    caps = json.load(open(DATA / "captions.json"))
    names = sorted(caps.keys())
    paths = [str(DATA / "img" / n) for n in names]
    texts = [caps[n] for n in names]
    keep_p, keep_t = [], []
    for p, t in zip(paths, texts):
        if Path(p).exists():
            keep_p.append(p); keep_t.append(t)
    return keep_p, keep_t


def _tokens(s):
    return set(w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 2)


def _hard_negatives(emb, texts):
    """For each i, the most visually similar j whose caption is lexically different."""
    sims = emb @ emb.T
    np.fill_diagonal(sims, -1.0)
    tok = [_tokens(t) for t in texts]
    hard = np.empty(len(texts), dtype=int)
    for i in range(len(texts)):
        order = np.argsort(-sims[i])
        pick = -1
        for j in order:
            if sims[i, j] > 0.98:  # near-duplicate frame, skip
                continue
            inter = len(tok[i] & tok[j]); uni = len(tok[i] | tok[j]) or 1
            if inter / uni < 0.40:  # genuinely different caption
                pick = j; break
        hard[i] = pick if pick >= 0 else int(order[0])
    return hard


def _align_scores(paths, texts):
    """Out-of-context score = 1 - claim alignment, from the real backend signal."""
    from contextual_analysis import _claim_alignment
    out = np.empty(len(paths), dtype=np.float32)
    for k, (p, t) in enumerate(zip(paths, texts)):
        img = Image.open(p).convert("RGB")
        out[k] = 1.0 - _claim_alignment([img], t)
        if (k + 1) % 100 == 0:
            print(f"  scored {k + 1}/{len(paths)}", flush=True)
    return out


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


def _best_threshold(y, s):
    """Threshold that maximizes accuracy on the given (val) split."""
    order = np.unique(s)
    best_t, best_a = 0.5, -1.0
    for t in order:
        a = ((s >= t).astype(int) == y).mean()
        if a > best_a:
            best_a, best_t = a, t
    return float(best_t)


def _perturb(img, kind):
    if kind == "jpeg":
        b = io.BytesIO(); img.save(b, "JPEG", quality=35); b.seek(0); return Image.open(b).convert("RGB")
    if kind == "resize":
        w, h = img.size; s = img.resize((max(32, w // 3), max(32, h // 3))); return s.resize((w, h))
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(1.5))
    return img


def main():
    paths, texts = _load()
    n = len(paths)
    if n < 40:
        print(f"context: not enough data ({n})"); return None
    print(f"context: {n} captioned images", flush=True)

    from clip_backbone import image_embeddings
    emb = np.zeros((n, 0), dtype=np.float32)
    B = 64
    embs = []
    for s in range(0, n, B):
        imgs = [Image.open(p).convert("RGB") for p in paths[s:s + B]]
        embs.append(image_embeddings(imgs))
        print(f"  embedded {min(s + B, n)}/{n}", flush=True)
    emb = np.vstack(embs)

    # in-context (own caption) and two out-of-context minings.
    # random negative: a caption from another image (derangement, no i->i).
    rand_idx = (np.arange(n) + RNG.randint(1, n, size=n)) % n
    hard_idx = _hard_negatives(emb, texts)

    t0 = time.time()
    print("scoring in-context pairs ...", flush=True)
    s_pos = _align_scores(paths, texts)                              # label 0
    print("scoring random out-of-context pairs ...", flush=True)
    s_rand = _align_scores(paths, [texts[j] for j in rand_idx])      # label 1
    print("scoring hard out-of-context pairs ...", flush=True)
    s_hard = _align_scores(paths, [texts[j] for j in hard_idx])      # label 1
    dt = time.time() - t0
    n_pairs = 3 * n

    def evaluate(neg_scores, tag):
        y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
        s = np.concatenate([s_pos, neg_scores])
        a = auc(y, s)
        # operating point chosen on a val split, reported on the test split
        idx = RNG.permutation(len(y)); h = len(y) // 2
        va, te = idx[:h], idx[h:]
        thr = _best_threshold(y[va], s[va])
        pm = point_metrics(y[te], s[te], thr)
        return {"auc": a, "operating_point": pm,
                "mean_score_in_context": float(s_pos.mean()),
                "mean_score_out_of_context": float(neg_scores.mean())}

    res = {
        "task": "context",
        "n_images": int(n),
        "n_pairs": int(n_pairs),
        "dataset": "COCO val 2014 captions (sayakpaul/coco-30-val-2014)",
        "signal": "claim_alignment (CLIP scene-vs-claim), OOC score = 1 - alignment",
        "random_negatives": evaluate(s_rand, "random"),
        "hard_negatives": evaluate(s_hard, "hard"),
        "latency_ms_per_pair": 1000 * dt / n_pairs,
    }
    json.dump(res, open(RESULTS / "context_eval.json", "w"), indent=2)
    np.save(RESULTS / "context_scores.npy",
            np.column_stack([s_pos, s_rand, s_hard]))
    print("context:", json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
    print("CONTEXT EVAL DONE", flush=True)
