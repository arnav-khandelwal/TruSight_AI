# -*- coding: utf-8 -*-
"""
Download a subset of captioned images (COCO val) to build an out-of-context
benchmark for the Context Integrity module. Each image is saved with its own
(true) caption; the evaluation then pairs an image either with its own caption
(in-context) or with a caption reassigned from another image (out-of-context),
following the construction used for out-of-context media datasets. Honest data,
no fabrication.
"""
import io, json, sys
from pathlib import Path
from PIL import Image

OUT = Path(__file__).parent / "data" / "context"
IMG = OUT / "img"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 700
MAX_SIDE = 640


def _cap(v):
    if isinstance(v, (list, tuple)):
        v = v[0] if v else ""
    return str(v).strip()


def main():
    from datasets import load_dataset
    IMG.mkdir(parents=True, exist_ok=True)
    caps = {}
    have = len(list(IMG.glob("*.jpg")))
    print(f"[context] start with {have} images, target {N}", flush=True)
    ds = load_dataset("sayakpaul/coco-30-val-2014", split="train", streaming=True)
    i = have
    for ex in ds:
        if i >= N:
            break
        cap = _cap(ex.get("caption"))
        img = ex.get("image")
        if not cap or len(cap) < 8 or img is None:
            continue
        try:
            if not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img["bytes"])) if isinstance(img, dict) else img
            img = img.convert("RGB")
            if max(img.size) > MAX_SIDE:
                img.thumbnail((MAX_SIDE, MAX_SIDE))
            name = f"{i:04d}.jpg"
            img.save(IMG / name, quality=90)
            caps[name] = cap
            i += 1
            if i % 100 == 0:
                print(f"[context] {i} images", flush=True)
        except Exception:
            continue
    # merge with any existing captions
    cj = OUT / "captions.json"
    if cj.exists():
        caps = {**json.load(open(cj)), **caps}
    json.dump(caps, open(cj, "w"))
    print(f"[context] DONE {len(caps)} image-caption pairs", flush=True)


if __name__ == "__main__":
    main()
