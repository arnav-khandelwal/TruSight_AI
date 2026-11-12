# -*- coding: utf-8 -*-
"""
Download bounded, labelled evaluation subsets for the TruSight AI detectors, saving
resized JPEGs to eval/data/<task>/{real,fake}/. Labels are mapped from each
dataset's declared class NAMES (never guessed from integers), so the ground truth
is provably correct. Honest evaluation only: real public datasets, no fabrication.

  synthetic : real photos/art vs AI-generated images  (tests analyze_synthetic)
  deepfake  : real faces vs deepfake faces             (tests analyze_deepfake)
"""
import io, sys
from pathlib import Path
from PIL import Image

OUT = Path(__file__).parent / "data"
N_PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 250
MAX_SIDE = 768   # downscale large frames to keep the set small and fast


def _to_pil(v):
    if isinstance(v, Image.Image):
        return v
    if isinstance(v, dict) and v.get("bytes"):
        return Image.open(io.BytesIO(v["bytes"]))
    if isinstance(v, (bytes, bytearray)):
        return Image.open(io.BytesIO(v))
    return None


def _class_of(label_name: str):
    n = str(label_name).lower()
    if any(k in n for k in ("fake", "ai", "synth", "deepfake", "generated")):
        return "fake"
    if any(k in n for k in ("real", "authentic", "natural", "genuine")):
        return "real"
    return None


def harvest(task, candidates, n=N_PER_CLASS):
    from datasets import load_dataset
    outp = OUT / task
    (outp / "real").mkdir(parents=True, exist_ok=True)
    (outp / "fake").mkdir(parents=True, exist_ok=True)
    have = {c: len(list((outp / c).glob("*.jpg"))) for c in ("real", "fake")}
    print(f"[{task}] start real={have['real']} fake={have['fake']} target={n}/class", flush=True)

    for name, split in candidates:
        if have["real"] >= n and have["fake"] >= n:
            break
        try:
            ds = load_dataset(name, split=split, streaming=True)
            names = ds.features["label"].names
            idx2cls = {i: _class_of(nm) for i, nm in enumerate(names)}
            print(f"  [{task}] {name}: label map {dict(enumerate(names))} -> {idx2cls}", flush=True)
        except Exception as e:
            print(f"  [{task}] skip {name}: {type(e).__name__} {str(e)[:90]}", flush=True)
            continue
        got = 0
        try:
            for ex in ds:
                if have["real"] >= n and have["fake"] >= n:
                    break
                cls = idx2cls.get(ex.get("label"))
                if cls is None or have[cls] >= n:
                    continue
                img = _to_pil(ex.get("image"))
                if img is None:
                    continue
                try:
                    img = img.convert("RGB")
                    if max(img.size) > MAX_SIDE:
                        img.thumbnail((MAX_SIDE, MAX_SIDE))
                    img.save(outp / cls / f"{cls}_{have[cls]:04d}.jpg", quality=90)
                    have[cls] += 1
                    got += 1
                except Exception:
                    continue
        except Exception as e:
            print(f"  [{task}] stream error {name}: {type(e).__name__} {str(e)[:90]}", flush=True)
        print(f"  [{task}] {name}: real={have['real']} fake={have['fake']} (+{got})", flush=True)
    print(f"[{task}] DONE real={have['real']} fake={have['fake']}", flush=True)
    return have


SYNTH = [
    ("Hemg/AI-Generated-vs-Real-Images-Datasets", "train"),   # 880x440, AiArtData/RealArt
    ("dragonintelligence/CIFAKE-image-dataset", "test"),      # 32x32 fallback, FAKE/REAL
]
DEEPFAKE = [
    ("prithivMLmods/OpenDeepfake-Preview", "train"),          # 1080x1080, fake/real
    ("JamieWithofs/Deepfake-and-real-images", "train"),       # 256x256, Fake/Real
]

if __name__ == "__main__":
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    if which in ("all", "synthetic"):
        harvest("synthetic", SYNTH)
    if which in ("all", "deepfake"):
        harvest("deepfake", DEEPFAKE)
    print("ALL DONE", flush=True)
