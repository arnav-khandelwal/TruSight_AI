# -*- coding: utf-8 -*-
"""
Build a real VIDEO-claim out-of-context benchmark set from MSR-VTT, the standard
video-captioning corpus of in-the-wild YouTube clips with human captions. We use
the recognised 1k-A test split (1000 clips). Each clip is decoded with OpenCV and
a handful of evenly spaced frames are saved, so the context-integrity module can be
run over genuine multi-frame video rather than a single-image proxy. The videos are
pulled from the Hugging Face mirror (friedrichor/MSR-VTT), not from YouTube, so the
download is reproducible.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

OUT = Path(__file__).parent / "data" / "video"
FR = OUT / "frames"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
K_FRAMES = 6
MAX_SIDE = 480


def _sample_frames(video_bytes, k=K_FRAMES):
    """Decode bytes with OpenCV via a temp file, return k evenly spaced RGB frames."""
    tmp = OUT / "_tmp.mp4"
    tmp.write_bytes(video_bytes)
    cap = cv2.VideoCapture(str(tmp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    if total > 0:
        idxs = np.linspace(0, total - 1, min(k, total)).astype(int)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if not ok:
                continue
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(fr)
            if max(im.size) > MAX_SIDE:
                im.thumbnail((MAX_SIDE, MAX_SIDE))
            frames.append(im)
    cap.release()
    try:
        tmp.unlink()
    except OSError:
        pass
    return frames


def main():
    from huggingface_hub import hf_hub_download
    FR.mkdir(parents=True, exist_ok=True)
    print("[video] fetching caption split ...", flush=True)
    cap_json = hf_hub_download("friedrichor/MSR-VTT", "msrvtt_test_1k.json", repo_type="dataset")
    items = json.load(open(cap_json, encoding="utf-8"))[:N]
    print(f"[video] {len(items)} clips targeted; downloading video archive (~2.2 GB, cached) ...", flush=True)
    zpath = hf_hub_download("friedrichor/MSR-VTT", "MSRVTT_Videos.zip", repo_type="dataset")

    caps = {}
    cj = OUT / "captions.json"
    if cj.exists():
        caps = json.load(open(cj))
    with zipfile.ZipFile(zpath) as zf:
        names = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".mp4")}
        done = 0
        for it in items:
            vid = it["video_id"]
            fname = it.get("video") or f"{vid}.mp4"
            outdir = FR / vid
            if outdir.exists() and list(outdir.glob("*.jpg")):
                caps[vid] = it["caption"]; done += 1; continue
            member = names.get(fname) or names.get(f"{vid}.mp4")
            if member is None:
                continue
            try:
                frames = _sample_frames(zf.read(member))
            except Exception:
                continue
            if len(frames) < 2:
                continue
            outdir.mkdir(parents=True, exist_ok=True)
            for i, im in enumerate(frames):
                im.save(outdir / f"f{i}.jpg", quality=88)
            caps[vid] = it["caption"]
            done += 1
            if done % 100 == 0:
                print(f"[video] {done} clips processed", flush=True)
                json.dump(caps, open(cj, "w"))
    json.dump(caps, open(cj, "w"))
    print(f"[video] DONE {len(caps)} clips with frames", flush=True)


if __name__ == "__main__":
    main()
