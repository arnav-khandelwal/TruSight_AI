# -*- coding: utf-8 -*-
"""
Download SECOND datasets for cross-dataset generalization tests: the detectors are
scored on a different source than the primary evaluation, with no re-tuning. Same
honest label-by-name mapping as prepare_data.py; images land in
eval/data/<task>_x/{real,fake}/.

  deepfake_x : JamieWithofs/Deepfake-and-real-images (256x256) - same face-forgery
               task, a different generator/source than the primary set.
  synthetic_x: dragonintelligence/CIFAKE-image-dataset (32x32) - a different
               generator (Stable Diffusion over CIFAR-10); also a small-image stress
               test, reported as such.
"""
import sys
from pathlib import Path
import prepare_data as pd

TASKS = {
    "deepfake_x": [("JamieWithofs/Deepfake-and-real-images", "train")],
    "synthetic_x": [("dragonintelligence/CIFAKE-image-dataset", "test")],
}

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    for task, cands in TASKS.items():
        if which in ("all", task):
            pd.harvest(task, cands, n=n)
    print("XDATA DONE", flush=True)
