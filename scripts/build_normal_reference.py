"""Build the per-category 'normal' reference for normal_model_descent.

For each MVTec category, encodes the four depth-1 quadrants of every
train/good/ (defect-free) image with the frozen encoder and averages them
per quadrant position. Saved once and reused; rebuilding needs one encoder
forward pass per quadrant per good image (~3600 images x 4 in the current
dataset), which is slow enough on CPU that it should not run on every
baseline check.

Usage: python -m scripts.build_normal_reference
Output: runs/baselines/normal_reference.npz  (gitignored, regenerable)
"""
import glob
import os

import numpy as np
from PIL import Image

from zsrl import CANVAS, get_device
from zsrl.dataset import CATEGORIES, MVTEC_ROOT
from zsrl.encoder import FrozenEncoder
from zsrl.env import child, node_box

OUT_PATH = os.path.join("runs", "baselines", "normal_reference.npz")


def good_image_paths(category, root=MVTEC_ROOT):
    d = os.path.join(root, category, "train", "good")
    if not os.path.isdir(d):
        return []
    return sorted(glob.glob(os.path.join(d, "*.png")))


def depth1_quadrant_crops(img):
    return [img.crop(node_box(child((0, 0, 0), q))) for q in range(4)]


def build(root=MVTEC_ROOT, categories=CATEGORIES):
    encoder = FrozenEncoder(get_device())
    reference = {}
    for cat in categories:
        paths = good_image_paths(cat, root)
        if not paths:
            print(f"{cat:12s} skipped, no train/good images")
            continue
        sums = np.zeros((4, encoder.dim), dtype=np.float64)
        for p in paths:
            img = Image.open(p).convert("RGB").resize((CANVAS, CANVAS))
            for q, crop in enumerate(depth1_quadrant_crops(img)):
                sums[q] += encoder.encode_pil(crop)
        reference[cat] = (sums / len(paths)).astype(np.float32)
        print(f"{cat:12s} n_good={len(paths)}")
    return reference


if __name__ == "__main__":
    ref = build()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(OUT_PATH, **ref)
    print("saved", OUT_PATH)
