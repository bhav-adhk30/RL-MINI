"""Run on Kaggle, where MVTec AD is mounted. Selects 15 defective TEST
images spread across as many categories as possible (round-robin, so >=4
categories is trivial unless the test split is unusually concentrated),
biased toward clearly-visible defects (largest mask-box area first, within
each category), resizes each to CANVAS x CANVAS and saves as JPEG q90 into
data/demo/, and writes data/demo/index.json.

Reuses MVTecDefects' own filtering (max_area_frac, min_side) and its
seeded train/test split -- the demo set is a small, static sample of
exactly what the real test split would give the app anyway, not a
different selection process.

Usage: python -m scripts.make_demo_bundle
"""
import json
import os

from PIL import Image

from zsrl import CANVAS
from zsrl.dataset import MVTecDefects

OUT_DIR = "data/demo"
N_IMAGES = 15
JPEG_QUALITY = 90


def select(records, n=N_IMAGES):
    by_cat = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    for cat in by_cat:
        by_cat[cat].sort(
            key=lambda r: (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]),
            reverse=True)  # largest (most visible) defect first, per category

    cats = sorted(by_cat, key=lambda c: -len(by_cat[c]))  # richest categories first
    chosen, i = [], 0
    while len(chosen) < n and any(by_cat[c] for c in cats):
        cat = cats[i % len(cats)]
        if by_cat[cat]:
            chosen.append(by_cat[cat].pop(0))
        i += 1
    return chosen


def main():
    test = MVTecDefects(split="test")
    print(f"test split: {len(test)} candidates across "
          f"{len(set(r['category'] for r in test.records))} categories")

    chosen = select(test.records, N_IMAGES)
    cats_used = sorted(set(r["category"] for r in chosen))
    print(f"selected {len(chosen)} images across {len(cats_used)} categories: {cats_used}")
    assert len(cats_used) >= 4, "fewer than 4 categories in the selection"

    os.makedirs(OUT_DIR, exist_ok=True)
    index = []
    for r in chosen:
        stem = r["key"].replace("|", "_")
        fname = f"{stem}.jpg"
        img = Image.open(r["image_path"]).convert("RGB").resize((CANVAS, CANVAS))
        img.save(os.path.join(OUT_DIR, fname), "JPEG", quality=JPEG_QUALITY)
        cat, defect = r["label"].split("/")
        index.append({
            "filename": fname,
            "box": list(r["box"]),  # already in CANVAS-space, from box_from_mask
            "category": cat,
            "defect_type": defect,
        })

    index_path = os.path.join(OUT_DIR, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    total = sum(os.path.getsize(os.path.join(OUT_DIR, e["filename"])) for e in index)
    total += os.path.getsize(index_path)
    print(f"\nwrote {len(index)} images + index.json to {OUT_DIR}/")
    print(f"total size: {total / 1024:.1f} KiB ({total / 1024 / 1024:.2f} MiB)")
    if total > 5 * 1024 * 1024:
        print("WARNING: over the 5MB target -- lower JPEG_QUALITY and re-run")


if __name__ == "__main__":
    main()
