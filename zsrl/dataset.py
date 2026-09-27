import glob
import os

import numpy as np
from PIL import Image

from zsrl import CANVAS

MVTEC_ROOT = os.environ.get("MVTEC_ROOT", "data/mvtec")

CATEGORIES = ("bottle", "cable", "capsule", "carpet", "grid",
              "hazelnut", "leather", "metal_nut", "pill", "screw",
              "tile", "toothbrush", "transistor", "wood", "zipper")


def box_from_mask(mask_path, out_size=CANVAS):
    """Bounding box of the defect, in coordinates of the resized image."""
    m = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    h, w = m.shape
    sx, sy = out_size / w, out_size / h
    return (float(xs.min()) * sx, float(ys.min()) * sy,
            float(xs.max() + 1) * sx, float(ys.max() + 1) * sy)


class MVTecDefects:
    """Defective MVTec images with a box derived from the supplied mask.

    Same interface the environment expects:
        len(ds), ds[i] -> (image_path, box, label, key)
    Boxes are in coordinates of the image AFTER resizing to CANVAS.
    """

    def __init__(self, root=MVTEC_ROOT, categories=CATEGORIES,
                 max_area_frac=0.05, min_side=12.0,
                 split="train", split_frac=0.8, seed=0):
        records = []

        for cat in categories:
            gt_dir = os.path.join(root, cat, "ground_truth")
            if not os.path.isdir(gt_dir):
                continue
            for defect in sorted(os.listdir(gt_dir)):
                for mask_path in sorted(glob.glob(
                        os.path.join(gt_dir, defect, "*_mask.png"))):
                    stem = os.path.basename(mask_path).replace("_mask.png", "")
                    img_path = os.path.join(root, cat, "test", defect, stem + ".png")
                    if not os.path.exists(img_path):
                        continue

                    box = box_from_mask(mask_path)
                    if box is None:
                        continue
                    bw, bh = box[2] - box[0], box[3] - box[1]
                    if bw < min_side or bh < min_side:
                        continue
                    if (bw * bh) > max_area_frac * CANVAS * CANVAS:
                        continue

                    records.append({
                        "image_path": img_path,
                        "box": box,
                        "label": cat + "/" + defect,
                        "key": cat + "|" + defect + "|" + stem,
                        "category": cat,
                    })

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(records))
        cut = int(split_frac * len(records))
        keep = order[:cut] if split == "train" else order[cut:]
        self.records = [records[int(i)] for i in keep]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return r["image_path"], r["box"], r["label"], r["key"]

    def categories(self):
        from collections import Counter
        return Counter(r["category"] for r in self.records)
