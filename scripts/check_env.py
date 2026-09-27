import os

import numpy as np
from PIL import Image

from zsrl import CANVAS, get_device
from zsrl.baselines import (exhaustive, normal_model_descent, oracle,
                             random_descent, saliency_descent)
from zsrl.dataset import MVTecDefects
from zsrl.encoder import FrozenEncoder

cs = MVTecDefects(split="test")
rng = np.random.default_rng(0)

rows = {"oracle": [], "random": [], "saliency": [], "exhaustive": []}

ref_path = os.path.join("runs", "baselines", "normal_reference.npz")
reference, encoder = None, None
if os.path.exists(ref_path):
    reference = np.load(ref_path)
    encoder = FrozenEncoder(get_device())
    rows["normal_model"] = []
else:
    print(f"note: {ref_path} not found, skipping normal_model_descent "
          f"(run `python -m scripts.build_normal_reference` first)")

for i in range(min(200, len(cs))):
    path, box, label, key = cs[i]
    # MVTec images are 700-1024px depending on category; box_from_mask
    # already scales boxes to CANVAS, so the image must match (same fix
    # IMPLEMENTATION.md applies to ZoomSearchEnv.reset).
    img = Image.open(path).convert("RGB").resize((CANVAS, CANVAS))
    rows["oracle"].append(oracle(box))
    rows["random"].append(random_descent(box, rng))
    rows["saliency"].append(saliency_descent(img, box))
    rows["exhaustive"].append(exhaustive(box))
    if reference is not None:
        cat = label.split("/")[0]
        cat_ref = reference[cat] if cat in reference else None
        rows["normal_model"].append(
            normal_model_descent(img, box, cat_ref, encoder.encode_pil))

for name, rs in rows.items():
    rec = float(np.mean([r["success"] for r in rs]))
    cost = float(np.mean([r["cost"] for r in rs]))
    print(f"{name:12s} recall {rec:.3f}   mean cost {cost:.1f}")
