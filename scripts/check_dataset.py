import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from zsrl import CANVAS
from zsrl.dataset import MVTecDefects

train = MVTecDefects(split="train")
test = MVTecDefects(split="test")
print("train", len(train), "test", len(test))
print(train.categories())

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for ax, i in zip(axes.ravel(), range(8)):
    path, box, label, key = train[i]
    ax.imshow(Image.open(path).convert("RGB").resize((CANVAS, CANVAS)))
    x1, y1, x2, y2 = box
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, lw=2, edgecolor="lime"))
    for k in range(1, 8):
        ax.axhline(k * CANVAS / 8, lw=0.3, color="white")
        ax.axvline(k * CANVAS / 8, lw=0.3, color="white")
    ax.set_title(label, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("results/dataset_check.png", dpi=110)
