import time

from PIL import Image

from zsrl import get_device
from zsrl.dataset import MVTecDefects
from zsrl.encoder import FrozenEncoder

enc = FrozenEncoder(get_device())
path, box, label, key = MVTecDefects(split="train")[0]
img = Image.open(path).convert("RGB")

root = enc.encode_node(img, key, (0, 0, 0))
child = enc.encode_node(img, key, (1, 0, 0))
print("shape", root.shape)
print("root vs child distance", float(((root - child) ** 2).sum() ** 0.5))

t0 = time.time()
for _ in range(300):
    enc.encode_node(img, key, (0, 0, 0))
print("300 repeats in", round(time.time() - t0, 3), "s", enc.stats())
