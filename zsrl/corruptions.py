import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def gaussian_noise(img, s):
    std = [0.04, 0.09, 0.18][s - 1] * 255.0
    a = np.asarray(img).astype(np.float32) + np.random.normal(0, std, np.asarray(img).shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def motion_blur(img, s):
    return img.filter(ImageFilter.GaussianBlur([1.5, 3.0, 5.5][s - 1]))


def brightness(img, s):
    return ImageEnhance.Brightness(img).enhance([0.70, 0.50, 0.32][s - 1])


def jpeg(img, s):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=[30, 15, 7][s - 1])
    buf.seek(0)
    return Image.open(buf).convert("RGB")


CORRUPTIONS = {"gaussian_noise": gaussian_noise, "motion_blur": motion_blur,
               "brightness": brightness, "jpeg": jpeg}


def make(name, severity):
    if name == "clean":
        return None, "clean"
    fn = CORRUPTIONS[name]
    return (lambda im: fn(im, severity)), f"{name}_s{severity}"
