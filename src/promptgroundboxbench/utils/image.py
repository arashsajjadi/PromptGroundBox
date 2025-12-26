from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    return img


def pil_to_numpy_rgb(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"))
    return arr
