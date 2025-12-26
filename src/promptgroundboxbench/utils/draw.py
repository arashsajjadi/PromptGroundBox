from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_boxes(
    image: Image.Image,
    boxes_xyxy: np.ndarray,
    labels: Iterable[str],
    scores: Iterable[float],
) -> Image.Image:
    """Draw boxes and labels on an image and return a new image."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = _default_font()

    for box, label, score in zip(boxes_xyxy, labels, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        text = f"{label} {score:.3f}"
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 6, y1], fill=(255, 0, 0))
        draw.text((x1 + 3, y1 - th - 2), text, fill=(255, 255, 255), font=font)

    return img


def _default_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 14)
    except Exception:
        return ImageFont.load_default()
