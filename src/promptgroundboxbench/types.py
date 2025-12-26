from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Unified detection schema for both engines.

    Boxes are xyxy absolute pixel coordinates in the input image coordinate system.
    """

    boxes_xyxy: np.ndarray  # shape (N, 4), float32
    labels: list[str]  # length N
    scores: np.ndarray  # shape (N,), float32
    image_size_hw: tuple[int, int]  # (H, W)
    source: str  # engine identifier

    def __post_init__(self) -> None:
        boxes = np.asarray(self.boxes_xyxy)
        scores = np.asarray(self.scores)

        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"boxes_xyxy must have shape (N, 4), got {boxes.shape}")

        if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
            raise ValueError(
                f"scores must have shape (N,), got {scores.shape} for {boxes.shape[0]} boxes"
            )

        if len(self.labels) != boxes.shape[0]:
            raise ValueError(f"labels length must match N, got {len(self.labels)} vs {boxes.shape[0]}")

        h, w = self.image_size_hw
        if h <= 0 or w <= 0:
            raise ValueError(f"image_size_hw must be positive, got {self.image_size_hw}")

    @property
    def n(self) -> int:
        return int(self.boxes_xyxy.shape[0])

    def clip_to_image(self) -> "Detection":
        """Return a copy with boxes clipped into valid image bounds."""
        h, w = self.image_size_hw
        b = self.boxes_xyxy.astype(np.float32, copy=True)
        b[:, 0] = np.clip(b[:, 0], 0, w)
        b[:, 2] = np.clip(b[:, 2], 0, w)
        b[:, 1] = np.clip(b[:, 1], 0, h)
        b[:, 3] = np.clip(b[:, 3], 0, h)
        return Detection(
            boxes_xyxy=b,
            labels=list(self.labels),
            scores=self.scores.astype(np.float32, copy=True),
            image_size_hw=self.image_size_hw,
            source=self.source,
        )

    def to_coco_detections(
        self, image_id: int, label_to_category_id: dict[str, int]
    ) -> list[dict[str, Any]]:
        """Convert to COCO detection JSON dicts with absolute xywh boxes."""
        b = self.boxes_xyxy.astype(float)
        out: list[dict[str, Any]] = []
        for box, label, score in zip(b, self.labels, self.scores):
            if label not in label_to_category_id:
                continue
            x1, y1, x2, y2 = map(float, box)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            out.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label_to_category_id[label]),
                    "bbox": [x1, y1, w, h],
                    "score": float(score),
                }
            )
        return out
