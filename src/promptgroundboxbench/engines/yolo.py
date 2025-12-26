from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from promptgroundboxbench.engines.base import EngineMeta
from promptgroundboxbench.types import Detection
from promptgroundboxbench.utils.prompt import normalize_label


@dataclass
class YOLOEngine:
    """Closed set YOLO11 detector via Ultralytics."""

    weights: str
    device: str
    conf: float
    label_normalize: bool = True

    def __post_init__(self) -> None:
        self.meta = EngineMeta(name="yolo", device=self.device)
        self._model = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing Ultralytics dependency. Install extras: pip install -e .[inference]"
            ) from e

        self._model = YOLO(self.weights)

    def predict(self, image: Image.Image, labels: list[str] | None = None) -> Detection:
        del labels  # closed set model ignores prompts
        self._lazy_init()
        assert self._model is not None

        img = image.convert("RGB")
        h, w = img.size[1], img.size[0]

        results = self._model.predict(
            source=img, conf=float(self.conf), device=self.device, verbose=False
        )
        r0 = results[0]
        boxes = r0.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = r0.boxes.conf.detach().cpu().numpy().astype(np.float32)
        cls_ids = r0.boxes.cls.detach().cpu().numpy().astype(int)

        names = r0.names
        lab_list = [str(names[int(i)]) for i in cls_ids]
        if self.label_normalize:
            lab_list = [normalize_label(s) for s in lab_list]

        return Detection(
            boxes_xyxy=boxes,
            labels=lab_list,
            scores=scores,
            image_size_hw=(h, w),
            source=self.meta.name,
        )
