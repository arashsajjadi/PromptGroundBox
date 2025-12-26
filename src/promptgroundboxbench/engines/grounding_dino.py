from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from promptgroundboxbench.engines.base import EngineMeta
from promptgroundboxbench.types import Detection
from promptgroundboxbench.utils.prompt import normalize_label


@dataclass
class GroundingDINOEngine:
    """Grounding DINO zero shot detector via Hugging Face Transformers."""

    model_id: str
    device: str
    box_threshold: float
    text_threshold: float
    label_normalize: bool = True

    def __post_init__(self) -> None:
        self.meta = EngineMeta(name="grounding_dino", device=self.device)
        self._processor = None
        self._model = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return

        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependencies for Grounding DINO. "
                "Install extras: pip install -e .[inference] and a compatible torch build."
            ) from e

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        self._torch = torch

    def predict(self, image: Image.Image, labels: list[str] | None = None) -> Detection:
        self._lazy_init()
        assert self._processor is not None
        assert self._model is not None

        if not labels:
            raise ValueError("Grounding DINO requires a non empty list of labels")

        img = image.convert("RGB")
        h, w = img.size[1], img.size[0]

        text_labels = [labels]  # batch size 1
        inputs = self._processor(images=img, text=text_labels, return_tensors="pt").to(
            self._model.device
        )

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        # Official post processing for grounded object detection
        results: list[dict[str, Any]] = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=float(self.box_threshold),
            text_threshold=float(self.text_threshold),
            target_sizes=[(h, w)],
        )
        r0 = results[0]
        boxes = r0["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = r0["scores"].detach().cpu().numpy().astype(np.float32)

        raw_labels = r0["labels"]
        if isinstance(raw_labels, (list, tuple)):
            lab_list = [str(x) for x in raw_labels]
        else:
            lab_list = [str(raw_labels)] * int(boxes.shape[0])

        if self.label_normalize:
            lab_list = [normalize_label(s) for s in lab_list]

        return Detection(
            boxes_xyxy=boxes,
            labels=lab_list,
            scores=scores,
            image_size_hw=(h, w),
            source=self.meta.name,
        )
