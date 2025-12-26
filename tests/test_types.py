from __future__ import annotations

import numpy as np
import pytest

from promptgroundboxbench.types import Detection


def test_detection_schema_validation() -> None:
    det = Detection(
        boxes_xyxy=np.asarray([[0, 0, 10, 10]], dtype=np.float32),
        labels=["person"],
        scores=np.asarray([0.9], dtype=np.float32),
        image_size_hw=(480, 640),
        source="unit",
    )
    assert det.n == 1


def test_detection_schema_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        Detection(
            boxes_xyxy=np.asarray([[0, 0, 10, 10]], dtype=np.float32),
            labels=["a", "b"],
            scores=np.asarray([0.9], dtype=np.float32),
            image_size_hw=(480, 640),
            source="unit",
        )


def test_to_coco_detections_xywh() -> None:
    det = Detection(
        boxes_xyxy=np.asarray([[10, 20, 30, 60]], dtype=np.float32),
        labels=["person"],
        scores=np.asarray([0.5], dtype=np.float32),
        image_size_hw=(100, 100),
        source="unit",
    )
    mapping = {"person": 1}
    coco = det.to_coco_detections(image_id=42, label_to_category_id=mapping)
    assert coco[0]["image_id"] == 42
    assert coco[0]["category_id"] == 1
    assert coco[0]["bbox"] == [10.0, 20.0, 20.0, 40.0]
