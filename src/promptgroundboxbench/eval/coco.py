from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from promptgroundboxbench.engines.base import DetectionEngine
from promptgroundboxbench.utils.prompt import normalize_label


@dataclass(frozen=True)
class CocoEvalResult:
    metrics: dict[str, float]
    stats: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": self.metrics, "stats": self.stats}


def _require_file(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def load_label_list(path: Path) -> list[str]:
    _require_file(path, "Label file")
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            labels.append(normalize_label(s))
    if not labels:
        raise ValueError(f"Label file is empty: {path}")
    return labels


def eval_coco_bbox(
    engine: DetectionEngine,
    coco_images_dir: Path,
    coco_ann_file: Path,
    prompt_labels: list[str] | None,
    limit: int,
    max_dets: int,
    sync_cuda: bool,
) -> tuple[Path, CocoEvalResult]:
    """Run COCO val2017 evaluation and return (pred_json_path, metrics)."""
    del sync_cuda  # placeholder for future batching and GPU timing options

    try:
        from pycocotools.coco import COCO  # type: ignore
        from pycocotools.cocoeval import COCOeval  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pycocotools is required for COCO evaluation: pip install -e .[eval]") from e

    _require_file(coco_ann_file, "COCO annotations")
    if not coco_images_dir.exists():
        raise FileNotFoundError(f"COCO images directory not found: {coco_images_dir}")

    coco_gt = COCO(str(coco_ann_file))

    cats = coco_gt.loadCats(coco_gt.getCatIds())
    name_to_cat_id = {normalize_label(c["name"]): int(c["id"]) for c in cats}

    img_ids = coco_gt.getImgIds()
    if limit and limit > 0:
        img_ids = img_ids[:limit]

    predictions: list[dict[str, Any]] = []
    for image_id in tqdm(img_ids, desc="COCO inference"):
        img_info = coco_gt.loadImgs([image_id])[0]
        file_name = img_info["file_name"]
        img_path = coco_images_dir / file_name
        _require_file(img_path, "COCO image")

        img = Image.open(img_path).convert("RGB")
        det = engine.predict(img, prompt_labels).clip_to_image()
        predictions.extend(det.to_coco_detections(int(image_id), name_to_cat_id))

    preds_path = Path("preds_coco.json")
    with preds_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)

    coco_dt = coco_gt.loadRes(str(preds_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.params.maxDets = [max_dets]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = (
        [float(x) for x in coco_eval.stats.tolist()]
        if hasattr(coco_eval.stats, "tolist")
        else [float(x) for x in coco_eval.stats]
    )

    metrics = {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APS": float(stats[3]),
        "APM": float(stats[4]),
        "APL": float(stats[5]),
    }
    return preds_path, CocoEvalResult(metrics=metrics, stats=stats)
