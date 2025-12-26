# tools/eval_vehicles_A.py
from __future__ import annotations

import argparse
import json
import os
import time
import inspect
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from ultralytics import YOLO

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _gd_postprocess(processor, outputs, input_ids, target_sizes, box_thr: float, text_thr: float):
    fn = processor.post_process_grounded_object_detection
    sig = inspect.signature(fn)
    kwargs = {"outputs": outputs, "input_ids": input_ids, "target_sizes": target_sizes}

    # Some versions expect box_threshold, some expect threshold
    if "box_threshold" in sig.parameters:
        kwargs["box_threshold"] = float(box_thr)
    elif "threshold" in sig.parameters:
        kwargs["threshold"] = float(box_thr)

    # text_threshold exists in most versions that support grounded_object_detection
    if "text_threshold" in sig.parameters:
        kwargs["text_threshold"] = float(text_thr)

    return fn(**kwargs)


def _xyxy_to_xywh(box: np.ndarray) -> List[float]:
    x1, y1, x2, y2 = box.tolist()
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="COCO root folder, contains annotations and val2017 etc")
    ap.add_argument("--split", default="val2017", choices=["train2017", "val2017", "test2017"])
    ap.add_argument("--prompt", default="vehicle")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gd_model_id", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--gd_box_threshold", type=float, default=0.35)
    ap.add_argument("--gd_text_threshold", type=float, default=0.25)

    ap.add_argument("--yolo_weights", default="yolo11x.pt")
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--yolo_iou", type=float, default=0.7)
    ap.add_argument("--yolo_imgsz", type=int, default=960)

    ap.add_argument("--max_images", type=int, default=0, help="0 means all")
    ap.add_argument("--out_dir", default="runs/vehicles_A")
    args = ap.parse_args()

    device = args.device
    root = Path(args.root)
    img_dir = root / args.split
    ann_path = root / "annotations" / f"instances_{args.split}.json"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(ann_path))
    img_ids = coco.getImgIds()
    if args.max_images and args.max_images > 0:
        img_ids = img_ids[: args.max_images]

    print("Loading Grounding DINO...")
    processor = AutoProcessor.from_pretrained(args.gd_model_id)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.gd_model_id).to(device)
    gd_model.eval()

    print("Loading YOLO...")
    yolo = YOLO(args.yolo_weights)

    prompt_text = args.prompt.strip()
    # GroundingDINO strongly prefers dot separated classes
    if not prompt_text.endswith("."):
        prompt_text = prompt_text + "."

    preds_gd = []
    preds_yolo = []

    t_gd = []
    t_yolo = []

    for idx, image_id in enumerate(img_ids, start=1):
        info = coco.loadImgs([image_id])[0]
        file_name = info["file_name"]
        img_path = img_dir / file_name
        if not img_path.exists():
            # Roboflow exports sometimes keep the file_name but images are present
            # If not found, try direct name in folder by listing
            raise FileNotFoundError(f"missing image: {img_path}")

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # Grounding DINO
        t0 = time.time()
        inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = gd_model(**inputs)
        results = _gd_postprocess(
            processor=processor,
            outputs=outputs,
            input_ids=inputs.input_ids,
            target_sizes=[(h, w)],
            box_thr=args.gd_box_threshold,
            text_thr=args.gd_text_threshold,
        )
        dt = time.time() - t0
        t_gd.append(dt)

        # results is a list of dict, take first
        r0 = results[0]
        boxes = r0.get("boxes", None)
        scores = r0.get("scores", None)

        if boxes is not None and scores is not None and len(boxes) > 0:
            boxes_np = boxes.detach().cpu().numpy()
            scores_np = scores.detach().cpu().numpy()
            for b, s in zip(boxes_np, scores_np):
                preds_gd.append(
                    {
                        "image_id": int(image_id),
                        "category_id": 1,  # assume single class vehicle has id 1 in this dataset
                        "bbox": _xyxy_to_xywh(np.array(b, dtype=np.float32)),
                        "score": float(s),
                    }
                )

        # YOLO
        t0 = time.time()
        yres = yolo.predict(
            source=np.array(image),
            conf=args.yolo_conf,
            iou=args.yolo_iou,
            imgsz=args.yolo_imgsz,
            verbose=False,
            device=0 if device.startswith("cuda") else "cpu",
        )[0]
        dt = time.time() - t0
        t_yolo.append(dt)

        if yres.boxes is not None and len(yres.boxes) > 0:
            xyxy = yres.boxes.xyxy.detach().cpu().numpy()
            conf = yres.boxes.conf.detach().cpu().numpy()
            cls = yres.boxes.cls.detach().cpu().numpy().astype(int)

            # Map YOLO classes to dataset single class
            # Keep only predictions that look like vehicles in COCO style if you want strict filtering,
            # but for now we keep all and label as vehicle because dataset is vehicle only.
            for b, s, c in zip(xyxy, conf, cls):
                preds_yolo.append(
                    {
                        "image_id": int(image_id),
                        "category_id": 1,
                        "bbox": _xyxy_to_xywh(np.array(b, dtype=np.float32)),
                        "score": float(s),
                    }
                )

        if idx % 25 == 0 or idx == len(img_ids):
            print(f"processed {idx}/{len(img_ids)}")

    # Save predictions
    gd_json = out_dir / f"preds_gd_{args.split}.json"
    yolo_json = out_dir / f"preds_yolo_{args.split}.json"
    gd_json.write_text(json.dumps(preds_gd, indent=2))
    yolo_json.write_text(json.dumps(preds_yolo, indent=2))

    # Evaluate with COCOeval
    def eval_preds(preds_path: Path, name: str) -> Dict:
        coco_dt = coco.loadRes(str(preds_path))
        ev = COCOeval(coco, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        stats = ev.stats.tolist()
        return {
            "name": name,
            "split": args.split,
            "metrics": {
                "AP": float(stats[0]),
                "AP50": float(stats[1]),
                "AP75": float(stats[2]),
                "AP_small": float(stats[3]),
                "AP_medium": float(stats[4]),
                "AP_large": float(stats[5]),
                "AR_1": float(stats[6]),
                "AR_10": float(stats[7]),
                "AR_100": float(stats[8]),
            },
        }

    print("\nCOCOeval Grounding DINO")
    m_gd = eval_preds(gd_json, "grounding_dino")

    print("\nCOCOeval YOLO")
    m_yolo = eval_preds(yolo_json, "yolo")

    report = {
        "dataset_root": str(root),
        "split": args.split,
        "prompt": args.prompt,
        "gd_model_id": args.gd_model_id,
        "gd_box_threshold": args.gd_box_threshold,
        "gd_text_threshold": args.gd_text_threshold,
        "yolo_weights": args.yolo_weights,
        "yolo_conf": args.yolo_conf,
        "yolo_iou": args.yolo_iou,
        "yolo_imgsz": args.yolo_imgsz,
        "timing_sec": {
            "gd_mean": float(np.mean(t_gd)) if len(t_gd) else None,
            "gd_p95": float(np.percentile(t_gd, 95)) if len(t_gd) else None,
            "yolo_mean": float(np.mean(t_yolo)) if len(t_yolo) else None,
            "yolo_p95": float(np.percentile(t_yolo, 95)) if len(t_yolo) else None,
        },
        "results": [m_gd, m_yolo],
    }

    report_path = out_dir / f"report_{args.split}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    # Make tokenizer parallelism quiet on Windows sometimes
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
