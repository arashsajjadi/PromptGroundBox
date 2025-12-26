from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pycocotools.coco import COCO


def _area_xywh(b: List[float]) -> float:
    return float(max(0.0, b[2]) * max(0.0, b[3]))


def _to_xyxy(b: List[float]) -> Tuple[float, float, float, float]:
    # COCO bbox is [x, y, w, h]
    x, y, w, h = map(float, b)
    return (x, y, x + w, y + h)


def _iou_xywh(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = _to_xyxy(a)
    bx1, by1, bx2, by2 = _to_xyxy(b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return float(inter / union) if union > 0.0 else 0.0


def _load_preds(path: Path) -> List[Dict]:
    x = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(x, list):
        raise ValueError(f"pred file must be list: {path}")
    # normalize
    out = []
    for r in x:
        if "image_id" not in r or "bbox" not in r:
            continue
        out.append(
            {
                "image_id": int(r["image_id"]),
                "bbox": [float(v) for v in r["bbox"]],
                "score": float(r.get("score", 1.0)),
            }
        )
    return out


def _group_by_image(preds: List[Dict]) -> Dict[int, List[Dict]]:
    g: Dict[int, List[Dict]] = {}
    for r in preds:
        g.setdefault(r["image_id"], []).append(r)
    # sort by score desc
    for k in g:
        g[k].sort(key=lambda z: z["score"], reverse=True)
    return g


def _gt_by_image(coco: COCO, img_ids: List[int]) -> Dict[int, List[Dict]]:
    g: Dict[int, List[dict]] = {}
    out: Dict[int, List[Dict]] = {}
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        # keep non-crowd only
        keep = []
        for a in anns:
            if int(a.get("iscrowd", 0)) == 1:
                continue
            keep.append(
                {
                    "bbox": [float(v) for v in a["bbox"]],
                    "area": float(a.get("area", _area_xywh(a["bbox"]))),
                }
            )
        out[int(img_id)] = keep
    return out


def _match_at_threshold(
    gt: Dict[int, List[Dict]],
    preds_by_img: Dict[int, List[Dict]],
    img_ids: List[int],
    score_thr: float,
    iou_thr: float,
) -> Dict:
    # greedy matching per image: preds sorted by score desc, match each pred to best unmatched gt if IoU>=thr
    tp = 0
    fp = 0
    fn = 0

    per_image = []
    matched_ious = []
    matched_scores = []

    for img_id in img_ids:
        gts = gt.get(img_id, [])
        preds = [p for p in preds_by_img.get(img_id, []) if p["score"] >= score_thr]

        used = [False] * len(gts)

        img_tp = 0
        img_fp = 0

        for p in preds:
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                iou = _iou_xywh(p["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= iou_thr:
                used[best_j] = True
                img_tp += 1
                matched_ious.append(best_iou)
                matched_scores.append(p["score"])
            else:
                img_fp += 1

        img_fn = int(sum(1 for u in used if not u))
        tp += img_tp
        fp += img_fp
        fn += img_fn

        per_image.append(
            {
                "image_id": img_id,
                "gt_count": len(gts),
                "pred_count": len(preds),
                "tp": img_tp,
                "fp": img_fp,
                "fn": img_fn,
            }
        )

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "score_thr": float(score_thr),
        "iou_thr": float(iou_thr),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "per_image": per_image,
        "matched_iou_summary": {
            "count": int(len(matched_ious)),
            "mean": float(np.mean(matched_ious)) if matched_ious else None,
            "p50": float(np.percentile(matched_ious, 50)) if matched_ious else None,
            "p90": float(np.percentile(matched_ious, 90)) if matched_ious else None,
        },
        "matched_score_summary": {
            "count": int(len(matched_scores)),
            "mean": float(np.mean(matched_scores)) if matched_scores else None,
            "p50": float(np.percentile(matched_scores, 50)) if matched_scores else None,
            "p90": float(np.percentile(matched_scores, 90)) if matched_scores else None,
        },
    }


def _summarize_counts(per_image: List[Dict]) -> Dict:
    gt = np.array([x["gt_count"] for x in per_image], dtype=np.float32)
    pr = np.array([x["pred_count"] for x in per_image], dtype=np.float32)
    tp = np.array([x["tp"] for x in per_image], dtype=np.float32)
    fp = np.array([x["fp"] for x in per_image], dtype=np.float32)
    fn = np.array([x["fn"] for x in per_image], dtype=np.float32)

    def s(a):
        return {
            "mean": float(a.mean()) if a.size else None,
            "p50": float(np.percentile(a, 50)) if a.size else None,
            "p90": float(np.percentile(a, 90)) if a.size else None,
            "max": float(a.max()) if a.size else None,
        }

    return {
        "per_image_gt": s(gt),
        "per_image_pred": s(pr),
        "per_image_tp": s(tp),
        "per_image_fp": s(fp),
        "per_image_fn": s(fn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="COCO root (contains annotations + split folder)")
    ap.add_argument("--split", required=True, choices=["train2017", "val2017", "test2017"])
    ap.add_argument("--run_dir", default="runs/vehicles_A")
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--score_grid", default="0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    ap.add_argument("--max_images", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    root = Path(args.root)
    ann_path = root / "annotations" / f"instances_{args.split}.json"
    run_dir = Path(args.run_dir)

    gd_path = run_dir / f"preds_gd_{args.split}.json"
    yolo_path = run_dir / f"preds_yolo_{args.split}.json"

    if not gd_path.exists():
        raise FileNotFoundError(f"missing: {gd_path}")
    if not yolo_path.exists():
        raise FileNotFoundError(f"missing: {yolo_path}")

    coco = COCO(str(ann_path))
    img_ids = coco.getImgIds()
    img_ids = [int(x) for x in img_ids]
    if args.max_images and args.max_images > 0:
        img_ids = img_ids[: args.max_images]

    gt = _gt_by_image(coco, img_ids)

    preds_gd = _group_by_image(_load_preds(gd_path))
    preds_yolo = _group_by_image(_load_preds(yolo_path))

    score_thrs = [float(x.strip()) for x in args.score_grid.split(",") if x.strip() != ""]
    iou_thr = float(args.iou_thr)

    out = {
        "dataset_root": str(root),
        "split": args.split,
        "iou_thr": iou_thr,
        "score_grid": score_thrs,
        "n_images": int(len(img_ids)),
        "n_gt_boxes": int(sum(len(gt[i]) for i in img_ids)),
        "models": {},
    }

    for name, preds in [("grounding_dino", preds_gd), ("yolo", preds_yolo)]:
        grid = []
        best = None
        for sthr in score_thrs:
            r = _match_at_threshold(gt, preds, img_ids, score_thr=sthr, iou_thr=iou_thr)
            r["count_summary"] = _summarize_counts(r["per_image"])
            # do not dump giant per_image for every threshold into file
            per_image = r.pop("per_image")
            r["n_images"] = len(per_image)
            grid.append(r)

            if best is None or r["f1"] > best["f1"]:
                best = dict(r)
                best["best_score_thr"] = float(sthr)

        out["models"][name] = {
            "best_by_f1": best,
            "threshold_grid": grid,
        }

    out_path = run_dir / f"analysis_{args.split}_iou{int(iou_thr*100):02d}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote analysis: {out_path}")


if __name__ == "__main__":
    main()
