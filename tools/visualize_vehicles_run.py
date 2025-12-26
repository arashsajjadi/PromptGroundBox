# tools/visualize_vehicles_run.py
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# matplotlib is used for plots
import matplotlib.pyplot as plt

# optional but nice, if missing we still run
try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

from pycocotools.coco import COCO


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def iou_xywh(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy()
    bx1, by1, bx2, by2 = b.xyxy()

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, a.w) * max(0.0, a.h)
    area_b = max(0.0, b.w) * max(0.0, b.h)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def parse_preds(preds: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_img: Dict[int, List[Dict[str, Any]]] = {}
    for r in preds:
        iid = int(r["image_id"])
        by_img.setdefault(iid, []).append(r)
    return by_img


def parse_gts(coco: COCO, img_ids: List[int]) -> Dict[int, List[Dict[str, Any]]]:
    gts: Dict[int, List[Dict[str, Any]]] = {}
    for iid in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[iid])
        anns = coco.loadAnns(ann_ids)
        gts[iid] = anns
    return gts


def match_one_image(
    gt_anns: List[Dict[str, Any]],
    preds: List[Dict[str, Any]],
    score_thr: float,
    iou_thr: float,
) -> Dict[str, Any]:
    """
    Greedy matching by descending score.
    Returns:
      tp_pairs: list of dict with keys pred, gt, iou
      fp_preds: list of pred dict
      fn_gts: list of gt dict
    """
    gts = []
    for a in gt_anns:
        bb = a.get("bbox", None)
        if bb is None or len(bb) != 4:
            continue
        gts.append({"ann": a, "box": Box(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]) )})

    cand = []
    for p in preds:
        s = safe_float(p.get("score"))
        if s is None or s < score_thr:
            continue
        bb = p.get("bbox", None)
        if bb is None or len(bb) != 4:
            continue
        cand.append({"pred": p, "box": Box(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])), "score": float(s)})

    cand.sort(key=lambda z: z["score"], reverse=True)

    used_gt = set()
    tp_pairs = []
    fp_preds = []

    for c in cand:
        best_iou = -1.0
        best_j = None
        for j, g in enumerate(gts):
            if j in used_gt:
                continue
            v = iou_xywh(g["box"], c["box"])
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j is not None and best_iou >= iou_thr:
            used_gt.add(best_j)
            tp_pairs.append({"pred": c["pred"], "gt": gts[best_j]["ann"], "iou": float(best_iou), "score": float(c["score"])})
        else:
            fp_preds.append(c["pred"])

    fn_gts = []
    for j, g in enumerate(gts):
        if j not in used_gt:
            fn_gts.append(g["ann"])

    return {"tp_pairs": tp_pairs, "fp_preds": fp_preds, "fn_gts": fn_gts}


def plot_bar_metrics(
    out_png: Path,
    title: str,
    metrics_by_model: Dict[str, Dict[str, float]],
    keys: List[str],
) -> None:
    ensure_dir(out_png.parent)
    labels = list(metrics_by_model.keys())

    x = np.arange(len(keys), dtype=np.float32)
    width = 0.8 / max(1, len(labels))

    plt.figure()
    for i, m in enumerate(labels):
        vals = [float(metrics_by_model[m].get(k, float("nan"))) for k in keys]
        plt.bar(x + i * width, vals, width=width, label=m)

    plt.xticks(x + (len(labels) - 1) * width / 2.0, keys, rotation=0)
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_pr_curve_from_grid(out_png: Path, title: str, grid: List[Dict[str, Any]]) -> None:
    ensure_dir(out_png.parent)
    thrs = [float(g["score_thr"]) for g in grid]
    prec = [float(g["precision"]) for g in grid]
    rec = [float(g["recall"]) for g in grid]
    f1 = [float(g["f1"]) for g in grid]

    plt.figure()
    plt.plot(rec, prec, marker="o")
    for t, r, p in zip(thrs, rec, prec):
        plt.annotate(f"{t:.1f}", (r, p), fontsize=8)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    out_png_f1 = out_png.with_name(out_png.stem + "_f1.png")
    plt.figure()
    plt.plot(thrs, f1, marker="o")
    plt.xlabel("Score threshold")
    plt.ylabel("F1")
    plt.title(title + " (F1 vs threshold)")
    plt.tight_layout()
    plt.savefig(out_png_f1, dpi=160)
    plt.close()


def plot_hist(out_png: Path, title: str, values: List[float], xlabel: str, bins: int = 30) -> None:
    ensure_dir(out_png.parent)
    v = [float(x) for x in values if safe_float(x) is not None]
    if len(v) == 0:
        return
    plt.figure()
    plt.hist(v, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_latency(out_png: Path, title: str, timing: Dict[str, Any]) -> None:
    ensure_dir(out_png.parent)
    items = []
    for k in ["gd_mean", "gd_p95", "yolo_mean", "yolo_p95"]:
        v = timing.get(k, None)
        if v is None:
            continue
        items.append((k, float(v)))

    if not items:
        return

    names = [a for a, _ in items]
    vals = [b for _, b in items]
    x = np.arange(len(names), dtype=np.float32)

    plt.figure()
    plt.bar(x, vals)
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylabel("Seconds")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def try_load_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Windows friendly fallback
    for name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_boxes(
    image: Image.Image,
    gt_boxes: List[Dict[str, Any]],
    pred_boxes: List[Dict[str, Any]],
    title_lines: List[str],
    max_preds: int = 25,
) -> Image.Image:
    img = image.copy().convert("RGB")
    d = ImageDraw.Draw(img)
    font = try_load_font(14)

    # ground truth in green
    for a in gt_boxes:
        bb = a.get("bbox", None)
        if bb is None or len(bb) != 4:
            continue
        x, y, w, h = [float(v) for v in bb]
        x1, y1, x2, y2 = x, y, x + w, y + h
        d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

    # predictions in red
    shown = 0
    for p in pred_boxes:
        if shown >= max_preds:
            break
        bb = p.get("bbox", None)
        if bb is None or len(bb) != 4:
            continue
        s = safe_float(p.get("score"))
        x, y, w, h = [float(v) for v in bb]
        x1, y1, x2, y2 = x, y, x + w, y + h
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        if s is not None:
            d.text((x1 + 2, y1 + 2), f"{s:.2f}", fill=(255, 0, 0), font=font)
        shown += 1

    # title banner
    pad = 6
    banner_h = 18 * max(1, len(title_lines)) + pad * 2
    banner = Image.new("RGB", (img.width, banner_h), (20, 20, 20))
    bd = ImageDraw.Draw(banner)
    y = pad
    for line in title_lines:
        bd.text((pad, y), line, fill=(240, 240, 240), font=font)
        y += 18
    out = Image.new("RGB", (img.width, img.height + banner_h), (0, 0, 0))
    out.paste(banner, (0, 0))
    out.paste(img, (0, banner_h))
    return out


def make_html_index(out_html: Path, title: str, sections: List[Tuple[str, List[Tuple[str, str]]]]) -> None:
    """
    sections: list of (section_title, items)
    items: list of (caption, relative_path)
    """
    ensure_dir(out_html.parent)
    lines = []
    lines.append("<!doctype html>")
    lines.append("<html><head><meta charset='utf-8'/>")
    lines.append(f"<title>{title}</title>")
    lines.append("<style>")
    lines.append("body{font-family:Arial, sans-serif; margin:20px;}")
    lines.append("h1{margin-bottom:6px;}")
    lines.append("h2{margin-top:26px;}")
    lines.append(".grid{display:grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap:14px;}")
    lines.append(".card{border:1px solid #ddd; border-radius:10px; padding:10px;}")
    lines.append("img{max-width:100%; height:auto; border-radius:8px;}")
    lines.append(".cap{font-size:13px; color:#333; margin-top:6px;}")
    lines.append("</style></head><body>")
    lines.append(f"<h1>{title}</h1>")
    for st, items in sections:
        lines.append(f"<h2>{st}</h2>")
        lines.append("<div class='grid'>")
        for cap, rel in items:
            lines.append("<div class='card'>")
            lines.append(f"<a href='{rel}' target='_blank'><img src='{rel}'/></a>")
            lines.append(f"<div class='cap'>{cap}</div>")
            lines.append("</div>")
        lines.append("</div>")
    lines.append("</body></html>")
    out_html.write_text("\n".join(lines), encoding="utf-8")


def table_to_png(
    out_png: Path,
    title: str,
    columns: List[str],
    rows: List[List[Any]],
) -> None:
    """
    Creates a readable table image using matplotlib.
    """
    ensure_dir(out_png.parent)
    plt.figure(figsize=(max(8, len(columns) * 1.2), max(2.5, len(rows) * 0.5 + 1.5)))
    plt.axis("off")
    plt.title(title)

    cell_text = []
    for r in rows:
        rr = []
        for v in r:
            if isinstance(v, float):
                rr.append(f"{v:.4f}")
            else:
                rr.append(str(v))
        cell_text.append(rr)

    tab = plt.table(cellText=cell_text, colLabels=columns, loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1.0, 1.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def build_metrics_tables(report: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    """
    report format is your eval_vehicles_A report json.
    Produces a table: rows per model, columns common metrics.
    """
    cols = ["model", "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_100"]
    rows = []
    results = report.get("results", [])
    for r in results:
        name = r.get("name", "model")
        m = r.get("metrics", {})
        rows.append([
            name,
            float(m.get("AP", float("nan"))),
            float(m.get("AP50", float("nan"))),
            float(m.get("AP75", float("nan"))),
            float(m.get("AP_small", float("nan"))),
            float(m.get("AP_medium", float("nan"))),
            float(m.get("AP_large", float("nan"))),
            float(m.get("AR_100", float("nan"))),
        ])
    return cols, rows


def build_f1_tables(analysis: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    """
    analysis format is analyze_vehicles_report output.
    Table with best_by_f1 per model.
    """
    cols = ["model", "best_score_thr", "precision", "recall", "f1", "tp", "fp", "fn"]
    rows = []
    models = analysis.get("models", {})
    for name, obj in models.items():
        b = obj.get("best_by_f1", {})
        rows.append([
            name,
            float(b.get("best_score_thr", b.get("score_thr", float("nan")))),
            float(b.get("precision", float("nan"))),
            float(b.get("recall", float("nan"))),
            float(b.get("f1", float("nan"))),
            int(b.get("tp", 0)),
            int(b.get("fp", 0)),
            int(b.get("fn", 0)),
        ])
    return cols, rows


def pick_hard_images(
    coco: COCO,
    img_ids: List[int],
    gts_by_img: Dict[int, List[Dict[str, Any]]],
    preds_by_img: Dict[int, List[Dict[str, Any]]],
    score_thr: float,
    iou_thr: float,
    k: int,
    seed: int,
    mode: str,
) -> List[int]:
    """
    mode:
      fn heavy : images with high FN count
      fp heavy : images with high FP count
      random   : random images
    """
    rng = random.Random(seed)
    scored = []

    if mode == "random":
        ids = img_ids[:]
        rng.shuffle(ids)
        return ids[:k]

    for iid in img_ids:
        m = match_one_image(gts_by_img.get(iid, []), preds_by_img.get(iid, []), score_thr, iou_thr)
        fnc = len(m["fn_gts"])
        fpc = len(m["fp_preds"])
        tpc = len(m["tp_pairs"])
        if mode == "fn_heavy":
            key = (fnc, -tpc)
        elif mode == "fp_heavy":
            key = (fpc, -tpc)
        else:
            key = (fnc + fpc, -tpc)
        scored.append((key, iid, fnc, fpc, tpc))

    scored.sort(key=lambda z: z[0], reverse=True)
    return [x[1] for x in scored[:k]]


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", required=True, help="COCO root folder: contains annotations and split folders")
    ap.add_argument("--split", default="val2017", choices=["train2017", "val2017", "test2017"])
    ap.add_argument("--run_dir", required=True, help="run directory that contains preds and reports, eg runs/vehicles_A")
    ap.add_argument("--out_subdir", default="visuals", help="subfolder name under run_dir for outputs")

    ap.add_argument("--analysis_json", default="", help="analysis json from analyze script, optional")
    ap.add_argument("--report_json", default="", help="report json from eval script, optional")

    ap.add_argument("--iou_thr", type=float, default=0.50)
    ap.add_argument("--score_thr_gd", type=float, default=None)   # type: ignore
    ap.add_argument("--score_thr_yolo", type=float, default=None) # type: ignore

    ap.add_argument("--qual_k", type=int, default=16)
    ap.add_argument("--qual_mode", default="fn_heavy", choices=["fn_heavy", "fp_heavy", "random", "mixed"])
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    root = Path(args.root)
    run_dir = Path(args.run_dir)

    ann_path = root / "annotations" / f"instances_{args.split}.json"
    img_dir = root / args.split

    if not ann_path.exists():
        raise FileNotFoundError(f"missing annotations: {ann_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"missing image dir: {img_dir}")

    # default file names produced by eval_vehicles_A.py
    preds_gd_path = run_dir / f"preds_gd_{args.split}.json"
    preds_yolo_path = run_dir / f"preds_yolo_{args.split}.json"
    report_path_default = run_dir / f"report_{args.split}.json"

    analysis_path = Path(args.analysis_json) if args.analysis_json else (run_dir / f"analysis_{args.split}_iou{int(args.iou_thr*100):02d}.json")
    report_path = Path(args.report_json) if args.report_json else report_path_default

    if not preds_gd_path.exists():
        raise FileNotFoundError(f"missing: {preds_gd_path}")
    if not preds_yolo_path.exists():
        raise FileNotFoundError(f"missing: {preds_yolo_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"missing: {report_path}")

    coco = COCO(str(ann_path))
    img_ids = coco.getImgIds()

    report = load_json(report_path)
    analysis = load_json(analysis_path) if analysis_path.exists() else None

    out_root = run_dir / args.out_subdir / args.split
    ensure_dir(out_root)

    # extract thresholds
    score_thr_gd = 0.0
    score_thr_yolo = 0.25
    if analysis is not None:
        try:
            score_thr_gd = float(analysis["models"]["grounding_dino"]["best_by_f1"]["best_score_thr"])
        except Exception:
            pass
        try:
            score_thr_yolo = float(analysis["models"]["yolo"]["best_by_f1"]["best_score_thr"])
        except Exception:
            pass

    if args.score_thr_gd is not None:
        score_thr_gd = float(args.score_thr_gd)
    if args.score_thr_yolo is not None:
        score_thr_yolo = float(args.score_thr_yolo)

    # load preds
    preds_gd = load_json(preds_gd_path)
    preds_yolo = load_json(preds_yolo_path)

    preds_gd_by_img = parse_preds(preds_gd)
    preds_yolo_by_img = parse_preds(preds_yolo)

    gts_by_img = parse_gts(coco, img_ids)

    # 1) tables
    cols_m, rows_m = build_metrics_tables(report)
    table_to_png(out_root / "table_cocoeval_metrics.png", f"COCOeval metrics ({args.split})", cols_m, rows_m)

    if analysis is not None:
        cols_f, rows_f = build_f1_tables(analysis)
        table_to_png(out_root / "table_best_f1_iou50.png", f"Best F1 at IoU {args.iou_thr:.2f} ({args.split})", cols_f, rows_f)

    # 2) bar plots for COCOeval metrics
    metrics_by_model = {}
    for r in report.get("results", []):
        metrics_by_model[r.get("name", "model")] = r.get("metrics", {})

    plot_bar_metrics(
        out_root / "bars_ap_family.png",
        f"AP family ({args.split})",
        metrics_by_model,
        keys=["AP", "AP50", "AP75"],
    )
    plot_bar_metrics(
        out_root / "bars_ap_by_area.png",
        f"AP by area ({args.split})",
        metrics_by_model,
        keys=["AP_small", "AP_medium", "AP_large"],
    )
    plot_bar_metrics(
        out_root / "bars_ar100.png",
        f"AR at maxDets=100 ({args.split})",
        metrics_by_model,
        keys=["AR_100"],
    )

    # 3) latency plot from report
    timing = report.get("timing_sec", {})
    plot_latency(out_root / "latency_bars.png", f"Latency summary ({args.split})", timing)

    # 4) PR and F1 curves from analysis grid
    if analysis is not None:
        for model_name in ["grounding_dino", "yolo"]:
            grid = analysis.get("models", {}).get(model_name, {}).get("threshold_grid", [])
            if grid:
                plot_pr_curve_from_grid(
                    out_root / f"pr_{model_name}.png",
                    f"PR curve ({model_name}) IoU {args.iou_thr:.2f} ({args.split})",
                    grid,
                )

    # 5) score and IoU histograms using our own matching on selected best thresholds
    def collect_tp_stats(model: str, score_thr: float) -> Tuple[List[float], List[float], List[int], List[int], List[int]]:
        ious = []
        scores = []
        per_img_tp = []
        per_img_fp = []
        per_img_fn = []
        by_img = preds_gd_by_img if model == "grounding_dino" else preds_yolo_by_img
        for iid in img_ids:
            m = match_one_image(gts_by_img.get(iid, []), by_img.get(iid, []), score_thr, args.iou_thr)
            tp = m["tp_pairs"]
            fp = m["fp_preds"]
            fn = m["fn_gts"]
            per_img_tp.append(len(tp))
            per_img_fp.append(len(fp))
            per_img_fn.append(len(fn))
            for t in tp:
                ious.append(float(t["iou"]))
                scores.append(float(t["score"]))
        return ious, scores, per_img_tp, per_img_fp, per_img_fn

    gd_ious, gd_scores, gd_tp, gd_fp, gd_fn = collect_tp_stats("grounding_dino", score_thr_gd)
    y_ious, y_scores, y_tp, y_fp, y_fn = collect_tp_stats("yolo", score_thr_yolo)

    plot_hist(out_root / "hist_tp_iou_grounding_dino.png", f"TP IoU distribution (Grounding DINO) ({args.split})", gd_ious, "IoU")
    plot_hist(out_root / "hist_tp_score_grounding_dino.png", f"TP score distribution (Grounding DINO) ({args.split})", gd_scores, "Score")

    plot_hist(out_root / "hist_tp_iou_yolo.png", f"TP IoU distribution (YOLO) ({args.split})", y_ious, "IoU")
    plot_hist(out_root / "hist_tp_score_yolo.png", f"TP score distribution (YOLO) ({args.split})", y_scores, "Score")

    plot_hist(out_root / "hist_per_image_tp_grounding_dino.png", f"Per image TP (Grounding DINO) ({args.split})", gd_tp, "TP count", bins=20)
    plot_hist(out_root / "hist_per_image_fp_grounding_dino.png", f"Per image FP (Grounding DINO) ({args.split})", gd_fp, "FP count", bins=20)
    plot_hist(out_root / "hist_per_image_fn_grounding_dino.png", f"Per image FN (Grounding DINO) ({args.split})", gd_fn, "FN count", bins=20)

    plot_hist(out_root / "hist_per_image_tp_yolo.png", f"Per image TP (YOLO) ({args.split})", y_tp, "TP count", bins=20)
    plot_hist(out_root / "hist_per_image_fp_yolo.png", f"Per image FP (YOLO) ({args.split})", y_fp, "FP count", bins=20)
    plot_hist(out_root / "hist_per_image_fn_yolo.png", f"Per image FN (YOLO) ({args.split})", y_fn, "FN count", bins=20)

    # 6) qualitative overlays
    # choose a mixed set if requested
    if args.qual_mode == "mixed":
        ids_fn = pick_hard_images(coco, img_ids, gts_by_img, preds_gd_by_img, score_thr_gd, args.iou_thr, max(1, args.qual_k // 2), args.seed, "fn_heavy")
        ids_fp = pick_hard_images(coco, img_ids, gts_by_img, preds_yolo_by_img, score_thr_yolo, args.iou_thr, max(1, args.qual_k // 2), args.seed, "fp_heavy")
        chosen = []
        for a, b in zip(ids_fn, ids_fp):
            chosen.append(a)
            chosen.append(b)
        chosen = chosen[: args.qual_k]
    else:
        # drive selection by grounding dino difficulty by default
        base_preds = preds_gd_by_img if args.qual_mode in ["fn_heavy", "random"] else preds_yolo_by_img
        score_thr = score_thr_gd if base_preds is preds_gd_by_img else score_thr_yolo
        chosen = pick_hard_images(coco, img_ids, gts_by_img, base_preds, score_thr, args.iou_thr, args.qual_k, args.seed, args.qual_mode)

    qual_dir = out_root / "qualitative"
    ensure_dir(qual_dir)

    qual_items = []

    for iid in chosen:
        info = coco.loadImgs([iid])[0]
        fn = info["file_name"]
        img_path = img_dir / fn
        if not img_path.exists():
            continue

        im = Image.open(img_path).convert("RGB")
        gts = gts_by_img.get(iid, [])

        # Grounding DINO image
        gd_preds = []
        for p in preds_gd_by_img.get(iid, []):
            s = safe_float(p.get("score"))
            if s is None or s < score_thr_gd:
                continue
            gd_preds.append(p)
        gd_preds.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

        # YOLO image
        y_preds = []
        for p in preds_yolo_by_img.get(iid, []):
            s = safe_float(p.get("score"))
            if s is None or s < score_thr_yolo:
                continue
            y_preds.append(p)
        y_preds.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

        m_gd = match_one_image(gts, preds_gd_by_img.get(iid, []), score_thr_gd, args.iou_thr)
        m_y = match_one_image(gts, preds_yolo_by_img.get(iid, []), score_thr_yolo, args.iou_thr)

        gd_img = draw_boxes(
            im,
            gts,
            gd_preds,
            [
                f"image_id {iid}   {fn}",
                f"Grounding DINO   thr {score_thr_gd:.2f}   TP {len(m_gd['tp_pairs'])} FP {len(m_gd['fp_preds'])} FN {len(m_gd['fn_gts'])}",
            ],
        )
        y_img = draw_boxes(
            im,
            gts,
            y_preds,
            [
                f"image_id {iid}   {fn}",
                f"YOLO11x   thr {score_thr_yolo:.2f}   TP {len(m_y['tp_pairs'])} FP {len(m_y['fp_preds'])} FN {len(m_y['fn_gts'])}",
            ],
        )

        out_gd = qual_dir / f"{iid:06d}_gd.png"
        out_y = qual_dir / f"{iid:06d}_yolo.png"
        gd_img.save(out_gd)
        y_img.save(out_y)

        qual_items.append((f"{iid} Grounding DINO overlay", str(out_gd.relative_to(out_root)).replace("\\", "/")))
        qual_items.append((f"{iid} YOLO overlay", str(out_y.relative_to(out_root)).replace("\\", "/")))

    # 7) optional csv exports for easy copy paste
    if pd is not None:
        df_m = pd.DataFrame(rows_m, columns=cols_m)
        df_m.to_csv(out_root / "table_cocoeval_metrics.csv", index=False)

        if analysis is not None:
            df_f = pd.DataFrame(rows_f, columns=cols_f)  # type: ignore
            df_f.to_csv(out_root / "table_best_f1_iou50.csv", index=False)

    # 8) build HTML index
    sections: List[Tuple[str, List[Tuple[str, str]]]] = []

    plots = [
        ("COCOeval metrics table", "table_cocoeval_metrics.png"),
        ("AP family bars", "bars_ap_family.png"),
        ("AP by area bars", "bars_ap_by_area.png"),
        ("AR100 bar", "bars_ar100.png"),
        ("Latency bars", "latency_bars.png"),
        ("TP IoU hist (Grounding DINO)", "hist_tp_iou_grounding_dino.png"),
        ("TP score hist (Grounding DINO)", "hist_tp_score_grounding_dino.png"),
        ("TP IoU hist (YOLO)", "hist_tp_iou_yolo.png"),
        ("TP score hist (YOLO)", "hist_tp_score_yolo.png"),
        ("Per image TP (Grounding DINO)", "hist_per_image_tp_grounding_dino.png"),
        ("Per image FP (Grounding DINO)", "hist_per_image_fp_grounding_dino.png"),
        ("Per image FN (Grounding DINO)", "hist_per_image_fn_grounding_dino.png"),
        ("Per image TP (YOLO)", "hist_per_image_tp_yolo.png"),
        ("Per image FP (YOLO)", "hist_per_image_fp_yolo.png"),
        ("Per image FN (YOLO)", "hist_per_image_fn_yolo.png"),
    ]
    if analysis is not None:
        plots.insert(1, ("Best F1 table at IoU 0.50", "table_best_f1_iou50.png"))
        plots.extend([
            ("PR curve Grounding DINO", "pr_grounding_dino.png"),
            ("PR curve YOLO", "pr_yolo.png"),
        ])

    plot_items = [(cap, rel) for cap, rel in plots if (out_root / rel).exists()]
    sections.append(("Summary plots", plot_items))

    if qual_items:
        sections.append(("Qualitative overlays", qual_items))

    html_path = out_root / "index.html"
    make_html_index(html_path, f"PromptGroundBoxBench visual report ({args.split})", sections)

    # write a small manifest
    manifest = {
        "root": str(root),
        "split": args.split,
        "run_dir": str(run_dir),
        "out_root": str(out_root),
        "iou_thr": float(args.iou_thr),
        "score_thr": {"grounding_dino": float(score_thr_gd), "yolo": float(score_thr_yolo)},
        "files": {
            "html": str(html_path),
        },
    }
    save_json(out_root / "visual_manifest.json", manifest)

    print(f"wrote visuals to: {out_root}")
    print(f"open: {html_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
