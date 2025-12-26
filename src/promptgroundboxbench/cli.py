from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from promptgroundboxbench.benchmarks.speed import benchmark_speed
from promptgroundboxbench.config import ConfigBundle, load_config_bundle
from promptgroundboxbench.demo.gradio_app import build_demo
from promptgroundboxbench.engines.grounding_dino import GroundingDINOEngine
from promptgroundboxbench.engines.yolo import YOLOEngine
from promptgroundboxbench.eval.coco import eval_coco_bbox, load_label_list
from promptgroundboxbench.utils.image import load_rgb_image
from promptgroundboxbench.utils.io import make_run_dirs, save_resolved_config, write_json
from promptgroundboxbench.utils.prompt import parse_prompt

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _engine_from_cfg(cfgb: ConfigBundle, engine: str):
    cfg = cfgb.raw
    device = cfgb.resolved_device
    if engine == "grounding_dino":
        gd = cfg.get("grounding_dino", {})
        return GroundingDINOEngine(
            model_id=str(gd.get("model_id")),
            device=device,
            box_threshold=float(gd.get("box_threshold", 0.35)),
            text_threshold=float(gd.get("text_threshold", 0.25)),
            label_normalize=bool(gd.get("label_normalize", True)),
        )
    if engine == "yolo":
        y = cfg.get("yolo", {})
        return YOLOEngine(
            weights=str(y.get("weights")),
            device=device,
            conf=float(y.get("conf", 0.25)),
            label_normalize=True,
        )
    raise typer.BadParameter("engine must be grounding_dino or yolo")


@app.callback()
def main(
    ctx: typer.Context,
    config: Path = typer.Option(Path("configs/defaults.yaml"), help="Path to YAML config"),
    set: list[str] = typer.Option(None, "--set", help="Override config keys, repeatable: a.b=1"),
):
    overrides = set or []
    cfgb = load_config_bundle(config, overrides)
    ctx.obj = cfgb


@app.command()
def demo(
    ctx: typer.Context,
    share: bool = typer.Option(False, help="Expose public Gradio link"),
):
    """Run the Gradio demo (Grounding DINO)."""
    cfgb: ConfigBundle = ctx.obj
    cfg = cfgb.raw
    gd = cfg.get("grounding_dino", {})
    engine = GroundingDINOEngine(
        model_id=str(gd.get("model_id")),
        device=cfgb.resolved_device,
        box_threshold=float(gd.get("box_threshold", 0.35)),
        text_threshold=float(gd.get("text_threshold", 0.25)),
        label_normalize=bool(gd.get("label_normalize", True)),
    )
    sync_cuda = cfgb.resolved_device == "cuda"
    demo_app = build_demo(engine, sync_cuda=sync_cuda)
    demo_app.launch(share=share)


@app.command()
def infer(
    ctx: typer.Context,
    engine: str = typer.Argument(..., help="grounding_dino or yolo"),
    image: Path = typer.Option(..., "--image", exists=False, help="Path to an RGB image"),
    prompt: Optional[str] = typer.Option(None, help="Prompt labels for Grounding DINO"),
):
    """Run single image inference and save JSON to runs/."""
    cfgb: ConfigBundle = ctx.obj
    cfg = cfgb.raw

    engine_obj = _engine_from_cfg(cfgb, engine)

    img = load_rgb_image(image)
    labels = parse_prompt(prompt) if prompt else None
    det = engine_obj.predict(img, labels).clip_to_image()

    run_dir, report_dir = make_run_dirs(cfg, tag=f"infer_{engine}")
    save_resolved_config(cfg, run_dir)

    out = {
        "engine": engine,
        "device": cfgb.resolved_device,
        "image": str(image),
        "n": det.n,
        "boxes_xyxy": det.boxes_xyxy.tolist(),
        "labels": det.labels,
        "scores": det.scores.tolist(),
    }
    write_json(run_dir / "infer_output.json", out)
    write_json(report_dir / "summary.json", out)
    console.print(f"[green]Saved[/green] {run_dir}")


benchmark_app = typer.Typer(no_args_is_help=True)
app.add_typer(benchmark_app, name="benchmark")


@benchmark_app.command("speed")
def benchmark_speed_cmd(
    ctx: typer.Context,
    engine: str = typer.Option("grounding_dino", help="grounding_dino or yolo"),
    image: Path = typer.Option(..., "--image", help="Path to an RGB image"),
    prompt: Optional[str] = typer.Option("a person.", help="Prompt labels for Grounding DINO"),
):
    """Measure latency per image and derived FPS."""
    cfgb: ConfigBundle = ctx.obj
    cfg = cfgb.raw

    engine_obj = _engine_from_cfg(cfgb, engine)

    img = load_rgb_image(image)
    labels = parse_prompt(prompt) if prompt else None

    warmup = int(cfg.get("benchmark", {}).get("warmup", 2))
    runs = int(cfg.get("benchmark", {}).get("runs", 50))
    sync_cuda = cfgb.resolved_device == "cuda"

    summary = benchmark_speed(
        engine=engine_obj,
        image=img,
        prompt_labels=labels,
        warmup=warmup,
        runs=runs,
        sync_cuda=sync_cuda,
    )

    run_dir, report_dir = make_run_dirs(cfg, tag=f"speed_{engine}")
    save_resolved_config(cfg, run_dir)

    out = {
        "engine": engine,
        "device": cfgb.resolved_device,
        "image": str(image),
        "prompt": prompt,
        "summary": summary.to_dict(),
    }
    write_json(run_dir / "speed_summary.json", out)
    write_json(report_dir / "summary.json", out)
    console.print(f"[green]Saved[/green] {run_dir}")


eval_app = typer.Typer(no_args_is_help=True)
app.add_typer(eval_app, name="eval")


@eval_app.command("coco")
def eval_coco_cmd(
    ctx: typer.Context,
    engine: str = typer.Option("yolo", help="grounding_dino or yolo"),
):
    """Evaluate bbox AP on COCO val2017 using COCOeval."""
    cfgb: ConfigBundle = ctx.obj
    cfg = cfgb.raw

    engine_obj = _engine_from_cfg(cfgb, engine)

    paths = cfg.get("paths", {})
    coco_images = Path(str(paths.get("coco_images", "data/coco/val2017")))
    coco_ann = Path(str(paths.get("coco_annotations", "data/coco/annotations/instances_val2017.json")))

    eval_cfg = cfg.get("coco_eval", {})
    limit = int(eval_cfg.get("limit", 0))
    max_dets = int(eval_cfg.get("max_detections", 100))
    sync_cuda = cfgb.resolved_device == "cuda"

    prompt_labels = None
    if engine == "grounding_dino":
        label_file = Path(str(paths.get("coco80_labels", "configs/coco80_labels.txt")))
        prompt_labels = load_label_list(label_file)

    run_dir, report_dir = make_run_dirs(cfg, tag=f"coco_{engine}")
    save_resolved_config(cfg, run_dir)

    preds_path, result = eval_coco_bbox(
        engine=engine_obj,
        coco_images_dir=coco_images,
        coco_ann_file=coco_ann,
        prompt_labels=prompt_labels,
        limit=limit,
        max_detections=max_dets,
        sync_cuda=sync_cuda,
    )

    preds_dst = run_dir / "preds_coco.json"
    preds_dst.write_bytes(preds_path.read_bytes())
    preds_path.unlink(missing_ok=True)

    metrics_out = {
        "engine": engine,
        "device": cfgb.resolved_device,
        "limit": limit,
        "max_detections": max_dets,
        "metrics": result.metrics,
        "stats": result.stats,
    }
    write_json(run_dir / "coco_metrics.json", metrics_out)
    write_json(report_dir / "summary.json", metrics_out)
    console.print(f"[green]Saved[/green] {run_dir}")
