from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from promptgroundboxbench.config import save_yaml


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dirs(cfg: dict[str, Any], tag: str) -> tuple[Path, Path]:
    runs_dir = Path(cfg["paths"]["runs_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    run_id = f"{timestamp_id()}_{tag}"
    run_dir = runs_dir / run_id
    report_dir = reports_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, report_dir


def save_resolved_config(cfg: dict[str, Any], run_dir: Path) -> None:
    save_yaml(cfg, run_dir / "config_resolved.yaml")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
