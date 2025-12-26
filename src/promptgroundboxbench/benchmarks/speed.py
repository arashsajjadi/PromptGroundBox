from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from promptgroundboxbench.engines.base import DetectionEngine
from promptgroundboxbench.utils.timing import timed_call


@dataclass(frozen=True)
class SpeedSummary:
    warmup: int
    runs: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    fps: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup": self.warmup,
            "runs": self.runs,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "fps": self.fps,
        }


def benchmark_speed(
    engine: DetectionEngine,
    image: Image.Image,
    prompt_labels: list[str] | None,
    warmup: int,
    runs: int,
    sync_cuda: bool,
) -> SpeedSummary:
    if warmup < 0 or runs <= 0:
        raise ValueError("warmup must be >= 0 and runs must be > 0")

    for _ in range(warmup):
        engine.predict(image, prompt_labels)

    times: list[float] = []
    for _ in range(runs):
        _, dt = timed_call(lambda: engine.predict(image, prompt_labels), sync_cuda=sync_cuda)
        times.append(dt * 1000.0)

    arr = np.asarray(times, dtype=np.float64)
    mean = float(arr.mean())
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    fps = 1000.0 / mean if mean > 0 else 0.0

    return SpeedSummary(
        warmup=warmup,
        runs=runs,
        mean_ms=mean,
        median_ms=median,
        p95_ms=p95,
        fps=fps,
    )
