from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from promptgroundboxbench.types import Detection


@dataclass(frozen=True)
class EngineMeta:
    name: str
    device: str


class DetectionEngine(Protocol):
    meta: EngineMeta

    def predict(self, image: Image.Image, labels: list[str] | None = None) -> Detection:
        """Run inference and return a Detection."""
        raise NotImplementedError
