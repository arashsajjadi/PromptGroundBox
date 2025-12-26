from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ConfigBundle:
    """Resolved configuration used by a command."""

    raw: dict[str, Any]
    source_path: Path
    resolved_device: str


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(data)}")
    return data


def _parse_scalar(value: str) -> Any:
    # Let YAML parse ints, floats, bools, null, and quoted strings.
    return yaml.safe_load(value)


def set_by_dotted_key(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: dict[str, Any] = cfg
    for p in parts[:-1]:
        nxt = cur.get(p)
        if nxt is None:
            cur[p] = {}
            nxt = cur[p]
        if not isinstance(nxt, dict):
            raise ValueError(f"Cannot set {dotted_key}: {p} is not a mapping")
        cur = nxt
    cur[parts[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = dict(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov}")
        k, v = ov.split("=", 1)
        set_by_dotted_key(out, k.strip(), _parse_scalar(v.strip()))
    return out


def try_resolve_device(device_cfg: str) -> str:
    device_cfg = device_cfg.lower().strip()
    if device_cfg in {"cpu", "cuda"}:
        if device_cfg == "cuda" and not _torch_cuda_available():
            return "cpu"
        return device_cfg
    if device_cfg != "auto":
        raise ValueError("device must be one of auto, cpu, cuda")

    if _torch_cuda_available():
        return "cuda"
    return "cpu"


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def load_config_bundle(config_path: Path, overrides: list[str]) -> ConfigBundle:
    raw = load_yaml(config_path)
    raw = apply_overrides(raw, overrides)

    device_cfg = str(raw.get("device", "auto"))
    resolved_device = try_resolve_device(device_cfg)

    # Persist the resolved device for reproducibility
    raw["device_resolved"] = resolved_device

    return ConfigBundle(raw=raw, source_path=config_path, resolved_device=resolved_device)


def save_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
