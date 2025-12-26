from __future__ import annotations

from time import perf_counter
from typing import Callable, TypeVar

T = TypeVar("T")


def cuda_synchronize_if_available() -> None:
    """Synchronize CUDA for accurate timing if torch and CUDA are available."""
    try:
        import torch  # type: ignore
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(fn: Callable[[], T], sync_cuda: bool) -> tuple[T, float]:
    """Time a callable, optionally synchronizing CUDA around timing.

    Returns
    result, elapsed_seconds
    """
    if sync_cuda:
        cuda_synchronize_if_available()
    t0 = perf_counter()
    result = fn()
    if sync_cuda:
        cuda_synchronize_if_available()
    t1 = perf_counter()
    return result, (t1 - t0)
