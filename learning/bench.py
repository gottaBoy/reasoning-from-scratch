"""Small synchronized latency harness. JIT/warm-up excluded from timed samples."""

from statistics import median
import time

import torch


def measure(fn, device="cpu", warmup=3, repeats=10):
    device = torch.device(device)
    if repeats < 1 or warmup < 0:
        raise ValueError("Invalid sample counts.")
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(1000 * (time.perf_counter() - start))
    return {
        "median_ms": median(samples),
        "min_ms": min(samples),
        "repeats": repeats,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "method": "synchronized end-to-end wall time, not kernel-only latency",
    }
