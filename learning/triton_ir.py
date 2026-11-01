"""Inspect real NVIDIA Triton compilation stages after a correct Vector Add launch."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch


STAGES = ("ttir", "ttgir", "llir", "ptx", "cubin")


def save_artifacts(artifacts, destination, report):
    """Preserve old experiment directories instead of silently replacing them."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for stage in STAGES:
        value = artifacts.get(stage)
        if value is None:
            continue
        path = destination / f"vector_add.{stage}"
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(value)
        files[stage] = path.name
    report["artifact_files"] = files
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return files


@torch.inference_mode()
def inspect_vector_add(size=1003, block=256, device="cuda", output=None):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This experiment requires NVIDIA CUDA; no CPU fallback.")
    if size < 1 or block not in (128, 256, 512):
        raise ValueError("Use size > 0 and block 128, 256 or 512.")
    import triton
    from learning.kernels.triton_ops import _add

    torch.manual_seed(123)
    with torch.cuda.device(device):
        x = torch.randn(size, device=device)
        y = torch.randn_like(x)
        result = torch.empty_like(x)
        compiled = _add[(triton.cdiv(size, block),)](
            x, y, result, size, block, num_warps=4
        )
        torch.cuda.synchronize(device)
        torch.testing.assert_close(result, x + y, rtol=0, atol=0)
    artifacts = compiled.asm
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
        "device": str(x.device),
        "gpu": torch.cuda.get_device_name(x.device),
        "capability": list(torch.cuda.get_device_capability(x.device)),
        "input_size": size,
        "block": block,
        "grid": [triton.cdiv(size, block)],
        "num_warps": 4,
        "numerical_check": "passed",
        "available_asm_keys": sorted(artifacts),
        "stages_present": {stage: stage in artifacts for stage in STAGES},
        "registers_per_thread": getattr(compiled, "n_regs", None),
        "spills": getattr(compiled, "n_spills", None),
        "shared_memory_bytes": getattr(compiled.metadata, "shared", None),
        "preview": {
            stage: "\n".join(artifacts[stage].splitlines()[:16])
            for stage in STAGES if isinstance(artifacts.get(stage), str)
        },
        "scope": "IR inspection plus output correctness; not a latency or speedup benchmark",
    }
    if output is not None:
        save_artifacts(artifacts, output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1003)
    parser.add_argument("--block", type=int, choices=(128, 256, 512), default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_vector_add(
        args.size, args.block, args.device, args.output_dir
    ), indent=2))


if __name__ == "__main__":
    main()
