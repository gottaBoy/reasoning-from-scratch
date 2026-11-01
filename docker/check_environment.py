"""Check the actual PyTorch execution path, not just GPU visibility."""

import argparse
import json
import platform
from pathlib import Path

import torch
from torch.nn import functional as F


def check_environment(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check container GPU injection and install a CUDA "
            "PyTorch wheel, not a +cpu wheel."
        )
    info = {
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "device": device,
    }
    if device == "cuda":
        info.update(
            gpu=torch.cuda.get_device_name(),
            capability=list(torch.cuda.get_device_capability()),
            compiled_architectures=torch.cuda.get_arch_list(),
        )
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(123)
    a = torch.randn(64, 64, device=device)
    result = a @ a.T
    torch.testing.assert_close(result.cpu(), a.cpu() @ a.cpu().T, atol=1e-4, rtol=1e-4)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    q = torch.randn(1, 4, 16, 32, device=device, dtype=dtype, requires_grad=True)
    output = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    output.float().square().mean().backward()
    if not torch.isfinite(output).all() or not torch.isfinite(q.grad).all():
        raise RuntimeError("Attention forward/backward produced non-finite values.")
    if device == "cuda":
        torch.cuda.synchronize()
        info["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 2)
    info.update(matmul="passed", attention_forward_backward="passed")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    info = check_environment(args.device)
    text = json.dumps(info, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
