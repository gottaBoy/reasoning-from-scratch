"""Compile and execute a small Triton kernel, including a masked tail."""

import json

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, output, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    values = tl.load(x + offsets, mask=mask, other=0) + tl.load(y + offsets, mask=mask, other=0)
    tl.store(output + offsets, values, mask=mask)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Triton check requires CUDA, not a CPU PyTorch wheel.")
    torch.manual_seed(123)
    sizes = (1003, 65537)
    for size in sizes:
        x = torch.randn(size, device="cuda")
        y = torch.randn_like(x)
        output = torch.empty_like(x)
        add_kernel[(triton.cdiv(size, 256),)](x, y, output, size, BLOCK_SIZE=256)
        torch.testing.assert_close(output, x + y)
    torch.cuda.synchronize()
    print(json.dumps({
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "sizes": sizes,
        "vector_add": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
