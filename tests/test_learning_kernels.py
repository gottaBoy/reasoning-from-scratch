import pytest
import torch
from torch.nn import functional as F


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA lab requires GPU")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_triton_elementwise_and_rows(dtype):
    pytest.importorskip("triton")
    from learning.kernels import triton_ops as ops
    torch.manual_seed(1)
    for n in (0, 1, 255, 257, 1003):
        x = torch.randn(n, device="cuda", dtype=dtype)
        y = torch.randn_like(x)
        torch.testing.assert_close(ops.add(x, y), x + y)
        ref = (F.silu(x.float()) * y.float()).to(dtype)
        torch.testing.assert_close(ops.swiglu(x, y), ref, atol=2e-2, rtol=2e-2)
    for width in (1, 7, 127, 1003):
        x = torch.randn(3, width, device="cuda", dtype=dtype)
        weight = torch.randn(width, device="cuda", dtype=dtype)
        rms = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
               * weight.float()).to(dtype)
        torch.testing.assert_close(ops.row_op(x, "rmsnorm", weight), rms, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(ops.row_op(x, "softmax"), x.float().softmax(-1).to(dtype),
                                   atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(ops.row_op(x, "layernorm"),
                                   F.layer_norm(x.float(), (width,), eps=1e-6).to(dtype),
                                   atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("tile", [16, 32, 64])
def test_tiled_matmul_irregular_shapes(tile):
    pytest.importorskip("triton")
    from learning.kernels.triton_ops import matmul
    for shape in ((17, 33, 19), (32, 32, 32)):
        m, n, k = shape
        a, b = torch.randn(m, k, device="cuda"), torch.randn(k, n, device="cuda")
        # Other repository tests enable TF32 globally; use an independent FP64 oracle.
        expected = (a.cpu().double() @ b.cpu().double()).to(device=a.device, dtype=a.dtype)
        torch.testing.assert_close(matmul(a, b, tile), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("length", [1, 17, 65])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_online_attention_gpu_against_sdpa(length, dtype):
    pytest.importorskip("triton")
    from learning.kernels.triton_ops import attention
    torch.manual_seed(1)
    for dim in (16, 32, 64):
        q, k, v = [torch.randn(1, 2, length, dim, device="cuda", dtype=dtype) for _ in range(3)]
        expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                                 is_causal=True).to(dtype)
        torch.testing.assert_close(attention(q, k, v), expected, atol=2e-2, rtol=2e-2)


def test_kernel_contracts():
    pytest.importorskip("triton")
    from learning.kernels import triton_ops as ops
    with pytest.raises(ValueError):
        ops.add(torch.ones(8), torch.ones(8))
    x = torch.ones(2, 8, device="cuda", requires_grad=True)
    with pytest.raises(ValueError, match="forward-only"):
        ops.row_op(x)
