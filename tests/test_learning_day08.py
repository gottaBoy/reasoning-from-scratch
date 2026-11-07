"""Day 8 CPU contracts: independent FP64 oracles, not production GPU benchmarks."""

import pytest
import torch

from learning.mini_transformer import RMSNorm, rope


def pairwise_rope_oracle(x, start, theta=10_000.0):
    reference = x.double()
    output = torch.empty_like(reference)
    dim = x.shape[-1]
    positions = torch.arange(start, start + x.shape[-2], dtype=torch.float64)
    for pair in range(dim // 2):
        angle = positions * theta ** (-2 * pair / dim)
        left, right = reference[..., pair], reference[..., pair + dim // 2]
        output[..., pair] = left * angle.cos() - right * angle.sin()
        output[..., pair + dim // 2] = left * angle.sin() + right * angle.cos()
    return output


@pytest.mark.parametrize("dtype,atol,rtol", [
    (torch.float32, 1e-6, 1e-5),
    (torch.float16, 2e-3, 2e-3),
    (torch.bfloat16, 2e-2, 1e-2),
])
@pytest.mark.parametrize("width", [3, 5])
def test_rmsnorm_rows_and_dtypes_match_fp64_formula(dtype, atol, rtol, width):
    values = torch.tensor([0., 2., 1e4, -1e4, 1e-4, -2e-4], dtype=dtype)
    x = values[:, None].expand(-1, width).clone().reshape(2, 3, width)
    gamma = torch.linspace(0.5, 1.5, width)
    module = RMSNorm(width, eps=1e-6)
    with torch.no_grad():
        module.weight.copy_(gamma)
    expected = x.double() / (x.double().square().mean(-1, keepdim=True) + 1e-6).sqrt()
    expected *= gamma.double()
    actual = module(x)
    assert actual.shape == x.shape and actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, atol=atol, rtol=rtol)


def test_rmsnorm_does_not_mix_tokens_or_center_constant_rows():
    module = RMSNorm(3)
    x = torch.tensor([[[2., 2., 2.], [1., 2., -3.]]])
    baseline = module(x)
    altered = x.clone()
    altered[:, 1] *= 99
    torch.testing.assert_close(module(altered)[:, 0], baseline[:, 0], atol=0, rtol=0)
    assert baseline[0, 0].mean().item() > 0.9


def test_rmsnorm_gradients_exist_and_are_finite():
    x = torch.tensor([[[1., 2., 3.], [-2., 1., 4.]]], requires_grad=True)
    module = RMSNorm(3)
    module(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()


@pytest.mark.parametrize("dim", [2, 4, 8])
@pytest.mark.parametrize("start", [0, 5])
def test_rope_matches_independent_split_halves_rotations(dim, start):
    generator = torch.Generator().manual_seed(8)
    x = torch.randn(2, 3, 7, dim, generator=generator)
    actual = rope(x, start=start)
    expected = pairwise_rope_oracle(x, start)
    assert actual.shape == x.shape and actual.dtype == x.dtype
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(actual.norm(dim=-1), x.norm(dim=-1),
                               atol=2e-6, rtol=1e-5)
    if start == 0:
        torch.testing.assert_close(actual[:, :, 0], x[:, :, 0], atol=0, rtol=0)


def test_rope_chunk_positions_align_and_wrong_reset_is_detected():
    x = torch.randn(1, 2, 7, 4, generator=torch.Generator().manual_seed(8))
    full = rope(x)
    chunks = torch.cat([rope(x[:, :, :3]), rope(x[:, :, 3:], start=3)], dim=2)
    wrong = torch.cat([rope(x[:, :, :3]), rope(x[:, :, 3:], start=0)], dim=2)
    torch.testing.assert_close(chunks, full, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(wrong, full)


def test_rope_same_content_dot_depends_on_relative_positions():
    q = torch.tensor([[[[1., 2., 3., 4.]]]])
    k = torch.tensor([[[[4., -1., 2., 3.]]]])
    before = (rope(q, 2) * rope(k, 5)).sum(-1)
    shifted = (rope(q, 7) * rope(k, 10)).sum(-1)
    torch.testing.assert_close(before, shifted, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dim,start", [(3, 0), (4, -1)])
def test_rope_rejects_odd_dimension_or_negative_offset(dim, start):
    with pytest.raises(ValueError):
        rope(torch.zeros(1, 1, 2, dim), start=start)
