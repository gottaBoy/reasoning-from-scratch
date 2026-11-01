"""Small forward-only CUDA/Triton references with explicit input contracts."""

import torch
import triton
import triton.language as tl


@triton.jit
def _add(X, Y, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + index, tl.load(X + index, index < N, 0) + tl.load(Y + index, index < N, 0),
             index < N)


@triton.jit
def _rows(X, W, OUT, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
          MODE: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    if MODE == 0:  # RMSNorm
        scale = tl.rsqrt(tl.sum(x * x, 0) / WIDTH + EPS)
        y = x * scale * tl.load(W + col, col < WIDTH, 0).to(tl.float32)
    elif MODE == 1:  # Stable row softmax
        x = tl.where(col < WIDTH, x, float("-inf"))
        y = tl.exp(x - tl.max(x, 0))
        y = y / tl.sum(y, 0)
    else:  # Non-affine LayerNorm; subtract the mean before the variance reduction.
        mean = tl.sum(x, 0) / WIDTH
        centered = tl.where(col < WIDTH, x - mean, 0.0)
        var = tl.sum(centered * centered, 0) / WIDTH
        y = centered * tl.rsqrt(var + EPS)
    tl.store(OUT + row * WIDTH + col, y, col < WIDTH)


@triton.jit
def _swiglu(G, U, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + index, index < N, 0).to(tl.float32)
    u = tl.load(U + index, index < N, 0).to(tl.float32)
    tl.store(OUT + index, (g / (1 + tl.exp(-g))) * u, index < N)


@triton.jit
def _matmul(A, B, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        offsets = start * BK + kk
        a = tl.load(A + rows[:, None] * K + offsets[None, :],
                    (rows[:, None] < M) & (offsets[None, :] < K), 0)
        b = tl.load(B + offsets[:, None] * N + cols[None, :],
                    (offsets[:, None] < K) & (cols[None, :] < N), 0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(OUT + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _attention(Q, K, V, OUT, T: tl.constexpr, D: tl.constexpr,
               BQ: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    bh = tl.program_id(1)
    qi = tl.program_id(0) * BQ + tl.arange(0, BQ)
    ki = tl.arange(0, BK)
    di = tl.arange(0, BD)
    base = bh * T * D
    q = tl.load(Q + base + qi[:, None] * D + di[None, :],
                (qi[:, None] < T) & (di[None, :] < D), 0)
    maximum = tl.full((BQ,), float("-inf"), tl.float32)
    denom = tl.full((BQ,), 0.0, tl.float32)
    acc = tl.full((BQ, BD), 0.0, tl.float32)
    for start in range(tl.cdiv(T, BK)):
        pos = start * BK + ki
        k = tl.load(K + base + pos[:, None] * D + di[None, :],
                    (pos[:, None] < T) & (di[None, :] < D), 0)
        v = tl.load(V + base + pos[:, None] * D + di[None, :],
                    (pos[:, None] < T) & (di[None, :] < D), 0)
        score = tl.dot(q, tl.trans(k), input_precision="ieee") * (D ** -0.5)
        # Invalid query rows retain one finite dummy score; they are never stored.
        allowed = (pos[None, :] < T) & (pos[None, :] <= qi[:, None])
        score = tl.where(allowed, score, float("-inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        scale = tl.exp(maximum - new_max)
        p = tl.exp(score - new_max[:, None])
        acc = acc * scale[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        denom = denom * scale + tl.sum(p, 1)
        maximum = new_max
    tl.store(OUT + base + qi[:, None] * D + di[None, :], acc / denom[:, None],
             (qi[:, None] < T) & (di[None, :] < D))


def _check(*tensors):
    if any(not x.is_cuda or not x.is_contiguous() or x.dtype not in
           (torch.float16, torch.bfloat16, torch.float32) for x in tensors):
        raise ValueError("Contiguous CUDA FP32/FP16/BF16 tensors required.")
    if any(x.device != tensors[0].device or x.dtype != tensors[0].dtype for x in tensors):
        raise ValueError("Device and dtype must match.")
    if any(x.requires_grad for x in tensors):
        raise ValueError("Teaching kernels are forward-only; detach inputs first.")


def add(x, y):
    _check(x, y)
    if x.shape != y.shape:
        raise ValueError("Shapes must match.")
    out = torch.empty_like(x)
    if x.numel():
        with torch.cuda.device(x.device):
            _add[(triton.cdiv(x.numel(), 256),)](x, y, out, x.numel(), 256)
    return out


def row_op(x, mode="rmsnorm", weight=None, eps=1e-6):
    _check(x)
    modes = {"rmsnorm": 0, "softmax": 1, "layernorm": 2}
    if x.ndim != 2 or not 1 <= x.shape[-1] <= 8192 or mode not in modes or eps <= 0:
        raise ValueError("Use a 2D row width 1..8192, valid mode and positive epsilon.")
    weight = torch.ones(x.shape[1], device=x.device, dtype=x.dtype) if weight is None else weight
    _check(weight)
    if weight.device != x.device or weight.dtype != x.dtype or weight.shape != (x.shape[1],):
        raise ValueError("Weight shape/device/dtype mismatch.")
    out = torch.empty_like(x)
    if x.shape[0]:
        with torch.cuda.device(x.device):
            _rows[(x.shape[0],)](x, weight, out, x.shape[1],
                                triton.next_power_of_2(x.shape[1]), modes[mode], eps)
    return out


def swiglu(gate, up):
    _check(gate, up)
    if gate.shape != up.shape:
        raise ValueError("Shapes must match.")
    out = torch.empty_like(gate)
    if out.numel():
        with torch.cuda.device(out.device):
            _swiglu[(triton.cdiv(out.numel(), 256),)](gate, up, out, out.numel(), 256)
    return out


def matmul(a, b, tile=32):
    _check(a, b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or tile not in (16, 32, 64):
        raise ValueError("Compatible matrices and tile 16/32/64 required.")
    m, k = a.shape
    n = b.shape[1]
    if min(m, n, k) < 1:
        raise ValueError("Nonempty matrices required.")
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    with torch.cuda.device(a.device):
        _matmul[(triton.cdiv(m, tile), triton.cdiv(n, tile))](a, b, out, m, n, k,
                                                          tile, tile, 32)
    return out


def attention(q, k, v):
    _check(q, k, v)
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Matching [B,H,T,D] required; no cache, padding or GQA here.")
    b, h, t, d = q.shape
    if min(b, h, t) < 1 or d not in (16, 32, 64):
        raise ValueError("Nonempty inputs and D=16/32/64 required.")
    out = torch.empty_like(q)
    with torch.cuda.device(q.device):
        _attention[(triton.cdiv(t, 16), b * h)](q, k, v, out, t, d, 16, 32,
                                               triton.next_power_of_2(d))
    return out
