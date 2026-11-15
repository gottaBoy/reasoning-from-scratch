"""Shared learner scaffold for P1-01 through P1-03, not completed implementations.

Read the hints here; implement each operation in learning/my_ops.py.
Keep this template unfinished and do not overwrite existing learner code.
P1-01: RMSNorm and rope; P1-02: causal_attention;
P1-03: split_heads and merge_heads.
P1-04: expand_kv_heads.
Chinese companion with the same interfaces and TODOs: learning/my_ops_template_zh.py.
Chinese walkthroughs:
    docs/MY_OPS_IMPLEMENTATION.md
    docs/PRACTICE_P1_02_ATTENTION.md
Importing this file is safe. Calling an unfinished operation raises
NotImplementedError on purpose; it must not be reported as a passed candidate.

Inside the existing Jupyter Terminal:
    cd /workspace
    python -m learning.local_practice --task norm-rope --device cpu \\
        --candidate learning/my_ops.py
    python -m learning.local_practice --task attention --device cpu \\
        --candidate learning/my_ops.py
    python -m learning.local_practice --task layout --device cpu \\
        --candidate learning/my_ops.py

After CPU acceptance, repeat with --device cuda. Each task checks only its
required exports, so an unfinished Attention body does not block P1-01.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Input [..., width] -> same shape, dtype, and device."""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        if width < 1 or eps <= 0:
            raise ValueError("Positive width and epsilon required.")
        self.eps = eps
        # Gamma belongs to the module, not to an individual input token.
        # The acceptance runner requires the parameter name "weight".
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.weight.numel():
            raise ValueError("The last dimension must match width.")
        input_dtype = x.dtype
        x_fp32 = x.float()

        # TODO R1: Compute mean_square from x_fp32 along dim=-1.
        # Square before taking the mean, with keepdim=True.
        # Shape: [..., width] -> [..., 1]. Do not subtract the mean.
        # TODO R2: Compute inv_rms = torch.rsqrt(mean_square + self.eps).
        # TODO R3: Compute normalized = x_fp32 * inv_rms.
        # Broadcast one normalization factor over each row's features.
        # TODO R4: Multiply normalized by self.weight to form output.
        # Restore input_dtype and return the result.
        # Preserve autograd: do not call detach() or enter no_grad here.
        # Create self.weight once in __init__, not on each forward call.
        _ = x_fp32, input_dtype
        raise NotImplementedError("Fill R1-R4 in your own implementation.")


def rope(x, start=0, theta=10_000.0):
    """Split-halves RoPE: input/output [B,H,T,D], D positive and even."""
    if x.ndim != 4:
        raise ValueError("Expected [B,H,T,D].")
    dim = x.shape[-1]
    if dim < 1 or dim % 2 or start < 0 or theta <= 0:
        raise ValueError("Positive even D, nonnegative start, positive theta required.")

    input_dtype = x.dtype
    x_fp32 = x.float()
    half = dim // 2

    # TODO P1: pair indices i=0..half-1 on x.device, using FP32.
    # TODO P2: frequency[i] = theta ** (-2*i/dim). Shape: [half].
    # TODO P3: positions = start..start+T-1 on x.device. Shape: [T].
    # TODO P4: angles = positions[:,None] * frequencies[None,:].
    # Shape: [T,half], broadcastable over [B,H,T,half].
    # TODO P5: left = x_fp32[..., :half], right = x_fp32[..., half:].
    # D=4 pairs are (x0,x2), (x1,x3), NOT (x0,x1), (x2,x3).
    # TODO P6: new_left = left*cos(angles) - right*sin(angles).
    #          new_right = left*sin(angles) + right*cos(angles).
    # TODO P7: concatenate new_left/new_right along dim=-1, restore dtype.
    # Do not modify x in-place or move a CUDA input back to CPU.
    _ = x_fp32, input_dtype, half
    raise NotImplementedError("Fill P1-P7 in your own implementation.")


def causal_attention(q, k, v, start=0):
    """P1-02: implement A1-A5 in learning/my_ops.py, preserving P1-01 code.

    Inputs and output
    -----------------
    q: Query vectors, [B,1,T_q,D_k].
    k: Key vectors, [B,1,T_kv,D_k].
    v: Value vectors, [B,1,T_kv,D_v].
    Output: [B,1,T_q,D_v], with the same device and FP32 dtype.
    B is Batch Size; this exercise fixes the head count to one.
    T_q and T_kv are Query and Key/Value Sequence Lengths, not feature widths.
    D_k is Query/Key Head Dimension; D_v is Value Head Dimension.
    D_v may differ from D_k. All inputs are nonempty, same-device FP32.
    Q/K/V are already projected; do not add Linear, RoPE or cache allocation.

    Hand calculation
    ----------------
    Q=[0, sqrt(2)*ln(3)], K0=[1,0], K1=[0,1], V0=[10], V1=[20].
    With start=1, both keys are visible.
    Scaled scores=[0,ln(3)] -> probabilities=[1/4,3/4] -> output=17.5.
    ln denotes the natural logarithm; these numbers do not model word meanings.

    Absolute positions and masking
    ------------------------------
    start is the first Query's absolute position, not a tensor slice offset.
    Prefill: start=0, T_q=T_kv=3; rows can read 1, 2 and 3 keys.
    Cached chunk: start=4, T_q=3, T_kv=7; rows can read 5, 6 and 7 keys.
    History K/V are supplied by the caller; this function does not store them.
    Fill future scores with -inf before softmax, not zero: exp(0)=1.

    Acceptance and deliberate mistakes
    ----------------------------------
    Run --task attention --candidate learning/my_ops.py, first CPU then CUDA.
    Fourteen checks cover numeric results, rectangular masks, gradients,
    unchanged inputs and invalid arguments, not timing or production readiness.
    After passing, break one thing: scaling, mask fill, softmax axis or start.
    Record the failing case, restore correctness and try a new input shape.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Use [B,1,T,D] tensors.")
    if (min(*q.shape, *k.shape, *v.shape) < 1
            or q.shape[1] != 1 or k.shape[1] != 1 or v.shape[1] != 1
            or q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3]
            or q.shape[-1] != k.shape[-1]):
        raise ValueError("Use compatible nonempty single-head inputs.")
    if any(x.device != q.device or x.dtype != torch.float32 for x in (q, k, v)):
        raise ValueError("Same-device FP32 inputs required.")
    if not isinstance(start, int) or isinstance(start, bool) or not 0 <= start <= k.shape[2] - q.shape[2]:
        raise ValueError("Invalid query start position.")

    # TODO A1: scores = q @ k.transpose(-2, -1), divided by sqrt(D_k).
    # D_k=q.shape[-1]. [T_q,D_k] @ [D_k,T_kv] -> [T_q,T_kv].
    # The dot product sums over D_k; preserve the batch and head axes.
    # Full scores shape: [B,1,T_q,T_kv].

    # TODO A2: Build query_positions and key_positions on q.device.
    # Query positions: start..start+T_q-1; Key positions: 0..T_kv-1.
    # Use [:,None] for Query rows and [None,:] for Key columns.

    # TODO A3: blocked[i,j] = key_positions[j] > query_positions[i].
    # Shape: [T_q,T_kv], broadcast over batch/head. True means blocked.
    # Use masked_fill with -torch.inf. SDPA bool masks use True=allowed.

    # TODO A4: probabilities = softmax(scores, dim=-1), over Key positions.
    # Each Query row sums to one; shape remains [B,1,T_q,T_kv].
    # This exercise has no dropout.

    # TODO A5: output = probabilities @ v; return output.
    # [T_q,T_kv] @ [T_kv,D_v] -> [T_q,D_v]: sum over Key positions.
    # Full output: [B,1,T_q,D_v]. Query length comes from Q, width from V.
    # Preserve autograd: no detach(), no no_grad, no in-place input changes.
    # Do not replace your implementation with an import of the reference.
    raise NotImplementedError("Fill A1-A5 in your own implementation.")


def split_heads(x, heads, head_dim):
    """P1-03: [B,T,H*D] -> [B,H,T,D], layout only."""
    if x.ndim != 3:
        raise ValueError("Expected [B,T,H*D].")
    if (not isinstance(heads, int) or isinstance(heads, bool) or heads < 1
            or not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim < 1):
        raise ValueError("Positive integer heads and head_dim required.")
    if x.shape[-1] != heads * head_dim:
        raise ValueError("Last dimension must equal heads * head_dim.")
    if x.dtype != torch.float32:
        raise ValueError("This task requires FP32 input.")
    if not x.is_contiguous():
        raise ValueError("split_heads requires contiguous input.")

    # TODO S1: view x as [B,T,heads,head_dim].
    # This only reinterprets contiguous storage; it does not calculate a projection.

    # TODO S2: transpose the T and heads axes -> [B,heads,T,head_dim].
    # Keep the resulting stride visible; do not call contiguous here.
    raise NotImplementedError("Fill S1-S2 in your own implementation.")


def merge_heads(heads):
    """P1-03: [B,H,T,D] -> [B,T,H*D], layout only."""
    if heads.ndim != 4:
        raise ValueError("Expected [B,H,T,D].")
    if min(*heads.shape) < 1:
        raise ValueError("Nonempty [B,H,T,D] input required.")
    if heads.dtype != torch.float32:
        raise ValueError("This task requires FP32 input.")

    # TODO M1: transpose heads axes back -> [B,T,H,D].
    # The head dimension must become adjacent to D before flattening.

    # TODO M2: reshape the final H,D axes -> [B,T,H*D].
    # Use reshape after the transpose; direct reshape on [B,H,T,D] is the bug exercise.
    raise NotImplementedError("Fill M1-M2 in your own implementation.")


def expand_kv_heads(kv, query_heads):
    """P1-04: compact [B,H_kv,T,D] -> query-aligned [B,H_q,T,D]."""
    if kv.ndim != 4:
        raise ValueError("Expected [B,H_kv,T,D].")
    if min(*kv.shape) < 1:
        raise ValueError("Nonempty [B,H_kv,T,D] input required.")
    if not isinstance(query_heads, int) or isinstance(query_heads, bool) or query_heads < 1:
        raise ValueError("Positive integer query_heads required.")
    if kv.dtype != torch.float32:
        raise ValueError("This task requires FP32 input.")
    if query_heads % kv.shape[1] != 0:
        raise ValueError("query_heads must be divisible by kv_heads.")

    # TODO G1: group_size = query_heads // kv.shape[1].
    # One compact KV head serves group_size Query heads.

    # TODO G2: expanded = kv.repeat_interleave(group_size, dim=1).
    # Output shape: [B,query_heads,T,D]. Do not change or mutate compact kv.
    raise NotImplementedError("Fill G1-G2 in your own implementation.")
