"""P1-01 learner scaffold, not a completed implementation.

Chinese walkthrough: docs/MY_OPS_IMPLEMENTATION.md.
Importing this file is safe. Calling an unfinished operation raises
NotImplementedError on purpose; it must not be reported as a passed candidate.
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
