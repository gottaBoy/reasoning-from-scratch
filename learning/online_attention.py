"""Streaming softmax reference in PyTorch; not a FlashAttention GPU kernel."""

import math

import torch


def online_softmax_chunks(values, chunk=3):
    if values.ndim != 1 or not values.numel() or chunk < 1:
        raise ValueError("Nonempty vector and positive chunk required.")
    maximum = values.new_tensor(-torch.inf)
    denominator = values.new_zeros(())
    history = []
    for part in values.split(chunk):
        new_max = torch.maximum(maximum, part.max())
        denominator = denominator * torch.exp(maximum - new_max) + torch.exp(part - new_max).sum()
        maximum = new_max
        history.append({"max": maximum.item(), "denominator": denominator.item()})
    return torch.exp(values - maximum) / denominator, history


def blocked_attention(q, k, v, query_block=4, key_block=3, causal=True, offset=0):
    """Tile both Q and K. Forward-only teaching path, no full T*T score allocation."""
    if q.ndim != 4 or k.shape != v.shape or q.shape[:2] != k.shape[:2]:
        raise ValueError("Use matching [B,H,T,D] Q/K/V with K and V of equal shape.")
    if q.shape[-1] != k.shape[-1] or min(query_block, key_block, q.shape[2], k.shape[2]) < 1:
        raise ValueError("Incompatible/empty dimensions.")
    if causal and (offset < 0 or offset + q.shape[2] > k.shape[2]):
        raise ValueError("Invalid causal offset.")
    output = torch.empty_like(q)
    for qs in range(0, q.shape[2], query_block):
        query = q[:, :, qs:qs + query_block].float()
        maximum = torch.full(query.shape[:-1], -torch.inf, device=q.device)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(query)
        for ks in range(0, k.shape[2], key_block):
            key = k[:, :, ks:ks + key_block].float()
            value = v[:, :, ks:ks + key_block].float()
            scores = query @ key.transpose(-1, -2) / math.sqrt(q.shape[-1])
            if causal:
                qi = torch.arange(offset + qs, offset + qs + query.shape[2], device=q.device)
                ki = torch.arange(ks, ks + key.shape[2], device=q.device)
                scores = scores.masked_fill(ki[None, :] > qi[:, None], -torch.inf)
            new_max = torch.maximum(maximum, scores.max(-1).values)
            scale = torch.exp(maximum - new_max)
            probabilities = torch.exp(scores - new_max[..., None])
            numerator = numerator * scale[..., None] + probabilities @ value
            denominator = denominator * scale + probabilities.sum(-1)
            maximum = new_max
        output[:, :, qs:qs + query_block] = (numerator / denominator[..., None]).to(q.dtype)
    return output
