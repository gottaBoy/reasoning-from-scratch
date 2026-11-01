"""FP32-first, decoder-only reference. Not a pretrained checkpoint loader."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    vocab: int = 64
    width: int = 32
    hidden: int = 64
    heads: int = 4
    kv_heads: int = 2
    head_dim: int = 8
    layers: int = 2
    context: int = 64
    theta: float = 10_000.0

    def __post_init__(self):
        if min(self.vocab, self.width, self.hidden, self.heads, self.kv_heads,
               self.head_dim, self.layers, self.context) < 1:
            raise ValueError("All dimensions must be positive.")
        if self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("heads must be divisible by kv_heads; head_dim must be even.")


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight).to(x.dtype)


def rope(x, start=0, theta=10_000.0):
    if start < 0 or x.shape[-1] % 2:
        raise ValueError("Nonnegative offset and even head dimension required.")
    dim = x.shape[-1]
    frequencies = theta ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
    positions = torch.arange(start, start + x.shape[-2], device=x.device)
    angles = positions[:, None] * frequencies[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return (x * angles.cos() + rotated * angles.sin()).to(x.dtype)


class Cache:
    """Per-request state; position is derived from K/V, never stored on the model."""

    def __init__(self, layers):
        self.entries = [None] * layers

    @property
    def length(self):
        entry = self.entries[0]
        return 0 if entry is None else entry[0].shape[2]

    def get(self, layer):
        return self.entries[layer]

    def update(self, layer, keys, values):
        self.entries[layer] = (keys, values)

    def clone(self):
        other = Cache(len(self.entries))
        other.entries = [None if entry is None else tuple(t.clone() for t in entry)
                         for entry in self.entries]
        return other


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q = nn.Linear(cfg.width, cfg.heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.width, cfg.kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.width, cfg.kv_heads * cfg.head_dim, bias=False)
        self.out = nn.Linear(cfg.heads * cfg.head_dim, cfg.width, bias=False)

    def forward(self, x, position, cache=None, layer=0):
        c = self.cfg
        b, t, _ = x.shape
        q = self.q(x).view(b, t, c.heads, c.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        q, k = rope(q, position, c.theta), rope(k, position, c.theta)
        old = None if cache is None else cache.get(layer)
        if old is not None:
            k, v = torch.cat((old[0], k), 2), torch.cat((old[1], v), 2)
        if cache is not None:
            cache.update(layer, k, v)
        expanded_k = k.repeat_interleave(c.heads // c.kv_heads, dim=1)
        expanded_v = v.repeat_interleave(c.heads // c.kv_heads, dim=1)
        key_positions = torch.arange(k.shape[2], device=x.device)
        query_positions = torch.arange(position, position + t, device=x.device)
        blocked = key_positions[None, :] > query_positions[:, None]
        scores = (q @ expanded_k.transpose(-1, -2)).float() / math.sqrt(c.head_dim)
        weights = scores.masked_fill(blocked, -torch.inf).softmax(-1).to(q.dtype)
        result = (weights @ expanded_v).transpose(1, 2).reshape(b, t, -1)
        return self.out(result)


class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.width, cfg.hidden, bias=False)
        self.up = nn.Linear(cfg.width, cfg.hidden, bias=False)
        self.down = nn.Linear(cfg.hidden, cfg.width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.width), RMSNorm(cfg.width)
        self.attention, self.ffn = Attention(cfg), FFN(cfg)

    def forward(self, x, position, cache=None, layer=0):
        x = x + self.attention(self.norm1(x), position, cache, layer)
        return x + self.ffn(self.norm2(x))


class Model(nn.Module):
    def __init__(self, cfg=Config()):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab, cfg.width)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.width)
        self.head = nn.Linear(cfg.width, cfg.vocab, bias=False)

    def forward(self, ids, cache=None):
        if ids.ndim != 2 or not ids.shape[0] or not ids.shape[1]:
            raise ValueError("Nonempty [batch, tokens] input required.")
        position = 0 if cache is None else cache.length
        if position + ids.shape[1] > self.cfg.context:
            raise ValueError("Context length exceeded before changing cache.")
        if cache is not None:
            for layer in range(self.cfg.layers):
                entry = cache.get(layer)
                if entry is not None and (entry[0].shape[0] != ids.shape[0]
                                          or entry[0].shape[2] != position):
                    raise ValueError("Inconsistent per-layer cache.")
        x = self.embedding(ids)
        for layer, block in enumerate(self.blocks):
            x = block(x, position, cache, layer)
        return self.head(self.norm(x))

    @torch.inference_mode()
    def generate(self, ids, tokens):
        if tokens < 0 or ids.shape[1] + max(tokens - 1, 0) > self.cfg.context:
            raise ValueError("Invalid generation budget.")
        if not tokens:
            return ids[:, :0].clone()
        cache = Cache(self.cfg.layers)
        scores = self(ids, cache)
        outputs = []
        for step in range(tokens):
            next_id = scores[:, -1].argmax(-1, keepdim=True)
            outputs.append(next_id)
            if step + 1 < tokens:
                scores = self(next_id, cache)
        return torch.cat(outputs, dim=1)
