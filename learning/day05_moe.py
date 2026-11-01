"""Small CPU teaching MoE for routing and expert-load experiments."""

import torch
from torch import nn

from reasoning_from_scratch.qwen3 import FeedForward


class ToyMoE(nn.Module):
    def __init__(self, hidden=8, intermediate=16, num_experts=4, top_k=2):
        super().__init__()
        if hidden < 1 or intermediate < 1 or not 1 <= top_k <= num_experts:
            raise ValueError("Use positive widths and 1 <= top_k <= num_experts.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden, num_experts, bias=True)
        cfg = {"emb_dim": hidden, "hidden_dim": intermediate, "dtype": torch.float32}
        self.experts = nn.ModuleList(
            FeedForward(cfg) for _ in range(num_experts)
        )

    def forward(self, x):
        if x.ndim != 2 or x.shape[-1] != self.router.in_features or x.shape[0] < 1:
            raise ValueError("Expected nonempty [tokens, hidden] input.")
        logits = self.router(x)
        selected_logits, indices = logits.topk(self.top_k, dim=-1)
        weights = selected_logits.softmax(dim=-1)
        output = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(indices == expert_id)
            if token_ids.numel():
                contribution = expert(x[token_ids]) * weights[token_ids, slots, None]
                output = output.index_add(0, token_ids, contribution)
        return output, {
            "router_logits": logits,
            "expert_ids": indices,
            "weights": weights,
            "counts": torch.bincount(indices.reshape(-1), minlength=self.num_experts),
        }
