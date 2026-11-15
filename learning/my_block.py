"""P1-06 personal candidate: implement FFN and Block here.

Shape symbols follow the reference Config:
E = config.width (residual width), F = config.hidden (FFN width).
"""

from torch import nn

from learning.mini_transformer.model import Attention, Config, RMSNorm

__all__ = ["Config", "RMSNorm", "FFN", "Block"]


class FFN(nn.Module):
    """SwiGLU feed-forward path: E -> F -> E."""

    def __init__(self, config):
        super().__init__()
        # TODO F1: create three independent bias-free projections:
        # gate/up: [B,T,E] -> [B,T,F], down: [B,T,F] -> [B,T,E].
        # The gate and up weights must not be the same parameter.
        intermediate_size = config.hidden
        hidden_size = config.width
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)
        # raise NotImplementedError("Implement FFN.__init__.")

    def forward(self, x):
        # TODO F2: activate only the gate branch, then multiply both F-wide
        # branches elementwise before projecting back to residual width E.
        gate_output = self.gate(x)
        up_output = self.up(x)
        return self.down(nn.functional.silu(gate_output) * up_output)
        # raise NotImplementedError("Implement FFN.forward.")


class Block(nn.Module):
    """Pre-norm attention followed by pre-norm SwiGLU and two residuals."""

    def __init__(self, config):
        super().__init__()
        # TODO B1: create two independent pre-norms plus the Attention and
        # SwiGLU submodules. Both norms operate on the residual width E.
        self.norm1 = RMSNorm(config.width)
        self.norm2 = RMSNorm(config.width)
        self.attention = Attention(config)
        self.ffn = FFN(config)
        # raise NotImplementedError("Implement Block.__init__.")

    def forward(self, x, position, cache=None, layer=0):
        # TODO B2: first residual produces x1. The second residual must add
        # the FFN update to x1, not to the original x.
        x1 = x + self.attention(self.norm1(x), position, cache, layer)
        return x1 + self.ffn(self.norm2(x1))
        # raise NotImplementedError("Implement Block.forward.")
