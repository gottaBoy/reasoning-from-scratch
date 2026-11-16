"""P1-07 personal model assembly; complete M1-M3 in forward.

Reuse the completed learner Block, Cache and RMSNorm. Attention inside Block
still uses the course reference. This is a small random model, not a checkpoint.
"""

from torch import nn

from learning.my_block import Block, Config
from learning.my_cache import Cache
from learning.my_ops import RMSNorm

__all__ = ["Cache", "Config", "Model"]


class Model(nn.Module):
    """Input token IDs [B,T] -> unnormalized logits [B,T,vocab]."""

    def __init__(self, config=Config()):
        super().__init__()
        self.cfg = config
        # Register weights once. Do not recreate modules inside forward.
        self.embedding = nn.Embedding(config.vocab, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab, bias=False)

    def forward(self, ids, cache=None):
        if ids.ndim != 2 or not ids.shape[0] or not ids.shape[1]:
            raise ValueError("Nonempty [batch, tokens] input required.")
        # Read position once, before any layer appends its new K/V.
        position = 0 if cache is None else cache.length
        if position + ids.shape[1] > self.cfg.context:
            raise ValueError("Context length exceeded before changing cache.")
        if cache is not None:
            for layer in range(self.cfg.layers):
                entry = cache.get(layer)
                if entry is not None and (entry[0].shape[0] != ids.shape[0]
                                          or entry[0].shape[2] != position):
                    raise ValueError("Inconsistent per-layer cache.")

        # TODO M1: embed ids, [B,T] -> [B,T,E], where E = self.cfg.width.
        x = self.embedding(ids)
        # TODO M2: loop over enumerate(self.blocks). Pass the same position
        # and request cache to each block, but its own layer index.
        for layer, block in enumerate(self.blocks):
            x = block(x, position, cache, layer)
        # TODO M3: apply the final norm, then head -> [B,T,self.cfg.vocab].
        # Return logits, not softmax probabilities or argmax token IDs.
        x = self.norm(x)
        x = self.head(x)
        return x
