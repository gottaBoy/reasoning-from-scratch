"""P1-05 personal candidate: implement request-scoped Cache here.

The reference Model is reused only to keep this exercise focused on cache
ownership, length tracking, append/update, and cloning.
"""

from learning.mini_transformer.model import Config, Model as _ReferenceModel

__all__ = ["Cache", "Config", "Model"]


class Cache:
    """Per-request K/V state used by the reference Model."""

    def __init__(self, layers):
        if not isinstance(layers, int) or isinstance(layers, bool) or layers < 1:
            raise ValueError("Positive integer layers required.")
        # Keep one independent slot for each Transformer layer.
        self.entries = [None] * layers

    @property
    def length(self):
        # Derive position from stored K/V rather than a second mutable counter.
        for entry in self.entries:
            if entry is not None:
                return entry[0].shape[2]
        return 0

    def get(self, layer):
        # Return the selected layer's K/V pair.
        if layer < 0 or layer >= len(self.entries):
            raise IndexError("Layer index out of range.")
        return self.entries[layer]

    def update(self, layer, keys, values):
        # Store the layer's K/V pair without copying or changing the tensors.
        if layer < 0 or layer >= len(self.entries):
            raise IndexError("Layer index out of range.")
        self.entries[layer] = (keys, values)

    def clone(self):
        # Clone tensors so later writes to one request cannot affect the other.
        new_cache = Cache(len(self.entries))
        for i, entry in enumerate(self.entries):
            if entry is not None:
                keys, values = entry
                new_cache.entries[i] = (keys.clone(), values.clone())
        return new_cache


class Model(_ReferenceModel):
    """Reuse model math while routing state through this candidate Cache."""
