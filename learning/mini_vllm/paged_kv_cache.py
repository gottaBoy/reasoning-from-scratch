"""Fixed-size physical blocks with per-request tables and explicit ownership."""

from collections import deque

import torch


class PagedPool:
    def __init__(self, cfg, blocks=64, block_size=4, device="cpu", dtype=torch.float32):
        if min(blocks, block_size) < 1:
            raise ValueError("Positive pool dimensions required.")
        self.cfg, self.block_size = cfg, block_size
        self.storage = torch.zeros(blocks, cfg.layers, 2, cfg.kv_heads, block_size,
                                   cfg.head_dim, device=device, dtype=dtype)
        self.free = deque(range(blocks))
        self.requests = {}

    def reserve(self, request_id, capacity):
        if request_id in self.requests or not 1 <= capacity <= self.cfg.context:
            raise ValueError("Unique request id and valid token capacity required.")
        needed = (capacity + self.block_size - 1) // self.block_size
        if needed > len(self.free):
            raise MemoryError("Not enough KV blocks; pool is unchanged.")
        table = [self.free.popleft() for _ in range(needed)]
        state = PagedRequest(self, request_id, table, capacity)
        self.requests[request_id] = state
        return state

    def release(self, request_id):
        state = self.requests.pop(request_id)
        state.active = False
        self.free.extend(state.table)


class PagedRequest:
    def __init__(self, pool, request_id, table, capacity):
        self.pool, self.request_id = pool, request_id
        self.table, self.capacity = table, capacity
        self.lengths = [0] * pool.cfg.layers
        self.active = True

    @property
    def length(self):
        self._check()
        return self.lengths[0]

    def _check(self):
        if not self.active:
            raise ValueError("Cache has been released; this handle is invalid.")

    def locate(self, token_position):
        self._check()
        if not 0 <= token_position < self.capacity:
            raise IndexError("Token position outside reserved capacity.")
        block, offset = divmod(token_position, self.pool.block_size)
        return self.table[block], offset

    def get(self, layer):
        self._check()
        size = self.lengths[layer]
        if not size:
            return None
        keys, values = [], []
        for start in range(0, size, self.pool.block_size):
            physical = self.table[start // self.pool.block_size]
            count = min(self.pool.block_size, size - start)
            keys.append(self.pool.storage[physical, layer, 0, :, :count])
            values.append(self.pool.storage[physical, layer, 1, :, :count])
        # Gathering is intentionally explicit: not a paged GPU Attention kernel.
        return torch.cat(keys, dim=1).unsqueeze(0), torch.cat(values, dim=1).unsqueeze(0)

    def update(self, layer, keys, values):
        self._check()
        expected = (1, self.pool.cfg.kv_heads, keys.shape[2], self.pool.cfg.head_dim)
        if keys.shape != values.shape or tuple(keys.shape) != expected:
            raise ValueError("Expected compact [1,KV,T,D] cache.")
        if not self.lengths[layer] <= keys.shape[2] <= self.capacity:
            raise ValueError("Cache must grow within its reservation.")
        if keys.device != self.pool.storage.device or keys.dtype != self.pool.storage.dtype:
            raise ValueError("Cache device/dtype mismatch.")
        for position in range(self.lengths[layer], keys.shape[2]):
            block, offset = self.locate(position)
            self.pool.storage[block, layer, 0, :, offset] = keys[0, :, position]
            self.pool.storage[block, layer, 1, :, offset] = values[0, :, position]
        self.lengths[layer] = keys.shape[2]
