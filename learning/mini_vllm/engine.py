"""Actual tiny-model execution; bucket batches by equal sequence length."""

from dataclasses import dataclass, field

import torch

from learning.mini_transformer import Cache
from .paged_kv_cache import PagedPool


class BatchCache:
    def __init__(self, states):
        if not states or len({s.length for s in states}) != 1:
            raise ValueError("Batch must have equal cache positions.")
        self.states = states

    @property
    def length(self):
        return self.states[0].length

    def get(self, layer):
        entries = [s.get(layer) for s in self.states]
        if entries[0] is None:
            if any(entry is not None for entry in entries):
                raise ValueError("Mixed empty caches.")
            return None
        return tuple(torch.cat([entry[i] for entry in entries], 0) for i in range(2))

    def update(self, layer, keys, values):
        for i, state in enumerate(self.states):
            state.update(layer, keys[i:i + 1], values[i:i + 1])


@dataclass
class Request:
    name: str
    prompt: list
    max_tokens: int
    arrival: int = 0
    output: list = field(default_factory=list)


class Engine:
    def __init__(self, model, slots=2, blocks=128, block_size=4):
        if slots < 1:
            raise ValueError("Positive request limit required.")
        self.model, self.slots = model.eval(), slots
        parameter = next(model.parameters())
        self.pool = PagedPool(model.cfg, blocks, block_size, parameter.device, parameter.dtype)

    @torch.inference_mode()
    def run(self, requests):
        if self.pool.requests:
            raise ValueError("Engine is already running.")
        if len({r.name for r in requests}) != len(requests):
            raise ValueError("Duplicate request ids.")
        capacities = {}
        for request in requests:
            capacity = len(request.prompt) + request.max_tokens - 1
            if (not request.prompt or request.max_tokens < 1 or request.output
                    or request.arrival < 0 or capacity > self.model.cfg.context
                    or any(not 0 <= token < self.model.cfg.vocab for token in request.prompt)):
                raise ValueError("Invalid or reused request.")
            if (capacity + self.pool.block_size - 1) // self.pool.block_size > len(self.pool.free):
                raise MemoryError("Request can never fit in pool.")
            capacities[request.name] = capacity
        pending = sorted(requests, key=lambda r: r.arrival)
        waiting, running, trace = [], [], []
        tick = 0
        device = next(self.model.parameters()).device
        try:
            while pending or waiting or running:
                while pending and pending[0].arrival <= tick:
                    waiting.append(pending.pop(0))
                while waiting and len(running) < self.slots:
                    request = waiting[0]
                    try:
                        state = self.pool.reserve(request.name, capacities[request.name])
                    except MemoryError:
                        break
                    waiting.pop(0)
                    running.append((request, state))
                buckets = {}
                for request, state in running:
                    ids = request.prompt if not request.output else request.output[-1:]
                    buckets.setdefault((state.length, len(ids)), []).append((request, state, ids))
                for (position, tokens), bucket in buckets.items():
                    states = [entry[1] for entry in bucket]
                    ids = torch.tensor([entry[2] for entry in bucket], device=device)
                    logits = self.model(ids, BatchCache(states))
                    next_ids = logits[:, -1].argmax(-1).tolist()
                    for (request, _, _), token in zip(bucket, next_ids):
                        request.output.append(token)
                    trace.append({"round": tick, "batch_size": len(bucket),
                                  "cache_position": position, "input_tokens_each": tokens})
                remaining = []
                for request, state in running:
                    if len(request.output) == request.max_tokens:
                        self.pool.release(request.name)
                    else:
                        remaining.append((request, state))
                running = remaining
                tick += 1
            return {"kind": "real tiny-model batched execution with gather-based paged cache",
                    "outputs": {r.name: r.output for r in requests}, "batches": trace}
        finally:
            for request_id in list(self.pool.requests):
                self.pool.release(request_id)


class PrefixStore:
    """Exact full-prefix snapshots; clone for isolation, no COW or memory saving claim."""

    def __init__(self, model):
        self.model = model
        self.entries = {}

    @torch.inference_mode()
    def prefill(self, ids):
        if ids.shape[0] != 1:
            raise ValueError("One prefix per entry.")
        key = tuple(ids[0].tolist())
        if key in self.entries:
            cache, logits = self.entries[key]
            return cache.clone(), logits.clone(), True
        cache = Cache(self.model.cfg.layers)
        logits = self.model(ids, cache)
        self.entries[key] = (cache.clone(), logits.clone())
        return cache, logits, False

    def clear(self):
        self.entries.clear()
