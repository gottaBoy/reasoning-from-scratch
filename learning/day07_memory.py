"""CPU drawer-and-directory lab using the existing teaching pool and tiny engine."""

import torch

from learning.mini_transformer import Config, Model
from learning.mini_vllm.engine import Engine, Request
from learning.mini_vllm.paged_kv_cache import PagedPool


def small_config():
    return Config(vocab=16, width=4, hidden=8, heads=2, kv_heads=1,
                  head_dim=2, layers=2, context=16)


def _payload(layer, tokens, base):
    keys = torch.arange(tokens * 2, dtype=torch.float32).reshape(1, 1, tokens, 2)
    keys = keys + base + 10 * layer
    return keys, keys + 1000


@torch.inference_mode()
def storage_lesson():
    """Store numeric markers, recycle pages, and retain snapshots for each scene."""
    cfg = small_config()
    pool = PagedPool(cfg, blocks=6, block_size=2)
    address = pool.storage.data_ptr()
    storage_bytes = pool.storage.numel() * pool.storage.element_size()
    events = []

    def snapshot(event):
        owners = {}
        requests = {}
        for name, state in pool.requests.items():
            requests[name] = {"table": list(state.table), "capacity": state.capacity,
                              "lengths": list(state.lengths)}
            for logical, physical in enumerate(state.table):
                owners[physical] = f"{name}:{logical}"
        assert len(owners) + len(pool.free) == pool.storage.shape[0]
        assert not set(owners).intersection(pool.free)
        assert pool.storage.data_ptr() == address
        events.append({
            "event": event, "owners": [owners.get(i, "free") for i in range(6)],
            "free": list(pool.free), "requests": requests,
            "pool_bytes": storage_bytes,
        })

    snapshot("initial")
    a = pool.reserve("A", 4)
    b = pool.reserve("B", 4)
    snapshot("reserve_A_B")
    for layer in range(cfg.layers):
        a.update(layer, *_payload(layer, 3, 100))
        b.update(layer, *_payload(layer, 2, 200))
    snapshot("prefill_A3_B2")
    for layer in range(cfg.layers):
        a.update(layer, *_payload(layer, 4, 100))
    snapshot("append_A_to4")

    b_physical = b.table[0]
    b_page = pool.storage[b_physical].clone()
    pool.release("B")
    torch.testing.assert_close(pool.storage[b_physical], b_page)
    try:
        b.get(0)
    except ValueError:
        stale_handle_rejected = True
    else:
        raise AssertionError("Released request handle must be invalid.")
    snapshot("release_B")

    c = pool.reserve("C", 6)
    assert c.table == [4, 5, 2]
    assert all(c.get(layer) is None for layer in range(cfg.layers))
    torch.testing.assert_close(pool.storage[b_physical], b_page)
    snapshot("reserve_C_empty")
    for layer in range(cfg.layers):
        expected = _payload(layer, 5, 300)
        c.update(layer, *expected)
        for actual, reference in zip(c.get(layer), expected):
            torch.testing.assert_close(actual, reference)
    snapshot("write_C5")
    position = 4
    physical, offset = c.locate(position)
    torch.testing.assert_close(
        pool.storage[physical, 0, 0, :, offset],
        _payload(0, 5, 300)[0][0, :, position],
    )
    retrieved_keys = c.get(0)[0].tolist()

    before_free = list(pool.free)
    before_owners = set(pool.requests)
    before_storage = pool.storage.clone()
    try:
        pool.reserve("D", 4)
    except MemoryError:
        oom_rejected = True
    else:
        raise AssertionError("Two requested blocks cannot fit into one free block.")
    assert list(pool.free) == before_free and set(pool.requests) == before_owners
    torch.testing.assert_close(pool.storage, before_storage)
    snapshot("reject_D_without_mutation")
    for name in list(pool.requests):
        pool.release(name)
    assert not pool.requests and len(pool.free) == 6
    snapshot("release_all")
    return {
        "scope": "CPU teaching pool; numeric KV markers, no GPU allocator or kernel timing",
        "storage_shape": list(pool.storage.shape),
        "storage_bytes": storage_bytes,
        "bytes_per_block": storage_bytes // 6,
        "events": events,
        "C_position": {"logical": position, "physical": physical, "offset": offset},
        "C_layer0_keys_in_logical_order": retrieved_keys,
        "released_handle_rejected": stale_handle_rejected,
        "release_does_not_zero_page": True,
        "oom_without_mutation": oom_rejected,
        "all_blocks_free": len(pool.free) == 6,
        "storage_address_unchanged": pool.storage.data_ptr() == address,
    }


@torch.inference_mode()
def engine_lesson():
    """Actually execute tiny-model prefills/decodes and compare independent greedy."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = Model(small_config()).eval()
    requests = [
        Request("A", [1, 2, 3], 3),
        Request("B", [4, 5, 6], 1),
        Request("C", [7, 8], 2, arrival=1),
    ]
    reference = {
        r.name: model.generate(torch.tensor([r.prompt]), r.max_tokens)[0].tolist()
        for r in requests
    }
    engine = Engine(model, slots=2, blocks=8, block_size=2)
    before = engine.pool.storage.data_ptr()
    report = engine.run(requests)
    assert report["outputs"] == reference
    assert not engine.pool.requests and len(engine.pool.free) == 8
    return {
        **report,
        "scope": "CPU random tiny model; not serving performance or a paged GPU kernel",
        "reserved_capacity": {r.name: len(r.prompt) + r.max_tokens - 1 for r in requests},
        "independent_greedy_alignment": "passed",
        "all_blocks_free": True,
        "storage_address_unchanged": engine.pool.storage.data_ptr() == before,
    }
