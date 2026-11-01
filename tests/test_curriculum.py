from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from learning.mini_transformer import Cache, Config, Model, RMSNorm, rope
from learning.mini_vllm.engine import Engine, PrefixStore, Request
from learning.mini_vllm.paged_kv_cache import PagedPool
from learning.mini_vllm.scheduler import Job, simulate
from learning.online_attention import blocked_attention, online_softmax_chunks


@pytest.mark.parametrize("chunk", [1, 3, 20])
def test_online_softmax_stability(chunk):
    x = torch.tensor([1000., 1001., 998., -999., 1004.])
    y, _ = online_softmax_chunks(x, chunk)
    torch.testing.assert_close(y, x.softmax(0))


@pytest.mark.parametrize("causal,offset,t,keys", [(False, 0, 7, 9), (True, 0, 7, 7),
                                                (True, 6, 1, 7)])
def test_blocked_attention_against_sdpa(causal, offset, t, keys):
    torch.manual_seed(1)
    q = torch.randn(2, 2, t, 16)
    k, v = torch.randn(2, 2, keys, 16), torch.randn(2, 2, keys, 16)
    allowed = torch.arange(keys)[None] <= torch.arange(offset, offset + t)[:, None]
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed if causal else None)
    actual = blocked_attention(q, k, v, causal=causal, offset=offset)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_norm_rope_against_repository():
    from reasoning_from_scratch.qwen3 import RMSNorm as ReferenceNorm
    from reasoning_from_scratch.qwen3 import apply_rope, compute_rope_params
    x = torch.randn(2, 4, 7, 8)
    torch.testing.assert_close(RMSNorm(8)(x), ReferenceNorm(8)(x))
    cos, sin = compute_rope_params(8, context_length=32)
    torch.testing.assert_close(rope(x, 5), apply_rope(x, cos, sin, 5))


@pytest.mark.parametrize("kv", [1, 2, 4])
def test_model_causal_cache_gradients_and_sdpa(kv):
    torch.manual_seed(1)
    cfg = Config(kv_heads=kv)
    model = Model(cfg)
    ids = torch.randint(0, cfg.vocab, (2, 9))
    full = model(ids)
    alternate = ids.clone()
    alternate[:, 5:] = (alternate[:, 5:] + 1) % cfg.vocab
    torch.testing.assert_close(full[:, :5], model(alternate)[:, :5])
    cache = Cache(cfg.layers)
    with torch.no_grad():
        pref = model(ids[:, :4], cache)
        parts = [pref]
        for i in range(4, 9):
            parts.append(model(ids[:, i:i + 1], cache))
        torch.testing.assert_close(full, torch.cat(parts, 1), atol=1e-6, rtol=1e-5)
        assert cache.length == 9
    full.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    att = model.blocks[0].attention
    x = torch.randn(2, 5, cfg.width)
    q = rope(att.q(x).view(2, 5, 4, 8).transpose(1, 2))
    k = rope(att.k(x).view(2, 5, kv, 8).transpose(1, 2)).repeat_interleave(4 // kv, 1)
    v = att.v(x).view(2, 5, kv, 8).transpose(1, 2).repeat_interleave(4 // kv, 1)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = att.out(expected.transpose(1, 2).reshape(2, 5, -1))
    torch.testing.assert_close(att(x, 0), expected, atol=1e-6, rtol=1e-5)


def test_rejected_context_keeps_cache_and_requests_isolated():
    model = Model(replace(Config(), context=8))
    cache = Cache(2)
    model(torch.ones(1, 7, dtype=torch.long), cache)
    with pytest.raises(ValueError, match="Context"):
        model(torch.ones(1, 2, dtype=torch.long), cache)
    assert cache.length == 7
    with pytest.raises(ValueError):
        model(torch.ones(2, 1, dtype=torch.long), cache)
    assert model(torch.ones(1, 1, dtype=torch.long), cache).shape == (1, 1, 64)


def test_pool_ownership_noncontiguous_mapping_and_atomic_allocation():
    cfg = Config()
    pool = PagedPool(cfg, blocks=4, block_size=2)
    a = pool.reserve("a", 2)
    pool.reserve("b", 2)
    pool.release("a")
    c = pool.reserve("c", 6)
    assert c.table == [2, 3, 0]
    assert c.locate(4) == (0, 0)
    with pytest.raises(MemoryError):
        pool.reserve("no-fit", 1)
    assert "no-fit" not in pool.requests and len(pool.free) == 0
    with pytest.raises(ValueError, match="released"):
        a.get(0)
    k, v = torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8)
    for layer in range(cfg.layers):
        c.update(layer, k, v)
        got = c.get(layer)
        torch.testing.assert_close(got[0], k)
        torch.testing.assert_close(got[1], v)
    pool.release("c")
    pool.release("b")
    assert len(pool.free) == 4 and len(set(pool.free)) == 4


def test_engine_executes_real_bucket_batches_and_matches_greedy():
    torch.manual_seed(1)
    model = Model()
    reqs = [Request("a", [1, 2], 4), Request("b", [3, 4], 2),
            Request("c", [5, 6, 7], 3, arrival=1)]
    expected = {r.name: model.generate(torch.tensor([r.prompt]), r.max_tokens)[0].tolist()
                for r in reqs}
    engine = Engine(model, blocks=6, block_size=2)
    actual = engine.run(reqs)
    assert actual["outputs"] == expected
    assert any(row["batch_size"] == 2 for row in actual["batches"])
    assert len(engine.pool.free) == 6 and not engine.pool.requests


def test_engine_waits_for_capacity_and_cleans_up_after_failure(monkeypatch):
    model = Model()
    engine = Engine(model, blocks=2, block_size=2)
    jobs = [Request("a", [1, 2], 3), Request("b", [3, 4], 3)]
    assert len(engine.run(jobs)["outputs"]) == 2
    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")
    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        engine.run([Request("c", [1, 2], 2)])
    assert len(engine.pool.free) == 2


def test_prefix_reuse_clones_state_and_matches_full_continuation():
    model = Model()
    prefix = torch.tensor([[1, 2, 3]])
    store = PrefixStore(model)
    _, _, hit1 = store.prefill(prefix)
    cache, _, hit2 = store.prefill(prefix)
    assert not hit1 and hit2
    next_ids = torch.tensor([[4]])
    torch.testing.assert_close(model(next_ids, cache)[:, -1],
                               model(torch.cat((prefix, next_ids), 1))[:, -1],
                               atol=1e-6, rtol=1e-5)
    fresh, _, hit3 = store.prefill(prefix)
    assert hit3 and fresh.length == 3 and cache.length == 4


def test_scheduler_admission_completion_and_round_budget():
    jobs = [Job("a", 1), Job("b", 5), Job("c", 1, 1), Job("d", 1, 3)]
    continuous = simulate(jobs)
    static = simulate(jobs, mode="static")
    assert continuous["first_round"]["c"] < static["first_round"]["c"]
    for mode in ("continuous", "static", "sequential"):
        result = simulate(jobs, mode=mode, token_budget=1)
        assert all(len(row["executed"]) <= 1 for row in result["trace"])
        assert len(result["finish_round"]) == len(jobs)
    assert simulate([])["rounds"] == 0
