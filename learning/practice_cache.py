"""P1-05: request-scoped KV Cache growth and isolation acceptance."""

from datetime import datetime, timezone
import hashlib

import torch

from learning.local_practice import _implementation, _resolve_device


def _make_model(module, device):
    config = module.Config(
        vocab=32, width=24, hidden=48, heads=4, kv_heads=2,
        head_dim=6, layers=2, context=8,
    )
    torch.manual_seed(20261115)
    return module.Model(config).to(device).eval()


def _run_chunks(model, ids, cache, chunks):
    outputs = []
    lengths = []
    offset = 0
    with torch.inference_mode():
        for size in chunks:
            outputs.append(model(ids[:, offset:offset + size], cache))
            offset += size
            lengths.append(cache.length)
    if offset != ids.shape[1]:
        raise AssertionError("Chunk sizes must cover the complete input.")
    return torch.cat(outputs, dim=1), lengths


def evaluate_cache(device="cpu", candidate=None):
    device = _resolve_device(device)
    module, role, source = _implementation(
        candidate, exports=("Cache", "Config", "Model"),
        reference_module="learning.mini_transformer.model",
    )
    cases = []
    ids = torch.tensor([[3, 7, 2, 9, 4, 1, 8, 6]], device=device)

    def run(name, check):
        try:
            evidence = check() or {}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cases.append({"name": name, "status": "passed", **evidence})
        except Exception as error:
            cases.append({"name": name, "status": "failed",
                          "error": f"{type(error).__name__}: {error}"})

    def chunk_3_2_3():
        model = _make_model(module, device)
        with torch.inference_mode():
            full = model(ids)
        cached, lengths = _run_chunks(model, ids, module.Cache(model.cfg.layers), (3, 2, 3))
        error = (full - cached).abs().max().item()
        torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-5)
        assert lengths == [3, 5, 8]
        return {"split": [3, 2, 3], "cache_lengths": lengths,
                "max_absolute_error": error}

    run("full_cached_alignment_3_2_3", chunk_3_2_3)

    def chunk_5_1_1_1():
        model = _make_model(module, device)
        with torch.inference_mode():
            full = model(ids)
        cached, lengths = _run_chunks(model, ids, module.Cache(model.cfg.layers), (5, 1, 1, 1))
        error = (full - cached).abs().max().item()
        torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-5)
        assert lengths == [5, 6, 7, 8]
        return {"split": [5, 1, 1, 1], "cache_lengths": lengths,
                "max_absolute_error": error}

    run("full_cached_alignment_5_1_1_1", chunk_5_1_1_1)

    def layer_lengths():
        model = _make_model(module, device)
        cache = module.Cache(model.cfg.layers)
        _run_chunks(model, ids, cache, (3, 2, 3))
        entries = [cache.get(layer) for layer in range(model.cfg.layers)]
        assert all(entry is not None for entry in entries)
        assert all(entry[0].shape[0] == 1 and entry[0].shape[2] == 8 for entry in entries)
        assert all(entry[1].shape == entry[0].shape for entry in entries)
        return {"layers": model.cfg.layers, "final_cache_length": cache.length}

    run("all_layers_share_final_length", layer_lengths)

    def request_isolation():
        model = _make_model(module, device)
        ids_a = ids[:, :5]
        ids_b = torch.tensor([[11, 5, 10, 12]], device=device)
        cache_a, cache_b = module.Cache(model.cfg.layers), module.Cache(model.cfg.layers)
        with torch.inference_mode():
            full_a, full_b = model(ids_a), model(ids_b)
        cached_a, lengths_a = _run_chunks(model, ids_a, cache_a, (2, 3))
        cached_b, lengths_b = _run_chunks(model, ids_b, cache_b, (1, 3))
        torch.testing.assert_close(full_a, cached_a, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(full_b, cached_b, atol=1e-5, rtol=1e-5)
        assert cache_a.length == 5 and cache_b.length == 4
        return {"request_a_lengths": lengths_a, "request_b_lengths": lengths_b,
                "independent": True}

    run("independent_request_caches", request_isolation)

    def reject_context_without_mutation():
        model = _make_model(module, device)
        cache = module.Cache(model.cfg.layers)
        with torch.inference_mode():
            model(ids[:, :3], cache)
        before = [(entry[0].clone(), entry[1].clone()) for entry in cache.entries]
        try:
            with torch.inference_mode():
                model(torch.zeros(1, 6, dtype=torch.long, device=device), cache)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected context rejection.")
        assert cache.length == 3
        for actual, expected in zip(cache.entries, before):
            torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
            torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
        return {"length_before": 3, "length_after": cache.length, "unchanged": True}

    run("reject_context_before_cache_mutation", reject_context_without_mutation)

    def reject_batch_mismatch():
        model = _make_model(module, device)
        cache = module.Cache(model.cfg.layers)
        with torch.inference_mode():
            model(ids[:, :3], cache)
        try:
            with torch.inference_mode():
                model(torch.zeros(2, 1, dtype=torch.long, device=device), cache)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected batch mismatch rejection.")
        assert cache.length == 3
        return {"cache_length_after_rejection": cache.length}

    run("reject_inconsistent_batch", reject_batch_mismatch)

    def causal_prefix():
        model = _make_model(module, device)
        altered = ids.clone()
        altered[:, 4:] = (altered[:, 4:] + 1) % model.cfg.vocab
        with torch.inference_mode():
            first = model(ids)
            second = model(altered)
        torch.testing.assert_close(first[:, :4], second[:, :4], atol=1e-5, rtol=1e-5)
        return {"prefix_tokens": 4, "future_changes_do_not_rewrite_prefix": True}

    run("causal_prefix_unchanged", causal_prefix)

    passed = sum(case["status"] == "passed" for case in cases)
    report = {
        "practice": "P1-05", "status": "passed" if passed == len(cases) else "failed",
        "implementation_role": role, "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "device": str(device), "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "passed": passed, "failed": len(cases) - passed, "cases": cases,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Request-scoped FP32 KV Cache only: full/cached alignment, multi-step growth, "
            "request isolation, causal prefix and pre-mutation context rejection."
        ),
    }
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device),
                      capability=list(torch.cuda.get_device_capability(device)))
    return report
