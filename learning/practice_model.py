"""P1-07: same-workload full/cached model baseline acceptance."""

from datetime import datetime, timezone
import hashlib

import torch

from learning.bench import measure
from learning.local_practice import _implementation, _resolve_device


_SEED = 20261116
_CONFIG = {
    "vocab": 32, "width": 24, "hidden": 48, "heads": 4, "kv_heads": 2,
    "head_dim": 6, "layers": 2, "context": 16, "theta": 10_000.0,
}


def _make_model(module, device):
    config = module.Config(**_CONFIG)
    torch.manual_seed(_SEED)
    return module.Model(config).to(device).eval(), config


def _cached_forward(model, module, ids, config):
    cache = module.Cache(config.layers)
    outputs = [model(ids[:, :5], cache)]
    for index in range(5, ids.shape[1]):
        outputs.append(model(ids[:, index:index + 1], cache))
    return torch.cat(outputs, dim=1), cache


def evaluate_model(device="cpu", candidate=None):
    device = _resolve_device(device)
    module, role, source = _implementation(
        candidate, exports=("Config", "Model", "Cache"),
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

    def full_cached_alignment():
        model, config = _make_model(module, device)
        with torch.inference_mode():
            full = model(ids)
            cached, cache = _cached_forward(model, module, ids, config)
        error = (full - cached).abs().max().item()
        torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-5)
        assert tuple(full.shape) == (ids.shape[0], ids.shape[1], config.vocab)
        assert full.device == cached.device == device
        assert full.dtype == cached.dtype == torch.float32
        assert cache.length == ids.shape[1]
        return {"logits_shape": list(full.shape), "cache_length": cache.length,
                "max_absolute_error": error}

    run("full_cached_logits_alignment", full_cached_alignment)

    def causal_prefix():
        model, _ = _make_model(module, device)
        altered = ids.clone()
        altered[:, 5:] = (altered[:, 5:] + 1) % model.cfg.vocab
        with torch.inference_mode():
            original = model(ids)
            changed = model(altered)
        torch.testing.assert_close(original[:, :5], changed[:, :5], atol=1e-5, rtol=1e-5)
        return {"prefix_tokens": 5, "future_does_not_change_prefix": True}

    run("causal_prefix_stability", causal_prefix)

    def same_workload_timing():
        model, config = _make_model(module, device)

        def full_workload():
            model(ids)

        def cached_workload():
            _cached_forward(model, module, ids, config)

        with torch.inference_mode():
            full_timing = measure(full_workload, device, warmup=1, repeats=3)
            cached_timing = measure(cached_workload, device, warmup=1, repeats=3)
        assert full_timing["repeats"] == cached_timing["repeats"] == 3
        return {
            "warmup": 1,
            "repeats": 3,
            "full": full_timing,
            "cached": cached_timing,
            "timing_scope": "synchronized end-to-end wall time; not a speedup claim",
        }

    run("same_workload_timing_metadata", same_workload_timing)

    def cache_shape():
        model, config = _make_model(module, device)
        with torch.inference_mode():
            _, cache = _cached_forward(model, module, ids, config)
        expected = (ids.shape[0], config.kv_heads, ids.shape[1], config.head_dim)
        for layer in range(config.layers):
            keys, values = cache.get(layer)
            assert tuple(keys.shape) == tuple(values.shape) == expected
            assert keys.device == values.device == device
            assert keys.dtype == values.dtype == torch.float32
        return {"compact_kv_shape": list(expected), "cache_length": cache.length,
                "checked_layers": config.layers}

    run("compact_cache_shape", cache_shape)

    passed = sum(case["status"] == "passed" for case in cases)
    report = {
        "practice": "P1-07", "status": "passed" if passed == len(cases) else "failed",
        "implementation_role": role, "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "device": str(device), "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "seed": _SEED, "configuration": dict(_CONFIG),
        "input_ids": ids.cpu().tolist(), "dtype": "torch.float32",
        "split": [5, 1, 1, 1], "atol": 1e-5, "rtol": 1e-5,
        "passed": passed, "failed": len(cases) - passed, "cases": cases,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Same-weight FP32 full/cached model baseline: complete logits alignment, "
            "causal prefix, compact cache shape and synchronized timing metadata."
        ),
    }
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device),
                      capability=list(torch.cuda.get_device_capability(device)))
    return report
