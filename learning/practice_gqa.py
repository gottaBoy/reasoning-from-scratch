"""P1-04: GQA/MQA compact KV head expansion baseline and acceptance."""

from datetime import datetime, timezone
import hashlib

import torch

from learning.local_practice import _implementation, _resolve_device


def expand_kv_heads(kv, query_heads):
    """Reference: [B,H_kv,T,D] -> [B,H_q,T,D]."""
    if kv.ndim != 4:
        raise ValueError("Expected [B,H_kv,T,D].")
    if min(*kv.shape) < 1:
        raise ValueError("Nonempty [B,H_kv,T,D] input required.")
    if not isinstance(query_heads, int) or isinstance(query_heads, bool) or query_heads < 1:
        raise ValueError("Positive integer query_heads required.")
    if kv.dtype != torch.float32:
        raise ValueError("This task requires FP32 input.")
    if query_heads % kv.shape[1] != 0:
        raise ValueError("query_heads must be divisible by kv_heads.")
    return kv.repeat_interleave(query_heads // kv.shape[1], dim=1)


def evaluate_gqa(device="cpu", candidate=None):
    device = _resolve_device(device)
    module, role, source = _implementation(
        candidate, exports=("expand_kv_heads",), reference_module=__name__,
    )
    cases = []
    generator = torch.Generator().manual_seed(111)

    def run(name, check):
        try:
            evidence = check() or {}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cases.append({"name": name, "status": "passed", **evidence})
        except Exception as error:
            cases.append({"name": name, "status": "failed",
                          "error": f"{type(error).__name__}: {error}"})

    def identity_mha():
        kv = torch.randn(2, 4, 3, 5, generator=generator).to(device)
        actual = module.expand_kv_heads(kv, 4)
        torch.testing.assert_close(actual, kv)
        assert tuple(actual.shape) == (2, 4, 3, 5)
        return {"structure": "MHA", "group_size": 1}

    run("mha_identity", identity_mha)

    def gqa_mapping():
        kv = torch.tensor([[[[0., 1.], [10., 11.]],
                            [[2., 3.], [12., 13.]]]], device=device)
        actual = module.expand_kv_heads(kv, 4)
        expected = torch.tensor([[[[0., 1.], [10., 11.]],
                                  [[0., 1.], [10., 11.]],
                                  [[2., 3.], [12., 13.]],
                                  [[2., 3.], [12., 13.]]]], device=device)
        torch.testing.assert_close(actual, expected)
        return {"structure": "GQA", "group_size": 2, "mapping": "Q0/Q1->KV0; Q2/Q3->KV1"}

    run("gqa_group_mapping", gqa_mapping)

    def mqa_mapping():
        kv = torch.tensor([[[[5., 6.], [15., 16.]]]], device=device)
        actual = module.expand_kv_heads(kv, 4)
        expected = kv.repeat(1, 4, 1, 1)
        torch.testing.assert_close(actual, expected)
        return {"structure": "MQA", "group_size": 4}

    run("mqa_group_mapping", mqa_mapping)

    def nonsquare_shape():
        kv = torch.randn(3, 2, 5, 7, generator=generator).to(device)
        actual = module.expand_kv_heads(kv, 6)
        assert tuple(actual.shape) == (3, 6, 5, 7)
        for query_head in range(6):
            torch.testing.assert_close(actual[:, query_head], kv[:, query_head // 3])
        return {"shape": [3, 2, 5, 7], "expanded_shape": [3, 6, 5, 7]}

    run("nonsquare_shape_mapping", nonsquare_shape)

    def compact_input_unchanged():
        kv = torch.randn(2, 2, 4, 3, generator=generator).to(device)
        before = kv.clone()
        _ = module.expand_kv_heads(kv, 4)
        torch.testing.assert_close(kv, before, atol=0, rtol=0)
        return {"compact_kv_unchanged": True}

    run("compact_input_unchanged", compact_input_unchanged)

    def expanded_is_independent():
        kv = torch.randn(1, 2, 2, 2, generator=generator).to(device)
        expanded = module.expand_kv_heads(kv, 4)
        expanded[:, 0] = 99
        assert not torch.equal(expanded[:, 2], expanded[:, 0])
        return {"materialized_expansion_is_independent": True}

    run("expanded_copy_is_independent", expanded_is_independent)

    def backward():
        kv = torch.randn(2, 2, 3, 4, generator=generator, device="cpu").to(device)
        kv.requires_grad_()
        module.expand_kv_heads(kv, 4).square().mean().backward()
        assert kv.grad is not None and torch.isfinite(kv.grad).all()
        return {"finite_compact_kv_gradient": True}

    run("compact_kv_backward", backward)

    def reject_not_divisible():
        try:
            module.expand_kv_heads(torch.zeros(1, 2, 2, 3, device=device), 3)
        except ValueError:
            return {"expected_value_error": True}
        raise AssertionError("Expected divisibility rejection.")

    run("reject_not_divisible", reject_not_divisible)

    def reject_query_less():
        try:
            module.expand_kv_heads(torch.zeros(1, 2, 2, 3, device=device), 1)
        except ValueError:
            return {"expected_value_error": True}
        raise AssertionError("Expected query-head rejection.")

    run("reject_query_heads_less_than_kv", reject_query_less)

    def reject_bad_rank():
        try:
            module.expand_kv_heads(torch.zeros(1, 2, 3, device=device), 4)
        except ValueError:
            return {"expected_value_error": True}
        raise AssertionError("Expected rank rejection.")

    run("reject_bad_rank", reject_bad_rank)

    passed = sum(case["status"] == "passed" for case in cases)
    report = {
        "practice": "P1-04", "status": "passed" if passed == len(cases) else "failed",
        "implementation_role": role, "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "device": str(device), "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "passed": passed, "failed": len(cases) - passed, "cases": cases,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "FP32 compact KV head expansion only: MHA/GQA/MQA mapping; no projection, "
            "attention scores, RoPE, cache allocation or performance."
        ),
    }
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device),
                      capability=list(torch.cuda.get_device_capability(device)))
    return report
