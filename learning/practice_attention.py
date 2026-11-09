"""P1-02: single-head causal attention baseline and independent CPU FP64 oracle."""

from datetime import datetime, timezone
import hashlib
import math

import torch
from torch.nn import functional as F

from learning.local_practice import _close, _implementation, _resolve_device


def causal_attention(q, k, v, start=0):
    """FP32 reference, not a fused kernel; q/k/v are already projected."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Use [B,1,T,D] tensors.")
    if (min(*q.shape, *k.shape, *v.shape) < 1 or q.shape[1] != 1
            or k.shape[1] != 1 or v.shape[1] != 1
            or q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3]
            or q.shape[-1] != k.shape[-1]):
        raise ValueError("Nonempty single-head compatible Q/K/V required.")
    if any(x.device != q.device or x.dtype != torch.float32 for x in (q, k, v)):
        raise ValueError("Use same-device FP32 inputs.")
    if not isinstance(start, int) or isinstance(start, bool) or not 0 <= start <= k.shape[2] - q.shape[2]:
        raise ValueError("Query positions must fit the visible key sequence.")
    score = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    query_positions = torch.arange(start, start + q.shape[2], device=q.device)
    key_positions = torch.arange(k.shape[2], device=q.device)
    blocked = key_positions[None, :] > query_positions[:, None]
    return score.masked_fill(blocked, -torch.inf).softmax(-1) @ v


def _row_oracle(q, k, v, start):
    """Slice visible keys per query instead of sharing the candidate's mask code."""
    q, k, v = (x.detach().cpu().double() for x in (q, k, v))
    result = torch.empty(q.shape[0], 1, q.shape[2], v.shape[-1], dtype=torch.float64)
    for batch in range(q.shape[0]):
        for row in range(q.shape[2]):
            visible = start + row + 1
            scores = k[batch, 0, :visible] @ q[batch, 0, row] / math.sqrt(q.shape[-1])
            result[batch, 0, row] = scores.softmax(0) @ v[batch, 0, :visible]
    return result


def evaluate_attention(device="cpu", candidate=None):
    device = _resolve_device(device)
    module, role, source = _implementation(
        candidate, exports=("causal_attention",), reference_module=__name__,
    )
    operation = module.causal_attention
    cases = []
    generator = torch.Generator().manual_seed(321)

    def sample(batch=2, queries=5, keys=5, dim=4, value_dim=3):
        return (
            torch.randn(batch, 1, queries, dim, generator=generator).to(device),
            torch.randn(batch, 1, keys, dim, generator=generator).to(device),
            torch.randn(batch, 1, keys, value_dim, generator=generator).to(device),
        )

    def run(name, check):
        try:
            evidence = check() or {}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cases.append({"name": name, "status": "passed", **evidence})
        except Exception as error:
            cases.append({"name": name, "status": "failed",
                          "error": f"{type(error).__name__}: {error}"})

    def alignment(q, k, v, start=0):
        return _close(
            operation(q, k, v, start=start), _row_oracle(q, k, v, start),
            (q.shape[0], 1, q.shape[2], v.shape[-1]), device,
        )

    # Select scores [0,ln(3)], so output is .25*10 + .75*20 = 17.5.
    hand_q = torch.tensor([[[[0., math.sqrt(2) * math.log(3)]]]], device=device)
    hand_k = torch.tensor([[[[1., 0.], [0., 1.]]]], device=device)
    hand_v = torch.tensor([[[[10.], [20.]]]], device=device)
    run("handcalculated_last_query", lambda: _close(
        operation(hand_q, hand_k, hand_v, start=1),
        torch.tensor([[[[17.5]]]]), (1, 1, 1, 1), device,
    ))
    for name, tensors, start in (
        ("causal_prefill", sample(), 0),
        ("single_position", sample(batch=1, queries=1, keys=1), 0),
        ("cached_single_query", sample(queries=1, keys=7), 6),
        ("cached_query_chunk", sample(queries=3, keys=7), 4),
    ):
        run(name, lambda tensors=tensors, start=start: alignment(*tensors, start=start))
    q, k, v = sample()

    def future_invariance():
        before = operation(q, k, v)
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[:, :, 3:] = 100
        changed_v[:, :, 3:] = -100
        after = operation(q, changed_k, changed_v)
        torch.testing.assert_close(after[:, :, :3], before[:, :, :3], atol=2e-6, rtol=1e-5)
        return {"earlier_queries_cannot_read_future": True}

    run("causal_future_invariance", future_invariance)

    def scale_and_zero():
        zero_q = torch.zeros_like(q)
        expected = _row_oracle(zero_q, k, v, 0)
        return _close(operation(zero_q, k, v), expected,
                      (q.shape[0], 1, q.shape[2], v.shape[-1]), device)

    run("zero_queries_uniform_over_visible_keys", scale_and_zero)

    def no_input_mutation():
        snapshots = [x.clone() for x in (q, k, v)]
        operation(q, k, v)
        for x, old in zip((q, k, v), snapshots):
            torch.testing.assert_close(x, old, atol=0, rtol=0)
        return {"q_k_v_unchanged": True}

    run("inputs_unchanged", no_input_mutation)

    def backward():
        inputs = [x.clone().requires_grad_() for x in (q, k, v)]
        output = operation(*inputs)
        output.square().mean().backward()
        for x in inputs:
            assert x.grad is not None and torch.isfinite(x.grad).all(), "Invalid Q/K/V gradient."
        references = [x.clone().requires_grad_() for x in (q, k, v)]
        expected = F.scaled_dot_product_attention(
            *references, dropout_p=0.0, is_causal=True,
        )
        expected.square().mean().backward()
        for actual, reference in zip(inputs, references):
            torch.testing.assert_close(actual.grad, reference.grad, atol=2e-6, rtol=1e-4)
        return {"finite_gradients_and_sdpa_gradient_alignment": True}

    run("q_k_v_backward", backward)

    def rejection(q, k, v, start):
        try:
            operation(q, k, v, start=start)
        except ValueError:
            return {"expected_value_error": True}
        raise AssertionError("Expected ValueError for unsupported input.")

    for name, invalid_q, invalid_k, invalid_v, start in (
        ("reject_negative_start", q, k, v, -1),
        ("reject_query_positions_past_keys", q, k, v, 1),
        ("reject_mismatched_key_value_lengths", q, k, v[:, :, :-1], 0),
        ("reject_mismatched_q_k_features", q, k[..., :-1], v, 0),
        ("reject_empty_key_sequence", q[:, :, :1], k[:, :, :0], v[:, :, :0], 0),
    ):
        run(name, lambda a=invalid_q, b=invalid_k, c=invalid_v, s=start: rejection(a, b, c, s))

    passed = sum(case["status"] == "passed" for case in cases)
    report = {
        "practice": "P1-02", "status": "passed" if passed == len(cases) else "failed",
        "implementation_role": role, "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "device": str(device), "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "passed": passed, "failed": len(cases) - passed, "cases": cases,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "FP32 single-head core only: Q/K/V already projected; no RoPE, cache allocation, "
            "output projection, generation, dropout, GPU timing or authorship claim."
        ),
    }
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device),
                      capability=list(torch.cuda.get_device_capability(device)))
    return report
