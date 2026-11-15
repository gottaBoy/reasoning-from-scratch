"""P1-03: multi-head split/merge layout baseline and acceptance."""

from datetime import datetime, timezone
import hashlib

import torch

from learning.local_practice import _implementation, _resolve_device


def split_heads(x, heads, head_dim):
    """Reference: contiguous [B,T,H*D] -> [B,H,T,D]."""
    if x.ndim != 3:
        raise ValueError("Expected [B,T,H*D].")
    if (not isinstance(heads, int) or isinstance(heads, bool) or heads < 1
            or not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim < 1):
        raise ValueError("Positive integer heads and head_dim required.")
    if x.shape[-1] != heads * head_dim:
        raise ValueError("Last dimension must equal heads * head_dim.")
    if x.dtype != torch.float32:
        raise ValueError("This task requires FP32 input.")
    if not x.is_contiguous():
        raise ValueError("split_heads requires contiguous input.")
    return x.view(x.shape[0], x.shape[1], heads, head_dim).transpose(1, 2)


def merge_heads(heads):
    """Reference: [B,H,T,D] -> [B,T,H*D]."""
    if heads.ndim != 4:
        raise ValueError("Expected [B,H,T,D].")
    if min(*heads.shape) < 1:
        raise ValueError("Nonempty [B,H,T,D] input required.")
    if heads.dtype != torch.float32:
        raise ValueError("This task requires FP32 input.")
    return heads.transpose(1, 2).reshape(heads.shape[0], heads.shape[2], -1)


def evaluate_layout(device="cpu", candidate=None):
    device = _resolve_device(device)
    module, role, source = _implementation(
        candidate, exports=("split_heads", "merge_heads"), reference_module=__name__,
    )
    cases = []
    generator = torch.Generator().manual_seed(910)

    def run(name, check):
        try:
            evidence = check() or {}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cases.append({"name": name, "status": "passed", **evidence})
        except Exception as error:
            cases.append({"name": name, "status": "failed",
                          "error": f"{type(error).__name__}: {error}"})

    x = torch.arange(2 * 3 * 2 * 3, device=device, dtype=torch.float32).reshape(2, 3, 6)
    heads = 2
    head_dim = 3

    def split_order():
        actual = module.split_heads(x, heads, head_dim)
        expected = x.view(2, 3, 2, 3).transpose(1, 2)
        torch.testing.assert_close(actual, expected)
        assert tuple(actual.shape) == (2, 2, 3, 3)
        assert tuple(actual.stride()) == (18, 3, 6, 1)
        assert actual.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        return {"shape": list(actual.shape), "stride": list(actual.stride()),
                "shares_storage": True}

    run("split_order_and_stride", split_order)

    def label_order():
        labels = torch.tensor([[[0., 1., 2., 3., 4., 5.],
                                [10., 11., 12., 13., 14., 15.]]], device=device)
        actual = module.split_heads(labels, 2, 3)
        expected = torch.tensor([[[[0., 1., 2.], [10., 11., 12.]],
                                  [[3., 4., 5.], [13., 14., 15.]]]], device=device)
        torch.testing.assert_close(actual, expected)
        return {"head0_then_head1_per_position": True}

    run("split_label_order", label_order)

    def merge_roundtrip():
        split = module.split_heads(x, heads, head_dim)
        actual = module.merge_heads(split)
        torch.testing.assert_close(actual, x)
        assert tuple(actual.shape) == tuple(x.shape)
        return {"round_trip": True, "contiguous": actual.is_contiguous()}

    run("merge_round_trip", merge_roundtrip)

    def merge_label_order():
        labels = torch.tensor([[[[0., 1.], [10., 11.], [20., 21.]],
                                [[2., 3.], [12., 13.], [22., 23.]]]], device=device)
        actual = module.merge_heads(labels)
        expected = torch.tensor([[[0., 1., 2., 3.],
                                  [10., 11., 12., 13.],
                                  [20., 21., 22., 23.]]], device=device)
        torch.testing.assert_close(actual, expected)
        return {"same_position_heads_adjacent": True}

    run("merge_label_order", merge_label_order)

    def multi_shape():
        random = torch.randn(3, 5, 4 * 7, generator=generator).to(device)
        actual = module.merge_heads(module.split_heads(random, 4, 7))
        torch.testing.assert_close(actual, random)
        return {"shape": [3, 5, 4, 7]}

    run("non_square_shape_round_trip", multi_shape)

    def storage_and_layout():
        actual = module.split_heads(x, 2, 3)
        assert not actual.is_contiguous()
        assert actual.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        merged = module.merge_heads(actual)
        assert merged.is_contiguous()
        return {"split_noncontiguous": True, "merged_contiguous": True}

    run("storage_and_contiguity", storage_and_layout)

    def backward():
        input_x = x.detach().clone().requires_grad_()
        output = module.merge_heads(module.split_heads(input_x, 2, 3))
        output.square().mean().backward()
        assert input_x.grad is not None and torch.isfinite(input_x.grad).all()
        return {"finite_round_trip_gradient": True}

    run("round_trip_backward", backward)

    def reject_width():
        try:
            module.split_heads(torch.zeros(1, 2, 5, device=device), 2, 3)
        except ValueError:
            return {"expected_value_error": True}
        raise AssertionError("Expected width rejection.")

    run("reject_incompatible_width", reject_width)

    def reject_noncontiguous():
        try:
            module.split_heads(x.transpose(1, 2), 2, 3)
        except ValueError:
            return {"expected_value_error": True}
        raise AssertionError("Expected noncontiguous input rejection.")

    run("reject_noncontiguous_split_input", reject_noncontiguous)

    def reject_bad_merge():
        try:
            module.merge_heads(torch.zeros(1, 2, 0, 3, device=device))
        except ValueError:
            return {"expected_value_error": True}
        raise AssertionError("Expected empty input rejection.")

    run("reject_empty_merge_input", reject_bad_merge)

    passed = sum(case["status"] == "passed" for case in cases)
    report = {
        "practice": "P1-03", "status": "passed" if passed == len(cases) else "failed",
        "implementation_role": role, "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "device": str(device), "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "passed": passed, "failed": len(cases) - passed, "cases": cases,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "FP32 multi-head layout only: split/view, transpose and merge/reshape; "
            "no Linear projection, attention scores, RoPE, GQA, cache or timing."
        ),
    }
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device),
                      capability=list(torch.cuda.get_device_capability(device)))
    return report
