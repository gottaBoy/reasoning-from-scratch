"""P1-01 acceptance runner: reference baseline or a trusted local learner module."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import torch


def _implementation(candidate):
    if candidate is None:
        from learning.mini_transformer import model as module
        role = "reference_baseline"
        source = Path(module.__file__).resolve()
    else:
        source = Path(candidate).resolve()
        if source.suffix != ".py" or not source.is_file():
            raise ValueError("Candidate must be an existing trusted local .py file.")
        spec = importlib.util.spec_from_file_location("local_practice_candidate", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        role = "candidate_file"
    if not callable(getattr(module, "RMSNorm", None)) or not callable(getattr(module, "rope", None)):
        raise ValueError("Module must export RMSNorm(width, eps) and rope(x, start, theta).")
    return module, role, source


def _rope_oracle(x, start, theta=10_000.0):
    x = x.detach().cpu().double()
    output = torch.empty_like(x)
    dim = x.shape[-1]
    positions = torch.arange(start, start + x.shape[-2], dtype=torch.float64)
    for pair in range(dim // 2):
        angle = positions * theta ** (-2 * pair / dim)
        left, right = x[..., pair], x[..., pair + dim // 2]
        output[..., pair] = left * angle.cos() - right * angle.sin()
        output[..., pair + dim // 2] = left * angle.sin() + right * angle.cos()
    return output


def _close(actual, expected, shape, device, atol=2e-6, rtol=1e-5):
    assert isinstance(actual, torch.Tensor), "Output must be a Tensor."
    assert tuple(actual.shape) == tuple(shape), "Output shape changed."
    assert actual.device == device, "Output device changed."
    assert actual.dtype == torch.float32, "This task requires FP32 output."
    assert torch.isfinite(actual).all(), "Output contains non-finite values."
    actual64, expected64 = actual.detach().cpu().double(), expected.detach().cpu().double()
    torch.testing.assert_close(actual64, expected64, atol=atol, rtol=rtol)
    return {"max_absolute_error": (actual64 - expected64).abs().max().item()}


def evaluate(device="cpu", candidate=None):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This practice supports cpu or cuda.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly requested but unavailable; no CPU fallback.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    module, role, source = _implementation(candidate)
    cases = []
    generator = torch.Generator().manual_seed(123)
    random = torch.randn(2, 3, 7, generator=generator).to(device)

    def run(name, operation):
        try:
            evidence = operation() or {}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cases.append({"name": name, "status": "passed", **evidence})
        except Exception as error:
            cases.append({"name": name, "status": "failed",
                          "error": f"{type(error).__name__}: {error}"})

    def norm_case(x):
        layer = module.RMSNorm(x.shape[-1], eps=1e-6).to(device)
        gamma = torch.linspace(0.5, 1.5, x.shape[-1], device=device)
        with torch.no_grad():
            layer.weight.copy_(gamma)
        oracle_x = x.detach().cpu().double()
        expected = oracle_x / (oracle_x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        expected *= gamma.cpu().double()
        return _close(layer(x), expected, x.shape, device)

    for name, x in (
        ("rms_random", random),
        ("rms_zero", torch.zeros_like(random)),
        ("rms_constant", torch.full_like(random, 2.0)),
        ("rms_large", random * 1e4),
        ("rms_tiny", random * 1e-4),
    ):
        run(name, lambda x=x: norm_case(x))

    def row_isolation():
        layer = module.RMSNorm(7, eps=1e-6).to(device)
        before = layer(random)
        altered = random.clone()
        altered[:, 1] *= 10
        after = layer(altered)
        torch.testing.assert_close(before[:, 0], after[:, 0], atol=0, rtol=0)
        return {"unchanged_other_token": True}

    run("rms_row_isolation", row_isolation)

    def backward():
        layer = module.RMSNorm(7, eps=1e-6).to(device)
        x = random.detach().clone().requires_grad_()
        layer(x).square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), "Input gradient invalid."
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all(), (
            "Gamma gradient invalid."
        )
        return {"finite_input_and_gamma_gradients": True}

    run("rms_backward", backward)
    q = torch.randn(2, 2, 7, 4, generator=generator).to(device)
    for start in (0, 5, 17):
        run(f"rope_offset_{start}", lambda start=start: _close(
            module.rope(q, start=start, theta=10_000.0),
            _rope_oracle(q, start), q.shape, device,
        ))

    def chunk_alignment():
        full = module.rope(q, start=0, theta=10_000.0)
        chunks = torch.cat((
            module.rope(q[:, :, :3], start=0, theta=10_000.0),
            module.rope(q[:, :, 3:], start=3, theta=10_000.0),
        ), dim=2)
        return _close(chunks, full, q.shape, device)

    run("rope_chunk_alignment", chunk_alignment)

    def norm_preservation():
        rotated = module.rope(q, start=5, theta=10_000.0)
        torch.testing.assert_close(rotated.norm(dim=-1), q.norm(dim=-1),
                                   atol=2e-6, rtol=1e-5)
        return {"norm_preserved": True}

    run("rope_norm_preservation", norm_preservation)

    def rejection(dim, start):
        try:
            module.rope(torch.zeros(1, 1, 2, dim, device=device), start=start, theta=10_000.0)
        except ValueError:
            return {"expected_value_error": True}
        raise AssertionError("Expected ValueError for invalid RoPE input.")

    run("rope_reject_odd_dimension", lambda: rejection(3, 0))
    run("rope_reject_negative_offset", lambda: rejection(4, -1))
    passed = sum(case["status"] == "passed" for case in cases)
    report = {
        "practice": "P1-01", "status": "passed" if passed == len(cases) else "failed",
        "implementation_role": role, "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "device": str(device), "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "passed": passed, "failed": len(cases) - passed, "cases": cases,
        "scope": (
            "FP32 correctness and boundary/gradient checks only; no timing, training quality, "
            "production kernels, or proof of independent authorship."
        ),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device),
                      capability=list(torch.cuda.get_device_capability(device)))
    return report


def write_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        output.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--candidate", type=Path, help="Trusted standalone learner .py module.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or Path("logs/practice") / (
        "p1-01-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    if output.exists():
        parser.error("Use a fresh report path; existing evidence will not be overwritten.")
    try:
        report = evaluate(args.device, args.candidate)
    except Exception as error:
        report = {"practice": "P1-01", "status": "error", "device": args.device,
                  "error": f"{type(error).__name__}: {error}"}
    write_report(report, output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Report:", output)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
