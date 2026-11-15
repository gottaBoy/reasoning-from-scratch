"""P1-06: Transformer Block, SwiGLU, LM output and backward acceptance."""

from datetime import datetime, timezone
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from learning.local_practice import _implementation, _resolve_device


class _Harness(nn.Module):
    """Small no-cache LM that wires the candidate Block into a full gradient path."""

    def __init__(self, module, config):
        super().__init__()
        self.cfg = config
        self.embedding = nn.Embedding(config.vocab, config.width)
        self.blocks = nn.ModuleList(module.Block(config) for _ in range(config.layers))
        self.norm = module.RMSNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab, bias=False)

    def forward(self, ids):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x, 0)
        return self.head(self.norm(x))


def _make_config(module):
    return module.Config(
        vocab=32, width=24, hidden=48, heads=4, kv_heads=2,
        head_dim=6, layers=2, context=8,
    )


def evaluate_block(device="cpu", candidate=None):
    device = _resolve_device(device)
    module, role, source = _implementation(
        candidate, exports=("Config", "RMSNorm", "FFN", "Block"),
        reference_module="learning.mini_transformer.model",
    )
    cases = []

    def run(name, check):
        try:
            evidence = check() or {}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cases.append({"name": name, "status": "passed", **evidence})
        except Exception as error:
            cases.append({"name": name, "status": "failed",
                          "error": f"{type(error).__name__}: {error}"})

    config = _make_config(module)

    def ffn_formula():
        torch.manual_seed(20261116)
        ffn = module.FFN(config).to(device)
        x = torch.randn(2, 3, config.width, device=device)
        actual = ffn(x)
        expected = ffn.down(F.silu(ffn.gate(x)) * ffn.up(x))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        assert tuple(actual.shape) == tuple(x.shape)
        return {"shape": list(actual.shape), "three_linear_swiglu": True}

    run("ffn_swiglu_formula", ffn_formula)

    def residual_chain():
        class Constant(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, x, *args):
                return torch.full_like(x, self.value)

        block = module.Block(config).to(device)
        block.attention = Constant(2.0)
        block.ffn = Constant(3.0)
        x = torch.full((1, 2, config.width), 10.0, device=device)
        actual = block(x, 0)
        expected = torch.full_like(x, 15.0)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        return {"x": 10, "attention_update": 2, "ffn_update": 3,
                "second_residual_uses_first_output": True}

    run("block_two_residual_chain", residual_chain)

    def ffn_backward():
        torch.manual_seed(20261116)
        ffn = module.FFN(config).to(device)
        x = torch.randn(2, 3, config.width, device=device).requires_grad_()
        ffn(x).square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in ffn.parameters())
        return {"finite_input_and_ffn_gradients": True}

    run("ffn_backward", ffn_backward)

    def model_logits():
        torch.manual_seed(20261116)
        model = _Harness(module, config).to(device)
        ids = torch.tensor([[1, 4, 2, 7]], device=device)
        logits = model(ids)
        assert tuple(logits.shape) == (1, 4, config.vocab)
        assert logits.requires_grad and torch.isfinite(logits).all()
        return {"logits_shape": list(logits.shape), "finite": True}

    run("lm_output_shape_and_finite", model_logits)

    def model_backward():
        torch.manual_seed(20261116)
        model = _Harness(module, config).to(device)
        ids = torch.tensor([[1, 4, 2, 7]], device=device)
        target = torch.tensor([[4, 2, 7, 3]], device=device)
        loss = F.cross_entropy(model(ids).reshape(-1, config.vocab), target.reshape(-1))
        assert torch.isfinite(loss)
        loss.backward()
        parameters = list(model.named_parameters())
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for _, parameter in parameters)
        return {"finite_loss": True, "parameters_with_grad": len(parameters)}

    run("lm_backward_all_parameters", model_backward)

    def eval_keeps_grad_contract():
        model = _Harness(module, config).to(device).eval()
        ids = torch.tensor([[1, 4]], device=device)
        logits = model(ids)
        assert logits.requires_grad and logits.grad_fn is not None
        return {"eval_mode": True, "autograd_still_enabled": True}

    run("eval_does_not_disable_autograd", eval_keeps_grad_contract)

    passed = sum(case["status"] == "passed" for case in cases)
    report = {
        "practice": "P1-06", "status": "passed" if passed == len(cases) else "failed",
        "implementation_role": role, "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "device": str(device), "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "passed": passed, "failed": len(cases) - passed, "cases": cases,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "FP32 Block/SwiGLU/LM output only: residual order, finite backward and "
            "autograd/eval behavior; no training convergence or optimizer benchmark."
        ),
    }
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device),
                      capability=list(torch.cuda.get_device_capability(device)))
    return report
