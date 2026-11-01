import importlib.util
from pathlib import Path

import pytest
import torch

from reasoning_from_scratch.qwen3 import Qwen3Model


spec = importlib.util.spec_from_file_location(
    "day03_shapes", Path(__file__).resolve().parents[1] / "learning/day03_shapes.py"
)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


@pytest.mark.parametrize("groups", [1, 2, 4], ids=["MQA", "GQA", "MHA"])
@pytest.mark.parametrize("batch,length,qk_norm", [(1, 8, True), (2, 3, False)])
def test_shape_trace_and_causal_checks(groups, batch, length, qk_norm):
    cfg = dict(lab.TINY_CONFIG, n_kv_groups=groups, qk_norm=qk_norm)
    torch.manual_seed(123)
    model = Qwen3Model(cfg).eval()
    ids = torch.arange(batch * length).reshape(batch, length)
    original_weights = {name: value.clone() for name, value in model.state_dict().items()}
    report = lab.trace_model(model, ids)
    assert set(report["checks"].values()) == {"passed"}
    shapes = {row["name"]: row["shape"] for row in report["shapes"]}
    assert shapes["Q projection"] == [batch, length, cfg["n_heads"] * cfg["head_dim"]]
    assert shapes["K compact [B,KV,T,D]"] == [batch, groups, length, cfg["head_dim"]]
    assert shapes["K expanded"] == [batch, cfg["n_heads"], length, cfg["head_dim"]]
    assert shapes["logits"] == [batch, length, cfg["vocab_size"]]
    assert all(not module._forward_hooks for module in model.modules())
    assert report["input_ids"] != report["changed_input_ids"]
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(weight, original_weights[name], rtol=0, atol=0)


def test_hooks_removed_after_failed_forward():
    model = Qwen3Model(lab.TINY_CONFIG)
    ids = torch.full((1, 2), lab.TINY_CONFIG["vocab_size"], dtype=torch.long)
    with pytest.raises(IndexError):
        lab.trace_model(model, ids)
    assert all(not module._forward_hooks for module in model.modules())


@pytest.mark.parametrize("shape", [(8,), (1, 1)])
def test_trace_rejects_invalid_sequence_shape(shape):
    model = Qwen3Model(lab.TINY_CONFIG)
    with pytest.raises(ValueError, match="at least two tokens"):
        lab.trace_model(model, torch.zeros(shape, dtype=torch.long))
