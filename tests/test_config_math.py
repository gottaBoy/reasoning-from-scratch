import pytest
import torch

from learning.config_math import inspect_config
from learning.day03_shapes import TINY_CONFIG
from reasoning_from_scratch.qwen3 import QWEN_CONFIG_06_B, Qwen3Model


def test_projection_estimates_match_real_parameters():
    model = Qwen3Model(TINY_CONFIG)
    report = inspect_config(TINY_CONFIG)
    att, ff = model.trf_blocks[0].att, model.trf_blocks[0].ff
    modules = {"W_query": att.W_query, "W_key": att.W_key, "W_value": att.W_value,
               "out_proj": att.out_proj, "gate_proj": ff.fc1, "up_proj": ff.fc2,
               "down_proj": ff.fc3}
    for name, module in modules.items():
        assert report["weight_shapes_out_in"][name] == list(module.weight.shape)
        assert report["parameters"][name] == module.weight.numel()


def test_qwen3_memory_and_sequence_scaling():
    report = inspect_config(QWEN_CONFIG_06_B, tokens=4096)
    assert report["weight_shapes_out_in"]["W_query"] == [2048, 1024]
    assert report["kv_bytes_all_layers"] == 448 * 2**20
    assert report["prefill_score_bytes_one_layer"] == 512 * 2**20
    doubled = inspect_config(QWEN_CONFIG_06_B, tokens=8192)
    assert doubled["kv_bytes_all_layers"] == 2 * report["kv_bytes_all_layers"]
    assert doubled["prefill_score_bytes_one_layer"] == 4 * report["prefill_score_bytes_one_layer"]
    assert doubled["projection_weight_bytes_all_layers"] == report["projection_weight_bytes_all_layers"]


def test_explicit_and_inferred_head_dimension_are_distinct():
    explicit = inspect_config(dict(TINY_CONFIG, emb_dim=48, n_heads=6,
                                   n_kv_groups=2, head_dim=16, hidden_dim=120))
    inferred = inspect_config(dict(TINY_CONFIG, emb_dim=48, n_heads=6,
                                   n_kv_groups=2, head_dim=None, hidden_dim=120))
    assert explicit["dimensions"]["D"] == 16
    assert inferred["dimensions"]["D"] == 8
    assert explicit["weight_shapes_out_in"]["W_query"] == [96, 48]
    assert inferred["weight_shapes_out_in"]["W_query"] == [48, 48]


@pytest.mark.parametrize("override", [
    {"emb_dim": 0}, {"n_heads": 3}, {"head_dim": -1}, {"n_kv_groups": True},
])
def test_invalid_configuration(override):
    with pytest.raises(ValueError):
        inspect_config(dict(TINY_CONFIG, **override))


def test_inspection_does_not_allocate_tensors(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Configuration arithmetic should not load tensors.")
    monkeypatch.setattr(torch, "empty", fail)
    assert inspect_config(QWEN_CONFIG_06_B)["sharing_ratio"] == 2
