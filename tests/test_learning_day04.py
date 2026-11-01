import pytest
import torch

from learning.day03_shapes import TINY_CONFIG
from learning.day04_cache import compare_cache
from reasoning_from_scratch.qwen3 import Qwen3Model


@pytest.mark.parametrize("groups", [1, 2, 4], ids=["MQA", "GQA", "MHA"])
@pytest.mark.parametrize("batch,length,split", [(1, 8, 7), (2, 5, 2)])
def test_cache_equivalence_and_growth(groups, batch, length, split):
    cfg = dict(TINY_CONFIG, n_kv_groups=groups)
    torch.manual_seed(123)
    model = Qwen3Model(cfg)
    ids = torch.arange(batch * length).reshape(batch, length)
    report = compare_cache(model, ids, split)
    assert report["logits_close"]
    assert report["outside_tolerance"] == 0
    assert report["last_decode_logits_shape"] == [batch, 1, cfg["vocab_size"]]
    assert report["phases"][0]["cache"][0]["K"] == [batch, groups, split, cfg["head_dim"]]
    assert report["phases"][-1]["cache"][0]["K"][2] == length
    assert model.current_pos == 0
    # A second run must not inherit the previous prompt's position or K/V.
    repeated = compare_cache(model, ids, split)
    assert repeated["max_absolute_error"] == report["max_absolute_error"]


@pytest.mark.parametrize("split", [0, 8])
def test_invalid_split(split):
    model = Qwen3Model(TINY_CONFIG)
    with pytest.raises(ValueError):
        compare_cache(model, torch.arange(8).unsqueeze(0), split)


def test_strict_comparison_detects_wrong_rope_offset(monkeypatch):
    torch.manual_seed(123)
    model = Qwen3Model(TINY_CONFIG)
    original = model.forward

    def incorrect_forward(ids, cache=None):
        logits = original(ids, cache=cache)
        if cache is not None:
            model.current_pos = 0
        return logits

    monkeypatch.setattr(model, "forward", incorrect_forward)
    with pytest.raises(AssertionError):
        compare_cache(model, torch.arange(8).unsqueeze(0), 7)
    assert model.current_pos == 0
