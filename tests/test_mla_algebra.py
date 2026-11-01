import pytest

from learning.mla_algebra import compare_paths


@pytest.mark.parametrize("tokens,heads,rank", [(1, 1, 1), (7, 4, 3), (13, 2, 5)])
def test_expanded_and_latent_attention_align(tokens, heads, rank):
    result = compare_paths(tokens=tokens, heads=heads, rank=rank)
    assert result["score_alignment"] == result["output_alignment"] == "passed"
    assert result["latent_shape"] == [tokens, rank]
    assert result["position_key_shape"] == [tokens, 2]
    assert result["expanded_key_shape"] == [heads, tokens, 6]
    assert result["expanded_value_shape"] == [heads, tokens, 3]
    assert result["expanded_elements_per_token"] == heads * (4 + 2 + 3)
    assert result["compact_elements_per_token"] == rank + 2
    assert result["output_max_error"] < 1e-6


def test_joint_latent_is_not_two_full_kv_arrays():
    result = compare_paths()
    assert result["expanded_elements_per_token"] == 36
    assert result["compact_elements_per_token"] == 5
    assert result["compact_bytes"] == 7 * 5 * 4
    assert result["expanded_bytes"] == 7 * 36 * 4


@pytest.mark.parametrize("parameters", [
    {"tokens": 0}, {"heads": -1}, {"rank": True}, {"rope_dim": 3},
])
def test_rejects_invalid_dimensions(parameters):
    with pytest.raises(ValueError):
        compare_paths(**parameters)
