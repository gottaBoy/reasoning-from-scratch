"""Verify the Day 10 stride lesson without importing any learner implementation."""

from pathlib import Path
import re

import pytest
import torch


def test_default_contiguous_stride_and_element_address():
    base = torch.arange(24, dtype=torch.float32)
    x = base.reshape(2, 3, 4)
    assert x.stride() == (12, 4, 1)
    assert x.element_size() == 4
    storage_index = x.storage_offset() + sum(i * s for i, s in zip((1, 2, 3), x.stride()))
    assert storage_index == 23
    assert x[1, 2, 3].item() == base[storage_index].item() == 23
    assert x[1, 2, 3].data_ptr() - base.untyped_storage().data_ptr() == 92


def test_split_and_transpose_preserve_storage_but_change_strides():
    base = torch.arange(24, dtype=torch.float32)
    split = base.view(2, 3, 2, 2)
    heads = split.transpose(1, 2)
    assert split.stride() == (12, 4, 2, 1)
    assert heads.shape == (2, 2, 3, 2)
    assert heads.stride() == (12, 2, 4, 1)
    assert heads[1, 1, 2, 0].item() == base[22].item() == 22
    assert heads.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    assert not heads.is_contiguous()


def test_slice_shares_storage_despite_different_first_element_pointer():
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    tail = x[:, 1:, :]
    assert tail.shape == (2, 2, 4)
    assert tail.stride() == (12, 4, 1)
    assert tail.storage_offset() == 4
    assert tail.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert tail.data_ptr() != x.data_ptr()
    assert tail.data_ptr() == x.data_ptr() + 4 * x.element_size()
    assert tail[0, 0, 0].item() == 4


def test_stepped_slice_can_have_nonunit_last_stride_and_still_allow_a_view():
    base = torch.arange(24, dtype=torch.float32)
    stepped = base[::2]
    assert stepped.stride() == (2,)
    assert not stepped.is_contiguous()
    viewed = stepped.view(3, 4)
    assert viewed.stride() == (8, 2)
    assert viewed.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    assert viewed[2, 3].item() == 22


def test_reshape_and_contiguous_do_not_fix_wrong_axis_semantics():
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    heads = x.view(2, 3, 2, 2).transpose(1, 2)
    merged = heads.transpose(1, 2).reshape(2, 3, 4)
    with pytest.raises(RuntimeError):
        heads.view(2, 3, 4)
    wrong = heads.reshape(2, 3, 4)
    packed = heads.contiguous()
    torch.testing.assert_close(merged, x)
    torch.testing.assert_close(packed, heads)
    assert wrong[0].tolist() == [[0, 1, 4, 5], [8, 9, 2, 3], [6, 7, 10, 11]]
    assert wrong.untyped_storage().data_ptr() != heads.untyped_storage().data_ptr()
    assert not torch.equal(packed.reshape(2, 3, 4), x)
    assert packed.contiguous() is packed


def test_expand_zero_stride_maps_multiple_indices_to_one_storage_element():
    base = torch.tensor([[10.], [20.]])
    expanded = base.expand(2, 3)
    assert expanded.stride() == (1, 0)
    assert expanded[0, 0].data_ptr() == expanded[0, 2].data_ptr()
    assert expanded.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()


def test_stride_lesson_python_blocks_run_in_document_order():
    root = Path(__file__).resolve().parents[1]
    document = (root / "docs/DAY10_MULTIHEAD.md").read_text(encoding="utf-8")
    section = document.split('<a id="tensor-stride"></a>', 1)[1].split(
        "## 3. Jupyter", 1,
    )[0]
    blocks = re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL)
    assert len(blocks) >= 7
    namespace = {}
    for block in blocks:
        exec(compile(block, "DAY10_MULTIHEAD.md#tensor-stride", "exec"), namespace)
    assert namespace["tail"].storage_offset() == 4
    assert namespace["heads"].stride() == (12, 2, 4, 1)
