"""Keep language variants aligned without loading the learner's implementation."""

import ast
from pathlib import Path

import pytest
import torch

from learning import my_ops_template_zh as chinese


def _executable_structure(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False)


def test_chinese_template_matches_english_executable_structure():
    root = Path(__file__).resolve().parents[1] / "learning"
    assert _executable_structure(root / "my_ops_template_zh.py") == _executable_structure(
        root / "my_ops_template.py",
    )


@pytest.mark.parametrize("operation", ("norm", "rope", "attention"))
def test_chinese_template_is_intentionally_unfinished(operation):
    x = torch.ones(1, 1, 1, 2)
    with pytest.raises(NotImplementedError):
        if operation == "norm":
            chinese.RMSNorm(2)(x)
        elif operation == "rope":
            chinese.rope(x)
        else:
            chinese.causal_attention(x, x, x)


def test_chinese_template_preserves_input_validation():
    x = torch.ones(1, 1, 1, 2)
    with pytest.raises(ValueError, match="Positive width"):
        chinese.RMSNorm(0)
    with pytest.raises(ValueError, match="last dimension"):
        chinese.RMSNorm(3)(x)
    with pytest.raises(ValueError, match="even D"):
        chinese.rope(torch.ones(1, 1, 1, 3))
    with pytest.raises(ValueError, match="start position"):
        chinese.causal_attention(x, x, x, start=-1)
