"""P1-03 acceptance tests for split/merge layout."""

import json

import pytest
import torch

from learning.local_practice import main
from learning.practice_layout import evaluate_layout, merge_heads, split_heads


def test_layout_reference_passes_all_cpu_checks():
    report = evaluate_layout("cpu")
    assert report["practice"] == "P1-03"
    assert report["status"] == "passed"
    assert report["passed"] == 10 and report["failed"] == 0
    assert report["implementation_role"] == "reference_baseline"


def test_layout_candidate_path_is_loaded(tmp_path):
    path = tmp_path / "layout_ops.py"
    path.write_text("from learning.practice_layout import split_heads, merge_heads\n")
    report = evaluate_layout("cpu", path)
    assert report["status"] == "passed"
    assert report["passed"] == 10
    assert report["implementation_role"] == "candidate_file"


def test_layout_wrong_merge_order_is_detected(tmp_path):
    path = tmp_path / "wrong_layout.py"
    path.write_text(
        "from learning.practice_layout import split_heads\n"
        "def merge_heads(heads):\n"
        "    return heads.reshape(heads.shape[0], heads.shape[2], -1)\n"
    )
    report = evaluate_layout("cpu", path)
    cases = {case["name"]: case for case in report["cases"]}
    assert report["status"] == "failed"
    assert cases["merge_round_trip"]["status"] == "failed"
    assert cases["merge_label_order"]["status"] == "failed"


def test_layout_cli_reports_p1_03(tmp_path):
    output = tmp_path / "layout.json"
    assert main(["--task", "layout", "--device", "cpu", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["practice"] == "P1-03"
    assert report["passed"] == 10


def test_layout_cuda_request_does_not_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        evaluate_layout("cuda")


def test_layout_reference_rejects_invalid_input():
    x = torch.zeros(1, 2, 6)
    with pytest.raises(ValueError):
        split_heads(x, 4, 2)
    with pytest.raises(ValueError):
        split_heads(x.transpose(1, 2), 2, 3)
    with pytest.raises(ValueError):
        merge_heads(torch.zeros(1, 2, 0, 3))
