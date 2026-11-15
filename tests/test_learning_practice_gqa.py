"""P1-04 acceptance tests for MHA/GQA/MQA KV head expansion."""

import json

import pytest
import torch

from learning.local_practice import main
from learning.practice_gqa import evaluate_gqa, expand_kv_heads


def test_gqa_reference_passes_all_cpu_checks():
    report = evaluate_gqa("cpu")
    assert report["practice"] == "P1-04"
    assert report["status"] == "passed"
    assert report["passed"] == 10 and report["failed"] == 0
    assert report["implementation_role"] == "reference_baseline"


def test_gqa_candidate_path_is_loaded(tmp_path):
    path = tmp_path / "gqa_ops.py"
    path.write_text("from learning.practice_gqa import expand_kv_heads\n")
    report = evaluate_gqa("cpu", path)
    assert report["status"] == "passed"
    assert report["passed"] == 10
    assert report["implementation_role"] == "candidate_file"


def test_wrong_mapping_is_detected(tmp_path):
    path = tmp_path / "wrong_gqa.py"
    path.write_text(
        "def expand_kv_heads(kv, query_heads):\n"
        "    return kv.repeat(1, query_heads, 1, 1)\n"
    )
    report = evaluate_gqa("cpu", path)
    cases = {case["name"]: case for case in report["cases"]}
    assert report["status"] == "failed"
    assert cases["gqa_group_mapping"]["status"] == "failed"
    assert cases["nonsquare_shape_mapping"]["status"] == "failed"


def test_gqa_cli_reports_p1_04(tmp_path):
    output = tmp_path / "gqa.json"
    assert main(["--task", "gqa", "--device", "cpu", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["practice"] == "P1-04"
    assert report["passed"] == 10


def test_gqa_cuda_request_does_not_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        evaluate_gqa("cuda")


def test_gqa_reference_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        expand_kv_heads(torch.zeros(1, 2, 2, 3), 3)
    with pytest.raises(ValueError):
        expand_kv_heads(torch.zeros(1, 2, 2, 3), 1)
    with pytest.raises(ValueError):
        expand_kv_heads(torch.zeros(1, 2, 3), 4)
