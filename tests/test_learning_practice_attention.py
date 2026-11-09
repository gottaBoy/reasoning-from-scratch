"""P1-02 acceptance tests; learner files are never imported by this test suite."""

import json
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

from learning.local_practice import main
from learning.practice_attention import causal_attention, evaluate_attention


def test_attention_reference_passes_all_cpu_checks():
    report = evaluate_attention("cpu")
    assert report["practice"] == "P1-02" and report["status"] == "passed"
    assert report["passed"] == 14 and report["failed"] == 0
    assert report["implementation_role"] == "reference_baseline"


def test_attention_candidate_path_is_loaded_not_norm_rope_interface(tmp_path):
    path = tmp_path / "attention_ops.py"
    path.write_text("from learning.practice_attention import causal_attention\n")
    report = evaluate_attention("cpu", path)
    assert report["status"] == "passed" and report["passed"] == 14
    assert report["implementation_role"] == "candidate_file"
    assert report["source"] == str(path)


def test_wrong_attention_offset_is_detected(tmp_path):
    path = tmp_path / "wrong_attention.py"
    path.write_text(
        "from learning.practice_attention import causal_attention as reference\n"
        "def causal_attention(q, k, v, start=0):\n"
        "    return reference(q, k, v, start=0)\n"
    )
    report = evaluate_attention("cpu", path)
    cases = {case["name"]: case for case in report["cases"]}
    assert report["status"] == "failed"
    assert cases["cached_single_query"]["status"] == "failed"
    assert cases["cached_query_chunk"]["status"] == "failed"
    assert cases["reject_negative_start"]["status"] == "failed"


def test_rectangular_attention_matches_explicit_absolute_sdpa_mask():
    generator = torch.Generator().manual_seed(1)
    q = torch.randn(2, 1, 3, 4, generator=generator)
    k = torch.randn(2, 1, 7, 4, generator=generator)
    v = torch.randn(2, 1, 7, 2, generator=generator)
    allowed = torch.arange(7)[None, :] <= torch.arange(4, 7)[:, None]
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0)
    torch.testing.assert_close(causal_attention(q, k, v, start=4), expected,
                               atol=2e-6, rtol=1e-5)


def test_attention_cli_reports_task_and_passed_cases(tmp_path):
    path = tmp_path / "report.json"
    assert main(["--task", "attention", "--device", "cpu", "--output", str(path)]) == 0
    result = json.loads(path.read_text())
    assert result["practice"] == "P1-02" and result["passed"] == 14


def test_attention_cuda_explicit_request_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        evaluate_attention("cuda")


def test_attention_cli_records_missing_candidate_as_p1_02_error(tmp_path):
    path = tmp_path / "failure.json"
    assert main(["--task", "attention", "--candidate", str(tmp_path / "missing.py"),
                 "--output", str(path)]) == 1
    result = json.loads(path.read_text())
    assert result["status"] == "error" and result["practice"] == "P1-02"


def test_attention_scaffold_has_valid_guards_but_not_a_completed_body():
    from learning.my_attention_template import causal_attention as unfinished
    q = torch.ones(1, 1, 1, 2)
    with pytest.raises(NotImplementedError):
        unfinished(q, q, q)
    with pytest.raises(ValueError):
        unfinished(q, q, q, start=-1)


def test_acceptance_suite_does_not_reference_the_live_learner_file():
    text = Path(__file__).read_text()
    assert "learning.my_attention" not in sys.modules
    assert "import my_attention\n" not in text
