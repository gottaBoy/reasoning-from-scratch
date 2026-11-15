"""P1-07 acceptance tests for same-workload model baselines."""

import json

import pytest
import torch

from learning.local_practice import main
from learning.practice_model import evaluate_model


def test_model_reference_passes_all_cpu_checks():
    report = evaluate_model("cpu")
    assert report["practice"] == "P1-07"
    assert report["status"] == "passed"
    assert report["passed"] == 4 and report["failed"] == 0
    assert report["implementation_role"] == "reference_baseline"
    assert report["configuration"]["kv_heads"] == 2
    assert report["input_ids"] == [[3, 7, 2, 9, 4, 1, 8, 6]]
    assert report["split"] == [5, 1, 1, 1]
    assert report["dtype"] == "torch.float32"


def test_model_cli_reports_p1_07(tmp_path):
    output = tmp_path / "model.json"
    assert main(["--task", "baseline", "--device", "cpu", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["practice"] == "P1-07"
    assert report["passed"] == 4


def test_baseline_candidate_is_loaded(tmp_path):
    path = tmp_path / "candidate.py"
    # A loader fixture, not a learner implementation.
    path.write_text("from learning.mini_transformer.model import Cache, Config, Model\n")
    report = evaluate_model("cpu", path)
    assert report["status"] == "passed"
    assert report["implementation_role"] == "candidate_file"
    assert report["source"] == str(path.resolve())


def test_unfinished_baseline_candidate_does_not_pass(tmp_path):
    path = tmp_path / "unfinished.py"
    path.write_text(
        "from learning.mini_transformer.model import Cache, Config\n"
        "def Model(config):\n"
        "    raise NotImplementedError('unfinished')\n"
    )
    report = evaluate_model("cpu", path)
    assert report["status"] == "failed"
    assert report["passed"] == 0 and report["failed"] == 4
    assert all("unfinished" in case["error"] for case in report["cases"])


def test_matching_but_wrong_logits_shape_is_rejected(tmp_path):
    path = tmp_path / "wrong_logits.py"
    path.write_text(
        "from learning.mini_transformer.model import Cache, Config, Model as Reference\n"
        "class Model(Reference):\n"
        "    def forward(self, ids, cache=None):\n"
        "        return super().forward(ids, cache)[..., :1]\n"
    )
    report = evaluate_model("cpu", path)
    cases = {case["name"]: case for case in report["cases"]}
    assert report["status"] == "failed"
    assert cases["full_cached_logits_alignment"]["status"] == "failed"


def test_baseline_cuda_request_does_not_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        evaluate_model("cuda")
