"""P1-06 acceptance tests for Block/SwiGLU/LM backward behavior."""

import json

from learning.local_practice import main
from learning.practice_block import evaluate_block


def test_block_reference_passes_all_cpu_checks():
    report = evaluate_block("cpu")
    assert report["practice"] == "P1-06"
    assert report["status"] == "passed"
    assert report["passed"] == 6 and report["failed"] == 0
    assert report["implementation_role"] == "reference_baseline"


def test_block_cli_reports_p1_06(tmp_path):
    output = tmp_path / "block.json"
    assert main(["--task", "block", "--device", "cpu", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["practice"] == "P1-06"
    assert report["passed"] == 6
