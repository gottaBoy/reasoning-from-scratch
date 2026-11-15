"""P1-05 acceptance tests for request-scoped KV Cache behavior."""

import json

from learning.local_practice import main
from learning.practice_cache import evaluate_cache


def test_cache_reference_passes_all_cpu_checks():
    report = evaluate_cache("cpu")
    assert report["practice"] == "P1-05"
    assert report["status"] == "passed"
    assert report["passed"] == 7 and report["failed"] == 0
    assert report["implementation_role"] == "reference_baseline"


def test_cache_cli_reports_p1_05(tmp_path):
    output = tmp_path / "cache.json"
    assert main(["--task", "cache", "--device", "cpu", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["practice"] == "P1-05"
    assert report["passed"] == 7
