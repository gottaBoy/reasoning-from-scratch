"""Exercise the acceptance runner, including intentionally wrong learner modules."""

import json

import pytest
import torch

from learning.local_practice import evaluate, main, write_report


def test_reference_baseline_passes_all_cpu_cases_without_claiming_learner_authorship():
    report = evaluate("cpu")
    assert report["status"] == "passed"
    assert report["passed"] == 14 and report["failed"] == 0
    assert report["implementation_role"] == "reference_baseline"
    assert len(report["source_sha256"]) == 64
    assert report["device"] == "cpu"


def test_trusted_candidate_file_is_actually_loaded(tmp_path):
    source = tmp_path / "my_ops.py"
    source.write_text("from learning.mini_transformer import RMSNorm, rope\n")
    report = evaluate("cpu", source)
    assert report["status"] == "passed"
    assert report["implementation_role"] == "candidate_file"
    assert report["source"] == str(source)


def test_incorrect_candidate_produces_failures_instead_of_printing_passed(tmp_path):
    source = tmp_path / "wrong_ops.py"
    source.write_text(
        "from learning.mini_transformer import RMSNorm\n"
        "def rope(x, start=0, theta=10000.0):\n"
        "    return x\n"
    )
    report = evaluate("cpu", source)
    assert report["status"] == "failed" and report["failed"] >= 3
    cases = {case["name"]: case for case in report["cases"]}
    assert cases["rope_offset_5"]["status"] == "failed"
    assert cases["rope_reject_odd_dimension"]["status"] == "failed"


def test_cuda_request_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        evaluate("cuda")


def test_bad_candidate_has_explicit_interface_error(tmp_path):
    source = tmp_path / "missing_exports.py"
    source.write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="export RMSNorm"):
        evaluate("cpu", source)


def test_report_writer_preserves_existing_evidence(tmp_path):
    path = tmp_path / "report.json"
    write_report({"status": "passed"}, path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_report({"status": "failed"}, path)
    assert path.read_bytes() == before


def test_cli_records_environment_errors_and_returns_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / "cuda-error.json"
    assert main(["--device", "cuda", "--output", str(path)]) == 1
    report = json.loads(path.read_text())
    assert report["status"] == "error" and report["device"] == "cuda"


def test_cli_refuses_overwriting_old_report(tmp_path):
    path = tmp_path / "old.json"
    path.write_text('{"original": true}\n')
    with pytest.raises(SystemExit) as error:
        main(["--output", str(path)])
    assert error.value.code == 2
    assert path.read_text() == '{"original": true}\n'
