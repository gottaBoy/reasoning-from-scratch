"""P2-01 protocol/environment tests; CUDA execution is verified separately."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from learning import practice_vector_add as practice


def _result():
    return {
        "practice": "P2-01", "status": "passed", "passed": 6, "failed": 0,
        "capability": [12, 1], "threads_per_block": 256,
        "cases": [
            {"n": n, "blocks": (n + 255) // 256, "status": "passed",
             "value_mismatches": 0, "guard_mismatches": 0, "input_mismatches": 0}
            for n in (1, 255, 256, 257, 1003, 65537)
        ],
    }


def _fake_tools(monkeypatch, compile_code=0, run_code=0, result=None, sanitizer_text=None):
    commands = []
    monkeypatch.setattr(practice.shutil, "which", lambda name: "/tools/" + name)

    def run(command, **kwargs):
        commands.append(command)
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, stdout="fixture nvcc", stderr="")
        if command[0] == "/tools/nvcc":
            return subprocess.CompletedProcess(
                command, compile_code, stdout="", stderr="compile error" if compile_code else "",
            )
        if "--log-file" in command and sanitizer_text is not None:
            Path(command[command.index("--log-file") + 1]).write_text(sanitizer_text)
        return subprocess.CompletedProcess(
            command, run_code, stdout=json.dumps(_result() if result is None else result),
            stderr="",
        )

    monkeypatch.setattr(practice.subprocess, "run", run)
    return commands


def test_valid_harness_protocol():
    assert practice._parse_result(json.dumps(_result()))["passed"] == 6


@pytest.mark.parametrize("fault", ("size", "grid", "summary", "negative", "nan", "case_status"))
def test_invalid_harness_protocol_cannot_pass(fault):
    result = deepcopy(_result())
    if fault == "size":
        result["cases"][0]["n"] = 0
    elif fault == "grid":
        result["cases"][3]["blocks"] = 1
    elif fault == "summary":
        result["passed"] = 5
    elif fault == "negative":
        result["cases"][0]["value_mismatches"] = -1
    elif fault == "nan":
        result["cases"][0]["value_mismatches"] = float("nan")
    else:
        result["cases"][0]["guard_mismatches"] = 1
    with pytest.raises(ValueError):
        practice._parse_result(json.dumps(result))


def test_trusted_candidate_is_compiled_and_identified(tmp_path, monkeypatch):
    source = tmp_path / "kernel.cu"
    source.write_text("// loader fixture, not a learner implementation\n")
    commands = _fake_tools(monkeypatch)
    report = practice.evaluate_vector_add("sm_121", source)
    assert report["status"] == "passed"
    assert report["implementation_role"] == "candidate_file"
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert str(source) in commands[1]
    assert "-arch=sm_121" in commands[1]


def test_compile_failure_is_not_success(monkeypatch):
    _fake_tools(monkeypatch, compile_code=1)
    report = practice.evaluate_vector_add("sm_121")
    assert report["status"] == "error" and report["stage"] == "compile"
    assert report["compile"]["stderr"] == "compile error"


def test_sanitizer_failure_overrides_passing_values(monkeypatch):
    commands = _fake_tools(monkeypatch, run_code=99)
    report = practice.evaluate_vector_add("sm_121", sanitize=True)
    assert report["status"] == "failed"
    assert "--error-exitcode" in commands[-1]
    assert report["sanitizer"]["returncode"] == 99


def test_sanitizer_requires_clean_summary_even_with_zero_exit(monkeypatch):
    _fake_tools(monkeypatch)
    report = practice.evaluate_vector_add("sm_121", sanitize=True)
    assert report["status"] == "failed"
    assert not report["sanitizer"]["clean_memcheck"]


def test_sanitizer_clean_summary_is_recorded(monkeypatch):
    _fake_tools(monkeypatch, sanitizer_text="========= ERROR SUMMARY: 0 errors\n")
    report = practice.evaluate_vector_add("sm_121", sanitize=True)
    assert report["status"] == "passed"
    assert report["sanitizer"]["clean_memcheck"]


def test_kernel_value_failure_remains_failed(monkeypatch):
    result = _result()
    result["cases"][0].update(status="failed", value_mismatches=1)
    result.update(status="failed", passed=5, failed=1)
    _fake_tools(monkeypatch, run_code=2, result=result)
    report = practice.evaluate_vector_add("sm_121")
    assert report["status"] == "failed" and report["failed"] == 1


def test_missing_compiler_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(practice.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        practice.evaluate_vector_add("sm_121")


def test_requested_sanitizer_is_required(monkeypatch):
    monkeypatch.setattr(
        practice.shutil, "which", lambda name: "/tools/nvcc" if name == "nvcc" else None,
    )
    with pytest.raises(RuntimeError, match="explicitly requested"):
        practice.evaluate_vector_add("sm_121", sanitize=True)


@pytest.mark.parametrize("arch", ("", "native", "sm_121;echo", "../sm_121"))
def test_architecture_is_explicit_and_validated(arch):
    with pytest.raises(ValueError, match="architecture"):
        practice.evaluate_vector_add(arch)


def test_candidate_must_be_a_kernel_source(tmp_path):
    source = tmp_path / "candidate.py"
    source.write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="\\.cu"):
        practice.evaluate_vector_add("sm_121", source)


def test_cli_preserves_existing_evidence(tmp_path):
    output = tmp_path / "old.json"
    output.write_text('{"original": true}\n')
    with pytest.raises(SystemExit):
        practice.main(["--arch", "sm_121", "--output", str(output)])
    assert json.loads(output.read_text()) == {"original": True}


def test_cli_records_missing_compiler(tmp_path, monkeypatch):
    monkeypatch.setattr(practice.shutil, "which", lambda name: None)
    output = tmp_path / "error.json"
    assert practice.main(["--arch", "sm_121", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "error"
    assert report["device"] == "cuda"
    assert "no CPU fallback" in report["error"]
