"""P2-02 environment/protocol tests; actual CUDA checks run separately."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from learning import practice_reduction as practice


def _result():
    return {
        "practice": "P2-02", "status": "passed", "passed": 6, "failed": 0,
        "capability": [12, 1], "threads_per_block": 256, "patterns": 3,
        "cases": [
            {"n": n, "blocks": (n + 255) // 256, "status": "passed",
             "partial_mismatches": 0, "total_mismatches": 0,
             "guard_mismatches": 0, "input_mismatches": 0}
            for n in (1, 255, 256, 257, 1003, 65537)
        ],
    }


def _fake_tools(monkeypatch, result=None, bad_tool=None):
    commands = []
    monkeypatch.setattr(practice.shutil, "which", lambda name: "/tools/" + name)

    def run(command, **kwargs):
        commands.append(command)
        stdout = ""
        code = 0
        if "--version" in command:
            stdout = "fixture nvcc"
        elif command[0] != "/tools/nvcc":
            stdout = json.dumps(_result() if result is None else result)
            if "--tool" in command:
                tool = command[command.index("--tool") + 1]
                log = ("RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                       if tool == "racecheck" else "ERROR SUMMARY: 0 errors")
                if tool == bad_tool:
                    log, code = "ERROR SUMMARY: 1 error", 99
                Path(command[command.index("--log-file") + 1]).write_text(log)
        return subprocess.CompletedProcess(command, code, stdout=stdout, stderr="")

    monkeypatch.setattr(practice.subprocess, "run", run)
    return commands


def test_reference_runs_all_sanitizers_sequentially(monkeypatch):
    commands = _fake_tools(monkeypatch)
    report = practice.evaluate_reduction("sm_121", sanitize=True)
    assert report["status"] == "passed"
    assert report["implementation_role"] == "reference_baseline"
    assert [row["tool"] for row in report["sanitizers"]] == list(practice._TOOLS)
    assert all(row["clean_summary"] for row in report["sanitizers"])
    assert len(commands) == 6  # version, compilation, base run, three sanitizers


@pytest.mark.parametrize("tool", practice._TOOLS)
def test_sanitizer_failure_cannot_be_hidden_by_correct_numbers(monkeypatch, tool):
    _fake_tools(monkeypatch, bad_tool=tool)
    report = practice.evaluate_reduction("sm_121", sanitize=True)
    assert report["status"] == "failed"
    assert report["sanitizers"][-1]["tool"] == tool
    assert report["sanitizers"][-1]["returncode"] == 99


def test_partial_error_is_detected_even_when_total_matches(monkeypatch):
    result = _result()
    result["cases"][3].update(status="failed", partial_mismatches=2)
    result.update(status="failed", passed=5, failed=1)
    _fake_tools(monkeypatch, result=result)
    report = practice.evaluate_reduction("sm_121", sanitize=True)
    assert report["status"] == "failed"
    assert report["cases"][3]["total_mismatches"] == 0
    assert report["sanitizers"] == []


@pytest.mark.parametrize("fault", ("grid", "negative", "nan", "summary", "status", "patterns"))
def test_inconsistent_evidence_is_rejected(fault):
    result = deepcopy(_result())
    if fault == "grid":
        result["cases"][3]["blocks"] = 1
    elif fault == "negative":
        result["cases"][0]["partial_mismatches"] = -1
    elif fault == "nan":
        result["cases"][0]["partial_mismatches"] = float("nan")
    elif fault == "summary":
        result["passed"] = 5
    elif fault == "status":
        result["cases"][0]["input_mismatches"] = 1
    else:
        result["patterns"] = 1
    with pytest.raises(ValueError):
        practice._parse_result(json.dumps(result))


def test_candidate_path_and_hash_are_recorded(tmp_path, monkeypatch):
    source = tmp_path / "candidate.cu"
    source.write_text("// fixture kernel\n")
    commands = _fake_tools(monkeypatch)
    report = practice.evaluate_reduction("sm_121", source)
    assert report["implementation_role"] == "candidate_file"
    assert report["source"] == str(source)
    assert len(report["source_sha256"]) == 64
    assert str(source) in commands[1]


def test_missing_compiler_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(practice.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        practice.evaluate_reduction("sm_121")


def test_requested_sanitizer_is_required(monkeypatch):
    monkeypatch.setattr(practice.shutil, "which",
                        lambda name: "/tools/nvcc" if name == "nvcc" else None)
    with pytest.raises(RuntimeError, match="explicitly requested"):
        practice.evaluate_reduction("sm_121", sanitize=True)


def test_missing_summary_cannot_pass():
    assert not practice._clean_summary("memcheck", "")
    assert not practice._clean_summary("racecheck", "ERROR SUMMARY: 0 errors")
    assert not practice._clean_summary(
        "racecheck", "RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)",
    )


def test_cli_preserves_old_report(tmp_path):
    output = tmp_path / "old.json"
    output.write_text('{"original": true}\n')
    with pytest.raises(SystemExit):
        practice.main(["--arch", "sm_121", "--output", str(output)])
    assert json.loads(output.read_text()) == {"original": True}


def test_cli_records_environment_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(practice.shutil, "which", lambda name: None)
    output = tmp_path / "error.json"
    assert practice.main(["--arch", "sm_121", "--output", str(output)]) == 1
    assert json.loads(output.read_text())["status"] == "error"
