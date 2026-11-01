import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_environment_check():
    spec = importlib.util.spec_from_file_location(
        "docker_environment_check", ROOT / "docker/check_environment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_environment_check():
    result = load_environment_check().check_environment("cpu")
    assert result["device"] == "cpu"
    assert result["matmul"] == "passed"
    assert result["attention_forward_backward"] == "passed"


def test_cuda_check_does_not_silently_fall_back(monkeypatch):
    module = load_environment_check()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        module.check_environment("cuda")


@pytest.mark.skipif(shutil.which("sh") is None, reason="POSIX shell required")
@pytest.mark.parametrize("failure", [False, True])
def test_smoke_stage_logging(tmp_path, failure):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    fake_python.write_text(
        '#!/bin/sh\n'
        'printf "fake python: %s\\n" "$*"\n'
        'case "$1" in\n'
        '  *evaluate_math500.py) exit "${FAKE_EXIT_CODE:-0}" ;;\n'
        'esac\n'
    )
    fake_python.chmod(0o755)
    logs = tmp_path / "logs with spaces"
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "SMOKE_LOG_DIR": str(logs),
        "FAKE_EXIT_CODE": "7" if failure else "0",
        "RUN_TRAINING": "1",
        "RUN_ADVANCED_TRAINING": "1",
        "RUN_TESTS": "1",
    }
    result = subprocess.run(
        ["sh", str(ROOT / "docker/smoke_examples.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    rows = (logs / "summary.tsv").read_text().splitlines()
    if failure:
        assert result.returncode == 7
        assert len(rows) == 4
        assert rows[-1].startswith("ch03_evaluation\t7\t")
        assert "Smoke examples completed." not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert len(rows) == 12
        assert all("\t0\t" in row for row in rows[1:])
        assert (logs / "ch07_format_reward.log").is_file()
