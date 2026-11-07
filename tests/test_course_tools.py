from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import torch

from learning.build_course_notebooks import build
from learning.curriculum import LESSONS, run_day
from learning.vllm_probe import probe


def test_generated_notebooks_have_valid_code_and_refuse_replacement(tmp_path):
    paths = build(tmp_path)
    assert len(paths) == 25
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert "LEARNING_PROGRESS.md" in notebook.cells[0].source
        assert "当前个人进度仍为 Day 5" not in notebook.cells[0].source
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, path, "exec")
    before = Path(paths[0]).read_bytes()
    with pytest.raises(FileExistsError):
        build(tmp_path)
    assert Path(paths[0]).read_bytes() == before
    assert set(LESSONS) == set(range(6, 31))


@pytest.mark.parametrize("day", range(6, 31))
def test_cpu_course_status_is_honest(day, monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("CUDA_LESSON_BINARY", raising=False)
    report = run_day(day, "cpu")
    expected = ("requires_cuda_compiler" if day in (15, 16) else
                "requires_cuda" if day in (17, 18, 19, 21) else
                "external_service_not_configured" if day == 22 else "passed")
    assert report["status"] == expected


def test_vllm_probe_checks_payload_without_contacting_external_service(monkeypatch):
    from learning import vllm_probe
    calls = []

    def response(data):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

    def get(url, timeout):
        calls.append(url)
        return response({"data": [{"id": "test-model"}]})

    def post(url, json, timeout):
        calls.append(url)
        assert json["model"] == "test-model"
        assert json["max_tokens"] == 8
        return response({"usage": {"completion_tokens": 1}, "choices": [{"text": "Paris"}]})

    monkeypatch.setattr(vllm_probe.requests, "get", get)
    monkeypatch.setattr(vllm_probe.requests, "post", post)
    result = probe("http://example.invalid/v1/")
    assert result["text"] == "Paris"
    assert calls == ["http://example.invalid/v1/models", "http://example.invalid/v1/completions"]


def test_cuda_is_not_silently_substituted(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        run_day(6, "cuda")
