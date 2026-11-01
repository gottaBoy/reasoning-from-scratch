import json

import pytest
import torch

from learning.triton_ir import inspect_vector_add, save_artifacts


def test_artifact_export_and_refusal_to_overwrite(tmp_path):
    report = {"numerical_check": "passed"}
    target = tmp_path / "run with spaces"
    artifacts = {"ttir": "module {}\n", "ptx": ".version 8.0\n", "cubin": b"\x7fELF"}
    files = save_artifacts(artifacts, target, report)
    assert files["cubin"] == "vector_add.cubin"
    assert (target / files["cubin"]).read_bytes() == b"\x7fELF"
    assert json.loads((target / "report.json").read_text())["numerical_check"] == "passed"
    with pytest.raises(FileExistsError):
        save_artifacts(artifacts, target, report)
    assert (target / files["ttir"]).read_text() == "module {}\n"


def test_ir_experiment_does_not_silently_use_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        inspect_vector_add()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real compiler inspection requires GPU")
@pytest.mark.parametrize("size,block", [(1, 128), (1003, 256), (1025, 512)])
def test_real_gpu_ir_and_masked_tail(size, block):
    pytest.importorskip("triton")
    report = inspect_vector_add(size, block)
    assert report["numerical_check"] == "passed"
    assert report["grid"] == [(size + block - 1) // block]
    assert all(report["stages_present"].values())
    assert all(report["preview"][stage].strip() for stage in ("ttir", "ttgir", "llir", "ptx"))
