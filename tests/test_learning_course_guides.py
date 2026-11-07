"""Detailed guides must cover all remaining days without replacing learner content."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest

from learning.course_guides import GUIDES, HEADINGS, attach_guide, enrich, guide_sections


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


@pytest.mark.parametrize("day", range(9, 31))
def test_all_remaining_days_have_concrete_guides_and_answers(day):
    parts = guide_sections(day, DOCS)
    assert len(parts) == 6
    assert "```" in parts[HEADINGS[0]]
    assert parts[HEADINGS[4]].count("**问：") >= 4
    assert parts[HEADINGS[4]].count("答：") >= 4
    assert "http" in parts[HEADINGS[5]]
    assert f"day{day:02d}_course.ipynb" in parts[HEADINGS[2]]
    notebook = nbformat.read(ROOT / "learning" / f"day{day:02d}_course.ipynb", as_version=4)
    managed = {cell.id: cell for cell in notebook.cells
               if cell.metadata.get("course_guide") == GUIDES[day]}
    assert parts[HEADINGS[0]] in managed[f"day{day:02d}-detail-intuition"].source
    assert parts[HEADINGS[4]] in managed[f"day{day:02d}-detail-answers"].source


def learner_notebook():
    return nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# Personal notes", id="personal-intro"),
        nbformat.v4.new_code_cell(
            "print('my experiment')", id="personal-code", execution_count=3,
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="my experiment\n")],
        ),
        nbformat.v4.new_markdown_cell("My answers: not finished", id="personal-record"),
    ], metadata={"personal_setting": "preserve"})


def test_attachment_preserves_original_cells_outputs_and_metadata_and_is_idempotent():
    notebook = learner_notebook()
    original = deepcopy(notebook)
    attach_guide(notebook, 9, DOCS)
    for old in original.cells:
        assert next(cell for cell in notebook.cells if cell.id == old.id) == old
    assert notebook.metadata == original.metadata
    assert len(notebook.cells) == 5
    enriched = deepcopy(notebook)
    attach_guide(notebook, 9, DOCS)
    assert notebook == enriched


def test_attachment_refuses_unmanaged_id_collision():
    notebook = learner_notebook()
    notebook.cells.append(nbformat.v4.new_markdown_cell("learner text", id="day09-detail-answers"))
    with pytest.raises(ValueError, match="unmanaged"):
        attach_guide(notebook, 9, DOCS)


def test_preview_and_apply_keep_original_backups(tmp_path):
    target = tmp_path / "learning"
    target.mkdir()
    for day in GUIDES:
        nbformat.write(learner_notebook(), target / f"day{day:02d}_course.ipynb")
    first = target / "day09_course.ipynb"
    original = first.read_bytes()
    assert len(enrich(target, DOCS)) == 22
    assert first.read_bytes() == original
    backup = tmp_path / "backups"
    assert len(enrich(target, DOCS, apply=True, backup_root=backup)) == 22
    preserved = list(backup.glob("*/day09_course.ipynb"))
    assert len(preserved) == 1 and preserved[0].read_bytes() == original
    assert enrich(target, DOCS, apply=True, backup_root=backup) == []


def test_invalid_last_guide_does_not_partially_write_notebooks(tmp_path):
    target = tmp_path / "learning"
    target.mkdir()
    for day in GUIDES:
        nbformat.write(learner_notebook(), target / f"day{day:02d}_course.ipynb")
    (target / "day30_course.ipynb").write_text("not a notebook")
    first = target / "day09_course.ipynb"
    original = first.read_bytes()
    with pytest.raises(nbformat.reader.NotJSONError):
        enrich(target, DOCS, apply=True, backup_root=tmp_path / "backups")
    assert first.read_bytes() == original


def test_verify_refuses_duplicate_days_without_creating_output(tmp_path):
    output = tmp_path / "reports"
    result = subprocess.run(
        [sys.executable, "-m", "learning.verify_course", "--days", "9", "9",
         "--output-dir", str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2 and "Duplicate days" in result.stderr
    assert not output.exists()


def test_verify_does_not_replace_existing_summary(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text('{"original": true}\n')
    result = subprocess.run(
        [sys.executable, "-m", "learning.verify_course", "--days", "9",
         "--output-dir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0 and "fresh output" in result.stderr
    assert summary.read_text() == '{"original": true}\n'
