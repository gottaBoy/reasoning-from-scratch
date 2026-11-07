"""Attach detailed lesson notes to existing notebooks without regenerating learner cells."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

import nbformat


GUIDES = {
    9: "DAY09_SINGLE_ATTENTION.md",
    10: "DAY10_MULTIHEAD.md",
    11: "DAY11_GQA_MQA.md",
    12: "DAY12_REQUEST_CACHE.md",
    13: "DAY13_BLOCK_BACKWARD.md",
    14: "DAY14_MODEL_BASELINE.md",
    15: "DAY15_CUDA_INDEXING.md",
    16: "DAY16_SHARED_REDUCTION.md",
    17: "DAY17_TRITON_ROWS.md",
    18: "DAY18_STABLE_ROW_OPS.md",
    19: "DAY19_TILED_MATMUL.md",
    20: "DAY20_ATTENTION_ORACLE.md",
    21: "DAY21_FUSED_KERNELS.md",
    22: "DAY22_SERVING_BOUNDARY.md",
    23: "DAY23_PAGED_ADDRESSING.md",
    24: "DAY24_SCHEDULER_FAIRNESS.md",
    25: "DAY25_OWNERSHIP_OOM.md",
    26: "DAY26_BATCHED_ENGINE.md",
    27: "DAY27_METRICS_BENCHMARK.md",
    28: "DAY28_PREFIX_CACHE.md",
    29: "DAY29_REPRODUCIBLE_REPORT.md",
    30: "DAY30_CAPSTONE.md",
}
HEADINGS = (
    "## 1. 图解与数字例子", "## 2. 概念与计算契约", "## 3. Jupyter与源码路线",
    "## 4. 调试与优化顺序", "## 5. 自测与参考答案", "## 6. 验收与资料",
)


def guide_sections(day, docs):
    """Read the explicit six-section format authored for this course."""
    text = (Path(docs) / GUIDES[day]).read_text()
    positions = []
    for heading in HEADINGS:
        marker = heading + "\n"
        if text.count(marker) != 1:
            raise ValueError(f"Expected exactly one {heading} in Day {day}.")
        positions.append(text.index(marker))
    if positions != sorted(positions):
        raise ValueError(f"Day {day} sections are out of order.")
    ends = positions[1:] + [len(text)]
    return {heading: text[start:end].strip()
            for heading, start, end in zip(HEADINGS, positions, ends)}


def attach_guide(notebook, day, docs):
    sections = guide_sections(day, docs)
    link = f"../docs/{GUIDES[day]}"
    sources = (
        f"[Day {day}完整讲义]({link})：概念、计算契约、源码实验、调试优化、答案与资料。\n\n"
        "先写自己的预测，再执行原有代码单元格；不要自动勾选个人验收。\n\n"
        + sections[HEADINGS[0]],
        sections[HEADINGS[4]] + f"\n\n完整实现限制、调试步骤与资料见[本日讲义]({link})。"
        "\n\n以上是参考答案，不是个人学习结果。",
    )
    for kind, source in zip(("intuition", "answers"), sources):
        cell_id = f"day{day:02d}-detail-{kind}"
        matches = [cell for cell in notebook.cells if cell.id == cell_id]
        if len(matches) > 1:
            raise ValueError(f"Duplicate managed cell {cell_id}.")
        if matches:
            cell = matches[0]
            if cell.cell_type != "markdown" or cell.metadata.get("course_guide") != GUIDES[day]:
                raise ValueError(f"Refusing unmanaged cell {cell_id}.")
            cell.source = source
        else:
            cell = nbformat.v4.new_markdown_cell(
                source, id=cell_id, metadata={"course_guide": GUIDES[day]},
            )
            notebook.cells.insert(1 if kind == "intuition" else len(notebook.cells) - 1, cell)
    nbformat.validate(notebook)
    return notebook


def enrich(directory, docs, apply=False, backup_root=None):
    directory, docs = Path(directory), Path(docs)
    prepared = []
    # Validate every source before changing any destination.
    for day in GUIDES:
        path = directory / f"day{day:02d}_course.ipynb"
        notebook = nbformat.read(path, as_version=4)
        original = nbformat.writes(notebook)
        attach_guide(notebook, day, docs)
        if nbformat.writes(notebook) != original:
            prepared.append((path, notebook))
    if apply and prepared:
        if backup_root is None:
            raise ValueError("Explicit backup root required when writing notebooks.")
        backup = Path(backup_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup.mkdir(parents=True, exist_ok=False)
        for path, notebook in prepared:
            shutil.copy2(path, backup / path.name)
            nbformat.write(notebook, path)
    return [str(path) for path, _ in prepared]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=root / "learning")
    parser.add_argument("--docs", type=Path, default=root / "docs")
    parser.add_argument("--backup-root", type=Path, default=root / "logs/course-guide-backup")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    for path in enrich(args.directory, args.docs, args.apply, args.backup_root):
        print(("Updated: " if args.apply else "Would update: ") + path)
