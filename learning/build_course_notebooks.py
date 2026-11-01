"""Generate new per-day notebooks; never overwrite existing learner notebooks by default."""

import argparse
from pathlib import Path

import nbformat

from learning.curriculum import LESSONS


def build(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for day, (title, explanation, source, questions) in LESSONS.items():
        week = "01_07" if day <= 7 else "08_14" if day <= 14 else "15_21" if day <= 21 else "22_30"
        path = directory / f"day{day:02d}_course.ipynb"
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing notebook: {path}")
        setup = (
            "import json\nimport os\nimport sys\nfrom pathlib import Path\n"
            "root = Path.cwd()\n"
            "if root.name == 'learning':\n    root = root.parent\n"
            "if not (root / 'learning/curriculum.py').is_file():\n"
            "    raise RuntimeError('Open from repository root or learning directory.')\n"
            "os.chdir(root)\n"
            "if str(root) not in sys.path:\n    sys.path.insert(0, str(root))\n"
            "import torch\nfrom learning.curriculum import run_day\n"
            "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "print('Device:', DEVICE)\n"
        )
        run = (
            f"report = run_day({day}, DEVICE)\n"
            "print(json.dumps(report, indent=2, ensure_ascii=False))\n"
            "destination = Path('logs') / f\"day{report['day']:02d}-course.json\"\n"
            "destination.parent.mkdir(parents=True, exist_ok=True)\n"
            "destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\\n')\n"
            "print('Report:', destination)\n"
        )
        nb = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(
                    f"# Day {day}: {title}\n\n{explanation}\n\n"
                    f"参考源码：`learning/{source}`。\n\n"
                    f"每日概念、误区、面试问答："
                    f"[知识讲义](../docs/KNOWLEDGE_DAY{week}.md)。\n\n"
                    "当前个人进度仍为 Day 5；运行参考答案不等于已独立掌握。"
                ),
                nbformat.v4.new_markdown_cell(
                    "## 先预测\n\n执行前写下预期 shape、输出或调度顺序；"
                    "在下面另建 Markdown 单元格保存自己的答案，不修改参考实现作为唯一证据。"
                ),
                nbformat.v4.new_code_cell(setup),
                nbformat.v4.new_code_cell(run),
                nbformat.v4.new_markdown_cell(
                    f"## 自测与失败实验\n\n{questions}\n\n"
                    "记录你改变的唯一变量、预期、实际结果和差异原因。\n\n"
                    "`requires_cuda` / `requires_cuda_compiler` / "
                    "`external_service_not_configured` 都不是通过。"
                ),
                nbformat.v4.new_markdown_cell(
                    "## 验收记录\n\n- [ ] 能解释原理\n- [ ] 能自行复现\n"
                    "- [ ] 能解释一次失败\n- [ ] 能说明当前实现不支持什么\n\n"
                    "个人笔记：\n\n待填写。"
                ),
            ],
            metadata={"kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)",
                                     "language": "python"}},
        )
        nbformat.write(nb, path)
        paths.append(str(path))
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="learning")
    print("\n".join(build(parser.parse_args().directory)))
