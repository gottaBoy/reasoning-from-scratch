"""Execute the generated notebooks on the current kernel environment, preserving reports."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/course-notebooks"))
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    summary = output / "summary.json"
    if summary.exists():
        raise FileExistsError("Use a fresh output directory to preserve earlier verification.")
    for day in range(6, 31):
        path = root / "learning" / f"day{day:02d}_course.ipynb"
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        print(f"Executing Day {day}...", flush=True)
        row = {"day": day, "executed": False}
        try:
            NotebookClient(notebook, timeout=args.timeout, kernel_name="python3",
                           resources={"metadata": {"path": str(root / "learning")}}).execute()
            report = json.loads((root / "logs" / f"day{day:02d}-course.json").read_text())
            row.update(executed=True, lab_status=report["status"])
        except Exception as error:
            row.update(lab_status="error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            nbformat.write(notebook, output / path.name)
            results.append(row)
            summary.write_text(json.dumps({
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "notebooks": results,
            }, indent=2) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
