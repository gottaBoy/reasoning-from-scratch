"""Execute the generated notebooks on the current kernel environment, preserving reports."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/course-notebooks"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--days", nargs="+", type=int, choices=range(6, 31),
                        default=list(range(6, 31)),
                        help="Selected days; default preserves the original Day 6-30 run.")
    args = parser.parse_args()
    if len(args.days) != len(set(args.days)):
        parser.error("Duplicate days are not allowed.")
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    summary = output / "summary.json"
    if summary.exists():
        raise FileExistsError("Use a fresh output directory to preserve earlier verification.")
    for day in args.days:
        path = root / "learning" / f"day{day:02d}_course.ipynb"
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        print(f"Executing Day {day}...", flush=True)
        row = {"day": day, "executed": False}
        daily_report = root / "logs" / f"day{day:02d}-course.json"
        if daily_report.is_file():
            backup = output / "prior-reports" / daily_report.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(daily_report, backup)
        try:
            NotebookClient(notebook, timeout=args.timeout, kernel_name="python3",
                           resources={"metadata": {"path": str(root / "learning")}}).execute()
            report = json.loads(daily_report.read_text())
            if report["day"] != day:
                raise ValueError("Report day does not match executed notebook.")
            shutil.copy2(daily_report, output / daily_report.name)
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
