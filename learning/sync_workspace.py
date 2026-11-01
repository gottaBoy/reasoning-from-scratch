"""Sync course sources to running Spark Jupyter; back up every differing existing file."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Default only lists source files.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = [p for folder in ("learning", "docs") for p in (root / folder).rglob("*")
               if p.is_file() and p.suffix in (".md", ".py", ".ipynb", ".cu")
               and "__pycache__" not in p.parts and not p.is_symlink()]
    sources += [root / name for name in (
        "LLM_30_DAY_STUDY_PLAN.md", "DGX_SPARK.md", "DGX_SPARK_VS_RTX3090.md", "README.md"
    )]
    sources += list((root / "tests").glob("test_learning*.py"))
    sources += [root / "tests/test_curriculum.py", root / "tests/test_course_tools.py",
                root / "tests/test_config_math.py", root / "tests/test_triton_ir.py",
                root / "tests/test_mla_algebra.py"]
    compose = ["docker", "compose", "-f", str(root / "docker-compose.spark.yml")]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    remote_code = (
        "from pathlib import Path; import hashlib,shutil,sys\n"
        "relative,digest,stamp=sys.argv[1:]\n"
        "target=Path('/workspace')/relative\n"
        "if any(p.is_symlink() for p in (target,*target.parents)):\n"
        " raise RuntimeError('Refusing symlink destination')\n"
        "if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==digest:\n"
        " print('unchanged'); raise SystemExit(0)\n"
        "if target.exists():\n"
        " backup=Path('/workspace/logs/course-sync-backup')/stamp/relative\n"
        " backup.parent.mkdir(parents=True,exist_ok=True)\n"
        " shutil.copy2(target,backup)\n"
        "target.parent.mkdir(parents=True,exist_ok=True)\n"
        "print('copy')\n"
    )
    for path in sorted(sources):
        relative = path.relative_to(root).as_posix()
        if not args.apply:
            print(relative)
            continue
        check = subprocess.run(
            compose + ["exec", "-T", "reasoning", "/opt/venv/bin/python", "-c",
                       remote_code, relative, hashlib.sha256(path.read_bytes()).hexdigest(), stamp],
            check=True, capture_output=True, text=True,
        )
        if check.stdout.strip() == "copy":
            subprocess.run(compose + ["cp", str(path), f"reasoning:/workspace/{relative}"], check=True,
                           capture_output=True, text=True)
            print("Synced:", relative, flush=True)
    if args.apply:
        print(f"Backups of changed existing files: /workspace/logs/course-sync-backup/{stamp}")
        print("No destination files were deleted. Notebooks are not forced to reload in the browser.")


if __name__ == "__main__":
    main()
