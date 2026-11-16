"""P2-02 standalone block reduction acceptance; no PyTorch dependency."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


_KERNELS = Path(__file__).resolve().parent / "kernels"
_SIZES = [1, 255, 256, 257, 1003, 65537]
_TOOLS = ("memcheck", "racecheck", "synccheck")


def _parse_result(text):
    result = json.loads(text)
    if (result["practice"] != "P2-02" or result["threads_per_block"] != 256
            or result["patterns"] != 3 or [row["n"] for row in result["cases"]] != _SIZES):
        raise ValueError("Unexpected reduction harness protocol.")
    for row in result["cases"]:
        counts = [row[key] for key in (
            "partial_mismatches", "total_mismatches", "guard_mismatches", "input_mismatches",
        )]
        if (any(type(count) is not int or count < 0 for count in counts)
                or row["blocks"] != (row["n"] + 255) // 256
                or row["status"] != ("passed" if all(count == 0 for count in counts)
                                     else "failed")):
            raise ValueError("Inconsistent reduction case evidence.")
    passed = sum(row["status"] == "passed" for row in result["cases"])
    if (result["passed"] != passed or result["failed"] != len(_SIZES) - passed
            or result["status"] != ("passed" if passed == len(_SIZES) else "failed")):
        raise ValueError("Inconsistent reduction summary.")
    return result


def _clean_summary(tool, log):
    pattern = (r"RACECHECK SUMMARY:\s*0 hazards? displayed \(0 errors, 0 warnings\)"
               if tool == "racecheck" else r"ERROR SUMMARY:\s*0 errors")
    return re.search(pattern, log) is not None


def evaluate_reduction(arch, candidate=None, sanitize=False):
    if not re.fullmatch(r"sm_[0-9]+[af]?", arch):
        raise ValueError("Pass an explicit CUDA architecture such as sm_121.")
    source = (_KERNELS / "block_sum.cu") if candidate is None else Path(candidate)
    source = source.resolve()
    if source.suffix != ".cu" or not source.is_file():
        raise ValueError("Candidate must be an existing trusted local .cu kernel.")
    compiler = shutil.which("nvcc")
    if compiler is None:
        raise RuntimeError("nvcc is unavailable; no CPU fallback.")
    sanitizer = shutil.which("compute-sanitizer") if sanitize else None
    if sanitize and sanitizer is None:
        raise RuntimeError("Compute Sanitizer explicitly requested but unavailable.")
    harness = _KERNELS / "block_sum_check.cu"
    report = {
        "practice": "P2-02", "implementation_role": (
            "reference_baseline" if candidate is None else "candidate_file"
        ),
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
        "device": "cuda", "architecture": arch, "dtype": "float32",
        "partial_atol": 1e-4, "partial_rtol": 1e-5,
        "final_merge": "CPU FP64; tolerance is the sum of per-block tolerances",
        "compiler": subprocess.run(
            [compiler, "--version"], capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "GPU block partials and CPU final sum; no performance claim.",
    }
    with tempfile.TemporaryDirectory(prefix="practice-reduction-") as directory:
        binary = Path(directory) / "block_sum_check"
        command = [compiler, "-arch=" + arch, "-O2", "-lineinfo",
                   str(harness), str(source), "-o", str(binary)]
        compiled = subprocess.run(command, capture_output=True, text=True, timeout=120)
        report["compile"] = {
            "command": command, "returncode": compiled.returncode,
            "stdout": compiled.stdout, "stderr": compiled.stderr,
        }
        if compiled.returncode:
            return {**report, "status": "error", "stage": "compile"}
        executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
        report["execution"] = {
            "returncode": executed.returncode, "stdout": executed.stdout, "stderr": executed.stderr,
        }
        try:
            report.update(_parse_result(executed.stdout))
        except (ValueError, KeyError, TypeError):
            return {**report, "status": "error", "stage": "execute",
                    "error": "No valid reduction evidence; inspect execution stdout/stderr."}
        if executed.returncode:
            report["status"] = "failed"
        # Check values first, then run sanitizers sequentially to avoid GPU pressure.
        report["sanitizers_requested"] = sanitize
        report["sanitizers"] = []
        if report["status"] != "passed" or not sanitize:
            return report
        for tool in _TOOLS:
            log_path = Path(directory) / (tool + ".log")
            command = [sanitizer, "--tool", tool, "--error-exitcode", "99",
                       "--log-file", str(log_path), str(binary)]
            checked = subprocess.run(command, capture_output=True, text=True, timeout=120)
            log = log_path.read_text() if log_path.exists() else ""
            evidence = {
                "tool": tool, "command": command, "returncode": checked.returncode,
                "stdout": checked.stdout, "stderr": checked.stderr, "log": log,
                "clean_summary": _clean_summary(tool, log),
            }
            try:
                evidence["harness_status"] = _parse_result(checked.stdout)["status"]
            except (ValueError, KeyError, TypeError):
                evidence["harness_status"] = "error"
            report["sanitizers"].append(evidence)
            if (checked.returncode or not evidence["clean_summary"]
                    or evidence["harness_status"] != "passed"):
                report.update(status="failed", error=f"{tool} did not pass.")
                break
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--candidate", type=Path, help="Trusted .cu source exporting block_sum.")
    parser.add_argument("--sanitize", action="store_true",
                        help="Require memcheck, racecheck and synccheck in sequence.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or Path("logs/practice") / (
        "p2-02-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    if output.exists():
        parser.error("Use a fresh report path; existing evidence will not be overwritten.")
    try:
        report = evaluate_reduction(args.arch, args.candidate, args.sanitize)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        report = {"practice": "P2-02", "status": "error", "device": "cuda",
                  "error": f"{type(error).__name__}: {error}"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Report:", output)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
