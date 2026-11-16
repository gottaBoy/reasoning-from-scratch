"""P2-01 standalone CUDA acceptance; run on a host with nvcc, without PyTorch."""

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


def _parse_result(text):
    result = json.loads(text)
    cases = result["cases"]
    if (result["practice"] != "P2-01" or [case["n"] for case in cases] != _SIZES
            or result["threads_per_block"] != 256):
        raise ValueError("CUDA harness returned an unexpected protocol.")
    for case in cases:
        counts = [case[key] for key in (
            "value_mismatches", "guard_mismatches", "input_mismatches",
        )]
        if (any(type(count) is not int or count < 0 for count in counts)
                or case["blocks"] != (case["n"] + 255) // 256
                or case["status"] != ("passed" if all(count == 0 for count in counts)
                                     else "failed")):
            raise ValueError("CUDA harness case evidence is inconsistent.")
    passed = sum(case["status"] == "passed" for case in cases)
    if (result["passed"] != passed or result["failed"] != len(cases) - passed
            or result["status"] != ("passed" if passed == len(cases) else "failed")):
        raise ValueError("CUDA harness summary is inconsistent.")
    return result


def evaluate_vector_add(arch, candidate=None, sanitize=False):
    if not re.fullmatch(r"sm_[0-9]+[af]?", arch):
        raise ValueError("Pass an explicit CUDA architecture such as sm_121.")
    source = (_KERNELS / "vector_add.cu") if candidate is None else Path(candidate)
    source = source.resolve()
    if source.suffix != ".cu" or not source.is_file():
        raise ValueError("Candidate must be an existing trusted local .cu kernel.")
    compiler = shutil.which("nvcc")
    if compiler is None:
        raise RuntimeError("nvcc is unavailable; no CPU fallback.")
    sanitizer = shutil.which("compute-sanitizer") if sanitize else None
    if sanitize and sanitizer is None:
        raise RuntimeError("Compute Sanitizer explicitly requested but unavailable.")
    harness = _KERNELS / "vector_add_check.cu"
    version = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=True, timeout=30,
    ).stdout.strip()
    report = {
        "practice": "P2-01", "implementation_role": (
            "reference_baseline" if candidate is None else "candidate_file"
        ),
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
        "device": "cuda", "architecture": arch, "dtype": "float32", "compiler": version,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CUDA vector add correctness, input preservation and output guards; no timing.",
    }
    with tempfile.TemporaryDirectory(prefix="practice-vector-add-") as directory:
        binary = Path(directory) / "vector_add_check"
        command = [compiler, "-arch=" + arch, "-O2", "-lineinfo",
                   str(harness), str(source), "-o", str(binary)]
        compiled = subprocess.run(command, capture_output=True, text=True, timeout=120)
        report["compile"] = {
            "command": command, "returncode": compiled.returncode,
            "stdout": compiled.stdout, "stderr": compiled.stderr,
        }
        if compiled.returncode:
            return {**report, "status": "error", "stage": "compile"}
        run_command = [str(binary)]
        sanitizer_log = Path(directory) / "memcheck.log"
        if sanitize:
            run_command = [
                sanitizer, "--tool", "memcheck", "--error-exitcode", "99",
                "--log-file", str(sanitizer_log), str(binary),
            ]
        executed = subprocess.run(run_command, capture_output=True, text=True, timeout=120)
        report["execution"] = {
            "command": run_command, "returncode": executed.returncode,
            "stdout": executed.stdout, "stderr": executed.stderr,
        }
        if sanitize:
            log = sanitizer_log.read_text() if sanitizer_log.exists() else ""
            report["sanitizer"] = {
                "tool": "memcheck", "returncode": executed.returncode,
                "log": log, "clean_memcheck": "ERROR SUMMARY: 0 errors" in log,
            }
        try:
            result = _parse_result(executed.stdout)
        except (ValueError, TypeError, KeyError):
            return {**report, "status": "error", "stage": "execute",
                    "error": "CUDA execution did not produce valid harness evidence."}
        report.update(result)
        if executed.returncode and result["status"] == "passed":
            report.update(status="failed", error="CUDA process or sanitizer failed.")
        if sanitize and not report["sanitizer"]["clean_memcheck"]:
            report.update(status="failed", error="No clean Compute Sanitizer summary.")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, help="Actual target architecture, e.g. sm_121.")
    parser.add_argument("--candidate", type=Path, help="Trusted kernel source, not a main program.")
    parser.add_argument("--sanitize", action="store_true", help="Require Compute Sanitizer memcheck.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or Path("logs/practice") / (
        "p2-01-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    if output.exists():
        parser.error("Use a fresh report path; existing evidence will not be overwritten.")
    try:
        report = evaluate_vector_add(args.arch, args.candidate, args.sanitize)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        report = {
            "practice": "P2-01", "status": "error", "device": "cuda",
            "error": f"{type(error).__name__}: {error}",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Report:", output)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
