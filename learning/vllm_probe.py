"""Read-only probe of a separately launched OpenAI-compatible vLLM endpoint."""

import argparse
import json
import os
from pathlib import Path
import time

import requests


def probe(base_url, timeout=30):
    base = base_url.rstrip("/")
    models = requests.get(f"{base}/models", timeout=timeout)
    models.raise_for_status()
    data = models.json()["data"]
    if not data:
        raise ValueError("Endpoint has no served models.")
    start = time.perf_counter()
    response = requests.post(
        f"{base}/completions",
        json={"model": data[0]["id"], "prompt": "The capital of France is",
              "max_tokens": 8, "temperature": 0},
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    return {"endpoint": base, "model": data[0]["id"], "elapsed_seconds": time.perf_counter() - start,
            "usage": result.get("usage"), "text": result["choices"][0]["text"],
            "scope": "non-streaming API integration only; no TTFT or kernel claims"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.base_url:
        parser.error("Supply a separately launched, authorized vLLM endpoint.")
    report = probe(args.base_url)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
