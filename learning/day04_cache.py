"""Compare full Qwen3 forward with prefill plus token-by-token cached decoding."""

import argparse
import json
from pathlib import Path

import torch

from learning.day03_shapes import TINY_CONFIG
from reasoning_from_scratch.qwen3 import KVCache, QWEN_CONFIG_06_B, Qwen3Model


@torch.inference_mode()
def compare_cache(model, input_ids, prefill_length, atol=1e-4, rtol=1e-4, strict=True):
    if input_ids.ndim != 2 or not 0 < prefill_length < input_ids.shape[1]:
        raise ValueError("Use [B,T] input and 0 < prefill_length < T.")
    if input_ids.shape[0] == 0:
        raise ValueError("Batch must not be empty.")
    model.eval()
    model.reset_kv_cache()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    phases = []
    prefix_cache = None
    batch = input_ids.shape[0]
    head_dim = model.trf_blocks[0].att.head_dim

    def snapshot(phase, tokens):
        nonlocal prefix_cache
        shapes = []
        for layer, (keys, values) in enumerate(cache.get_all()):
            expected = (batch, model.cfg["n_kv_groups"], model.current_pos, head_dim)
            if tuple(keys.shape) != expected or tuple(values.shape) != expected:
                raise AssertionError(f"Unexpected cache shape in layer {layer}.")
            if prefix_cache is not None:
                old_k, old_v = prefix_cache[layer]
                torch.testing.assert_close(keys[:, :, :old_k.shape[2]], old_k, atol=0, rtol=0)
                torch.testing.assert_close(values[:, :, :old_v.shape[2]], old_v, atol=0, rtol=0)
            shapes.append({"layer": layer, "K": list(keys.shape), "V": list(values.shape)})
        if prefix_cache is None:
            prefix_cache = [(k.clone(), v.clone()) for k, v in cache.get_all()]
        phases.append({
            "phase": phase,
            "input_tokens": tokens,
            "next_position": model.current_pos,
            "mask_shape": [1, 1, tokens, model.current_pos],
            "cache": shapes,
        })

    try:
        full = model(input_ids)
        prefill = model(input_ids[:, :prefill_length], cache=cache)
        snapshot("prefill", prefill_length)
        parts = [prefill]
        for position in range(prefill_length, input_ids.shape[1]):
            parts.append(model(input_ids[:, position:position + 1], cache=cache))
            if model.current_pos != position + 1:
                raise AssertionError("Model RoPE position did not advance with the cache.")
            snapshot(f"decode_{position}", 1)
        cached = torch.cat(parts, dim=1)
        if not torch.isfinite(full).all() or not torch.isfinite(cached).all():
            raise AssertionError("Non-finite logits.")

        reference = full[:, prefill_length:].float()
        actual = cached[:, prefill_length:].float()
        difference = (reference - actual).abs()
        close = torch.isclose(actual, reference, atol=atol, rtol=rtol)
        checked = reference.numel()
        passed = bool(close.all().item())
        last_full, last_cached = full[:, -1].float(), cached[:, -1].float()
        report = {
            "torch": torch.__version__,
            "device": str(input_ids.device),
            "dtype": str(full.dtype),
            "input_ids": input_ids.cpu().tolist(),
            "prefill_length": prefill_length,
            "full_logits_shape": list(full.shape),
            "cached_logits_shape": list(cached.shape),
            "last_decode_logits_shape": list(parts[-1].shape),
            "phases": phases,
            "cache_structure": "passed",
            "prefix_cache_preserved": "passed",
            "tolerance": {"atol": atol, "rtol": rtol},
            "logits_close": passed,
            "decoded_logits_compared": checked,
            "outside_tolerance": checked - int(close.sum().item()),
            "max_absolute_error": difference.max().item(),
            "mean_absolute_error": difference.mean().item(),
            "last_position_max_absolute_error": (last_full - last_cached).abs().max().item(),
            "last_top1_full": last_full.argmax(-1).cpu().tolist(),
            "last_top1_cached": last_cached.argmax(-1).cpu().tolist(),
            "strict": strict,
        }
        if strict:
            torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
        return report
    finally:
        # Position belongs to the model, while K/V belong to the separate cache object.
        model.reset_kv_cache()
        cache.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("tiny", "qwen3"), default="tiny")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; no automatic CPU fallback.")
    cfg = dict(TINY_CONFIG if args.config == "tiny" else QWEN_CONFIG_06_B)
    cfg["dtype"] = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    torch.manual_seed(123)
    model = Qwen3Model(cfg)
    if args.weights:
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        del state
    model.to(args.device)
    ids = torch.arange(8, device=args.device).unsqueeze(0)
    # BF16 is an observation against the same threshold, never an automatic pass.
    report = compare_cache(model, ids, 7, strict=args.dtype == "fp32")
    report["config"] = args.config
    report["weights"] = str(args.weights) if args.weights else "random"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
