"""Trace the repository's first Qwen3 block and verify its tensor flow."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from reasoning_from_scratch.qwen3 import QWEN_CONFIG_06_B, Qwen3Model, apply_rope


TINY_CONFIG = {
    "vocab_size": 64,
    "context_length": 32,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "hidden_dim": 96,
    "head_dim": 16,
    "qk_norm": True,
    "n_kv_groups": 2,
    "rope_base": 1_000_000.0,
    "dtype": torch.float32,
}


@torch.inference_mode()
def trace_model(model, input_ids):
    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("Use [batch, sequence] input with at least two tokens.")
    model.eval()
    saved = {}
    handles = []
    names = (
        "tok_emb",
        "trf_blocks.0.norm1",
        "trf_blocks.0.att",
        "trf_blocks.0.norm2",
        "trf_blocks.0.ff",
        "trf_blocks.0",
        "final_norm",
        "out_head",
    )

    def capture(name):
        def hook(module, inputs, output):
            saved[name] = (inputs, output)
        return hook

    try:
        for name in names:
            handles.append(model.get_submodule(name).register_forward_hook(capture(name)))
        logits = model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    cfg = model.cfg
    batch, length = input_ids.shape
    heads = cfg["n_heads"]
    groups = cfg["n_kv_groups"]
    dim = model.trf_blocks[0].att.head_dim
    emb = cfg["emb_dim"]
    shape_rows = []

    def record(name, tensor, expected):
        if tuple(tensor.shape) != tuple(expected):
            raise AssertionError(f"{name}: expected {expected}, got {tuple(tensor.shape)}")
        shape_rows.append({
            "name": name,
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        })

    fp32 = logits.dtype == torch.float32
    atol, rtol = (1e-5, 1e-5) if fp32 else (2e-2, 2e-2)
    checks = {}
    errors = {}

    def compare(name, actual, expected):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        checks[name] = "passed"
        errors[name] = (actual.float() - expected.float()).abs().max().item()

    x0 = saved["tok_emb"][1]
    normalized = saved["trf_blocks.0.norm1"][1]
    (att_input, mask, cos, sin), (att_output, cache) = saved["trf_blocks.0.att"]
    att = model.trf_blocks[0].att
    record("input_ids", input_ids, (batch, length))
    record("embedding / block_input", x0, (batch, length, emb))
    record("norm1", normalized, (batch, length, emb))
    compare("norm1_feeds_attention", att_input, normalized)

    # Module hooks cannot see view/matmul locals; replay this branch with the same weights.
    q = att.W_query(att_input)
    k = att.W_key(att_input)
    v = att.W_value(att_input)
    record("Q projection", q, (batch, length, heads * dim))
    record("K projection", k, (batch, length, groups * dim))
    record("V projection", v, (batch, length, groups * dim))
    q = q.view(batch, length, heads, dim)
    k = k.view(batch, length, groups, dim)
    v = v.view(batch, length, groups, dim)
    record("Q view [B,T,H,D]", q, (batch, length, heads, dim))
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    record("Q transpose [B,H,T,D]", q, (batch, heads, length, dim))
    record("K compact [B,KV,T,D]", k, (batch, groups, length, dim))
    record("V compact [B,KV,T,D]", v, (batch, groups, length, dim))
    if att.q_norm is not None:
        q = att.q_norm(q)
    if att.k_norm is not None:
        k = att.k_norm(k)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    record("Q after RoPE", q, (batch, heads, length, dim))
    record("K after RoPE", k, (batch, groups, length, dim))
    record("returned K (before expansion)", cache[0], (batch, groups, length, dim))
    record("returned V (before expansion)", cache[1], (batch, groups, length, dim))
    compare("compact_k_matches_returned_cache", k, cache[0])
    compare("compact_v_matches_returned_cache", v, cache[1])

    k = k.repeat_interleave(att.group_size, dim=1)
    v = v.repeat_interleave(att.group_size, dim=1)
    record("K expanded", k, (batch, heads, length, dim))
    record("V expanded", v, (batch, heads, length, dim))
    record("causal mask (True = blocked)", mask, (1, 1, length, length))
    expected_mask = torch.ones(length, length, device=mask.device, dtype=torch.bool).triu(1)
    torch.testing.assert_close(mask[0, 0], expected_mask)
    checks["mask_is_strict_upper_triangle"] = "passed"
    scores = q @ k.transpose(2, 3)
    record("Q @ K.T", scores, (batch, heads, length, length))
    weights = torch.softmax(scores.masked_fill(mask, -torch.inf) / dim**0.5, dim=-1)
    record("attention weights", weights, (batch, heads, length, length))
    compare("attention_rows_sum_to_one", weights.float().sum(-1),
            torch.ones(batch, heads, length, device=weights.device))
    if torch.count_nonzero(weights.masked_select(mask.expand_as(weights))).item() != 0:
        raise AssertionError("Future positions received nonzero attention weight.")
    checks["future_attention_weights_are_zero"] = "passed"
    context = weights @ v
    record("weights @ V", context, (batch, heads, length, dim))
    merged = context.transpose(1, 2).reshape(batch, length, heads * dim)
    record("merged heads", merged, (batch, length, heads * dim))
    replayed = att.out_proj(merged)
    record("attention output projection", att_output, (batch, length, emb))
    compare("attention_replay_matches_hook", replayed, att_output)

    x1 = saved["trf_blocks.0.norm2"][0][0]
    record("after attention residual", x1, (batch, length, emb))
    compare("attention_residual", x1, x0 + att_output)
    ff_input = saved["trf_blocks.0.ff"][0][0]
    compare("norm2_feeds_ffn", ff_input, saved["trf_blocks.0.norm2"][1])
    record("norm2 / FFN input", ff_input, (batch, length, emb))
    ff = model.trf_blocks[0].ff
    gate, up = ff.fc1(ff_input), ff.fc2(ff_input)
    record("FFN gate projection", gate, (batch, length, cfg["hidden_dim"]))
    record("FFN up projection", up, (batch, length, cfg["hidden_dim"]))
    gated = F.silu(gate) * up
    record("FFN gated features", gated, (batch, length, cfg["hidden_dim"]))
    ff_output = saved["trf_blocks.0.ff"][1]
    record("FFN down projection", ff_output, (batch, length, emb))
    compare("ffn_replay_matches_hook", ff.fc3(gated), ff_output)
    block_output = saved["trf_blocks.0"][1][0]
    record("first block output", block_output, (batch, length, emb))
    compare("ffn_residual", block_output, x1 + ff_output)
    record("final norm (after all blocks)", saved["final_norm"][1], (batch, length, emb))
    record("logits", logits, (batch, length, cfg["vocab_size"]))

    # Keep the prefix fixed and replace every suffix ID without changing sequence length.
    split = length // 2
    changed_ids = input_ids.clone()
    changed_ids[:, split:] = (changed_ids[:, split:] + 1) % cfg["vocab_size"]
    changed_logits = model(changed_ids)
    compare("future_tokens_do_not_change_prefix_logits",
            logits[:, :split], changed_logits[:, :split])
    if not torch.isfinite(logits).all():
        raise AssertionError("Logits contain non-finite values.")
    checks["logits_are_finite"] = "passed"
    return {
        "weights": "existing model parameters; no weight updates",
        "dimensions": {
            "B": batch, "T": length, "E": emb, "H": heads, "KV": groups, "D": dim,
            "layers": cfg["n_layers"], "vocab_size": cfg["vocab_size"],
        },
        "input_ids": input_ids.cpu().tolist(),
        "changed_input_ids": changed_ids.cpu().tolist(),
        "unchanged_prefix_length": split,
        "mask_1_means_blocked": mask[0, 0].int().cpu().tolist(),
        "tolerances": {"atol": atol, "rtol": rtol},
        "shapes": shape_rows,
        "checks": checks,
        "max_absolute_errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("tiny", "qwen3"), default="tiny")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--weights", type=Path, help="Optional local checkpoint; never downloaded.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = dict(TINY_CONFIG if args.config == "tiny" else QWEN_CONFIG_06_B)
    if not 2 <= args.seq_len <= cfg["context_length"] or args.batch_size < 1:
        parser.error("Use batch-size >= 1 and 2 <= seq-len <= context_length.")
    if args.config == "qwen3" and (args.seq_len > 32 or args.batch_size > 2):
        parser.error("This tracing lab limits Qwen3 to batch <= 2 and seq-len <= 32.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; no automatic CPU fallback.")
    torch.manual_seed(123)
    model = Qwen3Model(cfg)
    if args.weights:
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        del state
    model.to(args.device).eval()
    ids = torch.arange(args.batch_size * args.seq_len, device=args.device)
    ids = ids.reshape(args.batch_size, args.seq_len) % cfg["vocab_size"]
    report = trace_model(model, ids)
    report["config"] = args.config
    report["checkpoint"] = str(args.weights) if args.weights else None
    print("Random weights (not a text-quality test)" if not args.weights else args.weights)
    print("Dimensions:", report["dimensions"])
    for row in report["shapes"]:
        print(f'{row["name"]:38} {str(row["shape"]):24} {row["dtype"]} {row["device"]}')
    print("\nCausal mask: 1 = blocked")
    for row in report["mask_1_means_blocked"]:
        print(" ".join(map(str, row)))
    print("\nChecks:")
    for name, status in report["checks"].items():
        print(f"  {status}: {name}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()
