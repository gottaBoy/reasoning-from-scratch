"""Explain ordinary dense GQA projection shapes and memory without loading weights."""

from math import prod


def inspect_config(cfg, batch=1, tokens=8, weight_bytes=2, cache_bytes=2, score_bytes=2):
    names = ("emb_dim", "hidden_dim", "n_heads", "n_kv_groups", "head_dim", "n_layers")
    for name in names:
        value = cfg.get(name)
        if name == "head_dim" and value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Positive integer {name} required.")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
           for v in (batch, tokens, weight_bytes, cache_bytes, score_bytes)):
        raise ValueError("Positive integer sizes required.")
    e, f, hq, hkv, layers = (cfg[name] for name in names if name != "head_dim")
    if hq % hkv:
        raise ValueError("Query heads must be divisible by KV heads.")
    d = cfg.get("head_dim")
    if d is None:
        if e % hq:
            raise ValueError("This repository requires E divisible by Hq for inferred D.")
        d = e // hq
    shapes = {
        "W_query": [hq * d, e],
        "W_key": [hkv * d, e],
        "W_value": [hkv * d, e],
        "out_proj": [e, hq * d],
        "gate_proj": [f, e],
        "up_proj": [f, e],
        "down_proj": [e, f],
    }
    counts = {name: prod(shape) for name, shape in shapes.items()}
    projection_parameters = sum(counts.values())
    kv_per_layer = 2 * batch * tokens * hkv * d * cache_bytes
    return {
        "dimensions": {"E": e, "F": f, "Hq": hq, "Hkv": hkv, "D": d, "L": layers},
        "head_dim_source": "explicit" if cfg.get("head_dim") is not None else "E/Hq repository default",
        "sharing_ratio": hq // hkv,
        "weight_shapes_out_in": shapes,
        "parameters": counts,
        "projection_parameters_per_layer": projection_parameters,
        "projection_weight_bytes_all_layers": layers * projection_parameters * weight_bytes,
        "compact_k_shape": [batch, hkv, tokens, d],
        "compact_v_shape": [batch, hkv, tokens, d],
        "kv_bytes_per_layer": kv_per_layer,
        "kv_bytes_all_layers": layers * kv_per_layer,
        "prefill_score_bytes_one_layer": batch * hq * tokens * tokens * score_bytes,
        "element_bytes": {"weight": weight_bytes, "cache": cache_bytes, "score": score_bytes},
        "scope": (
            "Dense GQA, equal Q/K/V head_dim, no projection bias; excludes norms, embeddings, "
            "allocator, gradients/optimizer, working buffers and paged overhead. Score need not "
            "be fully materialized; actual score dtype may differ from weight/cache dtype."
        ),
    }
