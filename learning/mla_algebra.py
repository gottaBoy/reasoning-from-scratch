"""Factorized multi-head KV plus decoupled RoPE: algebra lab, not DeepSeek deployment."""

import math

import torch

from learning.mini_transformer import rope


@torch.inference_mode()
def compare_paths(tokens=7, heads=4, rank=3, content_dim=4, value_dim=3, rope_dim=2,
                  device="cpu"):
    dimensions = (tokens, heads, rank, content_dim, value_dim, rope_dim)
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in dimensions):
        raise ValueError("All dimensions must be positive integers.")
    if rope_dim % 2:
        raise ValueError("RoPE dimension must be even.")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Explicit CUDA requested but unavailable.")
    generator = torch.Generator(device=device).manual_seed(123)

    def sample(*shape):
        return torch.randn(
            *shape, generator=generator, device=device, dtype=torch.float32
        ) * 0.3

    hidden_dim = 12
    hidden = sample(tokens, hidden_dim)
    down = sample(rank, hidden_dim)
    up_key = sample(heads, content_dim, rank)
    up_value = sample(heads, value_dim, rank)
    query_content = sample(heads, content_dim)
    query_rope = rope(sample(1, heads, 1, rope_dim), start=tokens - 1)[0, :, 0]
    key_rope = rope(sample(1, 1, tokens, rope_dim))[0, 0]
    latent = hidden @ down.T

    # Expanded reference: materialize all historical keys and values.
    key_content = torch.einsum("hdr,tr->htd", up_key, latent)
    values = torch.einsum("hvr,tr->htv", up_value, latent)
    expanded_keys = torch.cat((key_content, key_rope[None].expand(heads, -1, -1)), -1)
    queries = torch.cat((query_content, query_rope), -1)
    scale = math.sqrt(content_dim + rope_dim)
    expanded_scores = torch.einsum("hd,htd->ht", queries, expanded_keys) / scale
    expanded_weights = expanded_scores.softmax(-1)
    expanded_output = torch.einsum("ht,htv->hv", expanded_weights, values)

    # Absorbed reference: query uses the key up-projection; history remains latent.
    latent_query = torch.einsum("hd,hdr->hr", query_content, up_key)
    content_scores = latent_query @ latent.T
    position_scores = query_rope @ key_rope.T
    latent_scores = (content_scores + position_scores) / scale
    latent_weights = latent_scores.softmax(-1)
    weighted_latent = latent_weights @ latent
    latent_output = torch.einsum("hr,hvr->hv", weighted_latent, up_value)
    torch.testing.assert_close(latent_scores, expanded_scores, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(latent_weights, expanded_weights, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(latent_output, expanded_output, atol=1e-6, rtol=1e-5)
    expanded_elements = expanded_keys.numel() + values.numel()
    compact_elements = latent.numel() + key_rope.numel()
    return {
        "scope": (
            "Same factorized random weights, one last-position query over a prefix. "
            "Includes a separate rotary key branch; omits query compression, normalization, "
            "output projection, training, serving, and optimized GPU kernels."
        ),
        "device": str(device),
        "heads": heads,
        "tokens": tokens,
        "latent_shape": list(latent.shape),
        "position_key_shape": list(key_rope.shape),
        "expanded_key_shape": list(expanded_keys.shape),
        "expanded_value_shape": list(values.shape),
        "latent_query_shape": list(latent_query.shape),
        "output_shape": list(latent_output.shape),
        "expanded_elements_per_token": expanded_elements // tokens,
        "compact_elements_per_token": compact_elements // tokens,
        "compact_bytes": compact_elements * latent.element_size(),
        "expanded_bytes": expanded_elements * latent.element_size(),
        "score_max_error": (latent_scores - expanded_scores).abs().max().item(),
        "output_max_error": (latent_output - expanded_output).abs().max().item(),
        "expanded_head0_output": expanded_output[0].cpu().tolist(),
        "latent_head0_output": latent_output[0].cpu().tolist(),
        "score_alignment": "passed",
        "output_alignment": "passed",
        "numerical_tolerance": {"atol": 1e-6, "rtol": 1e-5},
        "claims": "Algebra and array-size comparison only; not quality or speedup.",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(compare_paths(), indent=2))
