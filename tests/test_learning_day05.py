import pytest
import torch

from learning.day05_moe import ToyMoE


@pytest.mark.parametrize("top_k", [1, 2, 4], ids=["top1", "top2", "top4"])
def test_sparse_dispatch_matches_full_expert_reference(top_k):
    torch.manual_seed(7)
    moe = ToyMoE(top_k=top_k)
    x = torch.randn(12, 8)
    calls = [0] * moe.num_experts
    handles = []

    def capture(index):
        def hook(module, inputs, output):
            calls[index] += 1
        return hook

    for i, expert in enumerate(moe.experts):
        handles.append(expert.register_forward_hook(capture(i)))
    output, routing = moe(x)
    for handle in handles:
        handle.remove()
    assert sum(routing["counts"]).item() == x.shape[0] * top_k
    assert calls == (routing["counts"] > 0).int().tolist()
    torch.testing.assert_close(
        routing["weights"].sum(-1), torch.ones(x.shape[0])
    )
    all_outputs = torch.stack([expert(x) for expert in moe.experts], dim=1)
    selected = all_outputs.gather(
        1, routing["expert_ids"].unsqueeze(-1).expand(-1, -1, x.shape[-1])
    )
    reference = (selected * routing["weights"].unsqueeze(-1)).sum(1)
    torch.testing.assert_close(output, reference, atol=1e-6, rtol=1e-5)
    output.square().mean().backward()
    assert torch.isfinite(moe.router.weight.grad).all()


def test_collapsed_routing_counts_tokens_not_probabilities():
    moe = ToyMoE(top_k=2)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.bias.copy_(torch.tensor([4.0, 3.0, 0.0, -1.0]))
    output, routing = moe(torch.randn(32, 8))
    assert routing["counts"].tolist() == [32, 32, 0, 0]
    assert output.shape == (32, 8)


@pytest.mark.parametrize("k", [0, 5])
def test_invalid_top_k(k):
    with pytest.raises(ValueError):
        ToyMoE(top_k=k)
