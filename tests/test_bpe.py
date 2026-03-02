import torch

from nanogpt.bpe import BPE


def test_bpe_forward_without_targets_returns_logits_and_no_loss():
    model = BPE(vocab_size=7)
    x = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)

    logits, loss = model(x)

    assert logits.shape == (2, 3, 7)
    assert loss is None


def test_bpe_forward_with_targets_returns_flat_logits_and_scalar_loss():
    torch.manual_seed(0)
    model = BPE(vocab_size=5)
    x = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long)
    y = torch.tensor([[2, 3, 4], [2, 1, 0]], dtype=torch.long)

    logits, loss = model(x, y)

    assert logits.shape == (6, 5)
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_bpe_generate_appends_tokens_and_preserves_prefix():
    torch.manual_seed(0)
    model = BPE(vocab_size=11)
    prefix = torch.tensor([[1, 4, 2], [3, 0, 7]], dtype=torch.long)

    out = model.generate(prefix, max_token=5)

    assert out.shape == (2, 8)
    assert torch.equal(out[:, :3], prefix)
    assert int(out.min()) >= 0
    assert int(out.max()) < 11
