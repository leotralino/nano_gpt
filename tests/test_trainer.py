import torch
import torch.nn as nn

from nanogpt.trainer import approximate_loss


class _DummyDataLoader:
    def __init__(self):
        self.calls = []

    def load_batch(self, split="train"):
        self.calls.append(split)
        x = torch.zeros((2, 3), dtype=torch.long)
        y = torch.zeros((2, 3), dtype=torch.long)
        return x, y


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.train(True)

    def forward(self, x, y):
        batch, tokens = x.shape
        logits = torch.zeros((batch, tokens, 4), dtype=torch.float32)
        loss = torch.tensor(1.5, dtype=torch.float32)
        return logits, loss


def test_approximate_loss_returns_train_and_val_and_restores_train_mode():
    model = _DummyModel()
    dl = _DummyDataLoader()

    out = approximate_loss(model, dl)

    assert set(out.keys()) == {"train", "val"}
    assert torch.isclose(out["train"], torch.tensor(1.5))
    assert torch.isclose(out["val"], torch.tensor(1.5))
    assert model.training is True
    assert dl.calls.count("train") == 100
    assert dl.calls.count("val") == 100
