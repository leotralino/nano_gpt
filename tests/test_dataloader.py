import torch

import nanogpt.dataloader as dl_module
from nanogpt.dataloader import DataLoader
from nanogpt.tokenizer import CharTokenizer


class _DummyLogger:
    def info(self, _msg):
        return None


def test_dataloader_initialization_and_split(monkeypatch):
    monkeypatch.setattr(dl_module, "logger", _DummyLogger(), raising=False)
    monkeypatch.setattr(DataLoader, "download", lambda self, _url: "abca")

    tokenizer = CharTokenizer("abca")
    dl = DataLoader(urls=["u1", "u2"], train_val_split=0.75)
    dl.encode_full_text(tokenizer=tokenizer)

    assert dl.full_text == "abcaabca"
    assert dl.size == 8
    assert len(dl.train) == 6
    assert len(dl.val) == 2


def test_dataloader_load_batch_has_expected_shapes_and_shift(monkeypatch):
    monkeypatch.setattr(dl_module, "logger", _DummyLogger(), raising=False)
    monkeypatch.setattr(DataLoader, "download", lambda self, _url: "abcabcabcabc")

    tokenizer = CharTokenizer("abc")
    dl = DataLoader(urls=["u1"], train_val_split=0.8)
    dl.encode_full_text(tokenizer=tokenizer)

    x, y = dl.load_batch(split="train", context_length=4, batch_size=3)

    assert x.shape == (3, 4)
    assert y.shape == (3, 4)
    assert torch.equal(x[:, 1:], y[:, :-1])
