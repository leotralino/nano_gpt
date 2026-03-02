import torch

from nanogpt.tokenizer import CharTokenizer


def test_char_tokenizer_roundtrip_for_known_tokens():
    tokenizer = CharTokenizer("abca")

    encoded = tokenizer.encode("caba")
    decoded = tokenizer.decode(encoded)

    assert isinstance(encoded, torch.Tensor)
    assert decoded == "caba"
    assert tokenizer.n_vocab == 4  # a, b, c + [UNK]


def test_char_tokenizer_uses_unk_for_missing_token():
    tokenizer = CharTokenizer("abc")

    encoded = tokenizer.encode("az")
    unk_idx = tokenizer.char_to_idx["[UNK]"]

    assert encoded.tolist() == [tokenizer.char_to_idx["a"], unk_idx]
    assert tokenizer.decode(encoded) == "a[UNK]"
