from abc import ABC, abstractmethod



import torch
class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, s: str) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, vec: torch.Tensor) -> str:
        pass


class CharTokenizer(BaseTokenizer):
    def __init__(self, full_text: str):
        chars = sorted(set(full_text))

        self.char_to_idx = {}
        self.idx_to_char = {}

        for idx, char in enumerate(chars):
            self.char_to_idx[char] = int(idx)
            self.idx_to_char[int(idx)] = char

        self.add_extra_char("[UNK]")

        assert len(self.char_to_idx) == len(self.idx_to_char)

    @property
    def n_vocab(self):
        return len(self.char_to_idx)

    def has(self, s: str):
        return s in self.char_to_idx

    def add_extra_char(self, s: str):
        if s in self.char_to_idx:
            return

        curr_size = self.n_vocab
        self.char_to_idx[s] = curr_size
        self.idx_to_char[curr_size] = s

    def encode(self, s: str):
        return torch.tensor(
            [
                self.char_to_idx[char] if self.has(char) else self.char_to_idx["[UNK]"]
                for char in s
            ],
            dtype=torch.long,
        )

    def decode(self, vec: torch.Tensor):
        return "".join([self.idx_to_char[int(idx)] for idx in vec])
