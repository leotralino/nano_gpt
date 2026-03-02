import torch
import torch.nn as nn
from torch.nn import functional as F


class BPE(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # row idx interpret as logits for next token given current token at idx
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.token_embedding(idx)  # (Batch, token, channel)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(input=logits, target=targets)

        return logits, loss

    def generate(self, idx, max_token):
        for _ in range(max_token):
            logits, loss = self(idx)

            proba = F.softmax(logits[:, -1, :], dim=-1)  # B, C

            next_idx = torch.multinomial(proba, num_samples=1)  # gives token id per row

            idx = torch.cat([idx, next_idx], dim=1)  # B, T+1

        return idx
