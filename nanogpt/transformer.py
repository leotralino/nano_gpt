import logging

import torch
import torch.functional as F
import torch.nn as nn
from utils import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

CONFIG = {
    "batch_size": 64,  # how many independent sequences will we process in parallel?
    "block_size": 256,  # what is the maximum context length for predictions?
    "max_iters": 5000,
    "eval_interval": 500,
    "learning_rate": 3e-4,
    "eval_iters": 200,
    "n_embed": 384,  # token embedding dim
    "n_head": 6,
    "n_layer": 6,
    "dropout": 0.2,
}

device = "cuda" if torch.cuda.is_available() else "mps"


class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(in_features=CONFIG["n_embed"], out_features=head_size, bias=False)
        self.query = nn.Linear(in_features=CONFIG["n_embed"], out_features=head_size, bias=False)
        self.value = nn.Linear(in_features=CONFIG["n_embed"], out_features=head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(CONFIG["block_size"], CONFIG["batch_size"])))
        self.dropout = nn.Dropout(CONFIG["dropout"])

    def forward(self, x):
        """
        Input x has size (B,T,C)
        """
        B, T, C = x.shape

        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        wei = q @ k.transpose(-2, -1) / torch.sqrt(C)  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(torch.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        return out


class MultiheadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.all_attentions = [SelfAttention(head_size) for _ in range(n_head)]
        self.proj = nn.Embedding(CONFIG["n_embed"], CONFIG["n_embed"])

    def forward(self, x):
        out = torch.concat([att(x) for att in self.all_attentions], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),  # 4 comes from attention paper
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),  # projection layer (for residual connection)
            nn.Dropout(CONFIG["dropout"]),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, n_head, n_embed):
        super().__init__()
        head_size = n_embed // n_head
        self.att = (MultiheadAttention(n_head, head_size),)
        self.ff = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.att(self.ln1(x))  # residual connection
        x = x + self.ff(self.ln2(x))  # residual connection
        return x
