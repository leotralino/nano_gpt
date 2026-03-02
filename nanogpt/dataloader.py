import logging

import requests
import torch

from nanogpt.tokenizer import BaseTokenizer
from nanogpt.utils import setup_logging

setup_logging()


logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(
        self,
        urls: list[str] = [
            "https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt"
        ],
        seed: int = 42,
        train_val_split=0.8,
    ):
        torch.manual_seed(seed)
        self.train_val_split = train_val_split

        self.full_text = """"""

        for url in urls:
            raw_text = self.download(url)
            self.full_text += raw_text
            logger.info(f"loaded {url}.\nExample:\n\n{raw_text[:100]}")

    def encode_full_text(self, tokenizer: BaseTokenizer):
        encoded_text = tokenizer.encode(self.full_text)

        self.data = torch.tensor(encoded_text, dtype=torch.long)
        self.train = self.data[: int(self.train_val_split * self.size)]
        self.val = self.data[int(self.train_val_split * self.size) :]

    @property
    def size(self):
        return len(self.data)

    def download(self, url: str):
        response = requests.get(url)

        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch file {url}")

    def load_batch(self, split="train", context_length: int = 8, batch_size: int = 4):
        if split == "train":
            use_data = self.train
        else:
            use_data = self.val

        start_indices = torch.randint(
            0, len(use_data) - context_length, size=(batch_size,)
        )

        X = torch.stack([use_data[idx : idx + context_length] for idx in start_indices])

        y = torch.stack(
            [use_data[idx + 1 : idx + 1 + context_length] for idx in start_indices]
        )

        return X, y
