import logging

import torch

from nanogpt.utils import setup_logging

setup_logging()


logger = logging.getLogger(__name__)


@torch.no_grad()
def approximate_loss(model, data_loader):
    model.eval()
    out = {}
    for spl in ["train", "val"]:
        total_loss = torch.zeros(100)
        for i in range(100):
            xb, yb = data_loader.load_batch(split=spl)
            _logits, _loss = model(xb, yb)
            total_loss[i] = _loss.item()  # break from graph to save memory!!!
        out[spl] = float(total_loss.mean())

    model.train()
    return out


if __name__ == "__main__":
    from nanogpt.bpe import BPE
    from nanogpt.dataloader import DataLoader
    from nanogpt.tokenizer import CharTokenizer

    data_loader = DataLoader(
        urls=[
            "https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt"
        ]
    )
    tokenizer = CharTokenizer(full_text=data_loader.full_text)
    data_loader.encode_full_text(tokenizer=tokenizer)
    model = BPE(vocab_size=tokenizer.n_vocab)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)

    for step in range(10000):
        X, Y = data_loader.load_batch(split="train", context_length=8, batch_size=32)
        logits, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            out = approximate_loss(model, data_loader)
            logger.info(f"Step {step}; Approximated loss: {out}")

    output = model.generate(torch.tensor([[55], [49], [35]], dtype=torch.long), 100)
    out_s = [tokenizer.decode(i) for i in output]
    logger.info(f"Sample generation:\n-------\n{''.join(out_s)}\n-------")
