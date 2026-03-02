import torch
from nanogpt.utils import setup_logging
setup_logging()

import logging
logger = logging.getLogger(__name__)


@torch.no_grad()
def approximate_loss(model, dl):
    model.eval()
    out = {}
    for spl in ['train', 'val']:
        total_loss = torch.zeros(100)
        for i in range(100):
            xb, yb = dl.load_batch(split=spl)
            _logits, _loss = model(xb, yb)
            total_loss[i] = _loss.item() # break from graph to save memory!!!
        out[spl] = total_loss.mean()

    model.train()
    return out

if __name__ == "__main__":
    from nanogpt.tokenizer import CharTokenizer
    from nanogpt.dataloader import DataLoader
    from nanogpt.bpe import BPE

    dl = DataLoader(urls=["https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt"])
    tokenizer = CharTokenizer(full_text=dl.full_text)
    dl.encode_full_text(tokenizer=tokenizer)
    bpe = BPE(vocab_size=tokenizer.n_vocab)

    optimizer = torch.optim.AdamW(params=bpe.parameters(), lr=1e-3)


    for step in range(10000):
        X, Y = dl.load_batch(split='train', context_length=8, batch_size=32)
        logits, loss = bpe(X, Y)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()
        optimizer.step()

    output = bpe.generate(torch.tensor([[55], [49], [35]], dtype=torch.long), 100)
    out_s = [tokenizer.decode(i) for i in output]
    for i in out_s:
        print(i)