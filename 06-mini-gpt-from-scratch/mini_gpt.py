"""
mini_gpt.py - PRODUCTION: a trainable GPT-style transformer in PyTorch (nanoGPT-style).

    python mini_gpt.py        # run on a Kaggle GPU; needs torch + a text file

WHAT : stacks the self-attention you built (self_attention.py) into transformer BLOCKS (attention +
       MLP + residual + layer-norm), adds token & positional embeddings, and trains to predict the
       next token. Generating text = predict next token, append, repeat.
WHY  : this IS a small GPT. Understanding it makes every large LLM legible.
HOW  : embed tokens+positions -> N decoder blocks (causal attention) -> predict next-token logits.
WHERE: the production layer. self_attention.py teaches the core operation underneath.

Requires: torch. Train on TinyShakespeare (a small Kaggle dataset).
"""
from __future__ import annotations


def build_model(vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=128):
    import torch, torch.nn as nn

    class Block(nn.Module):                                # one transformer decoder block
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
            self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
            self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                     nn.Linear(4 * n_embd, n_embd))
            self.register_buffer("mask", torch.triu(torch.ones(block_size, block_size), 1).bool())

        def forward(self, x):
            T = x.size(1)
            a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=self.mask[:T, :T])
            x = x + a                                      # residual connection
            return x + self.mlp(self.ln2(x))

    class MiniGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab_size, n_embd)    # token embeddings
            self.pos = nn.Embedding(block_size, n_embd)    # positional embeddings
            self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
            self.ln = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, vocab_size)

        def forward(self, idx):
            T = idx.size(1)
            pos = torch.arange(T, device=idx.device)
            x = self.tok(idx) + self.pos(pos)
            return self.head(self.ln(self.blocks(x)))      # logits over the vocabulary

    return MiniGPT()


def main():
    import torch, torch.nn.functional as F
    text = open("input.txt").read()                       # e.g. TinyShakespeare
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    block_size, batch = 128, 32
    model = build_model(len(chars), block_size=block_size)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def get_batch():
        ix = torch.randint(len(data) - block_size - 1, (batch,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])   # targets = next tokens
        return x, y

    for step in range(2000):
        x, y = get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.3f}")   # log to W&B in practice


if __name__ == "__main__":
    main()
