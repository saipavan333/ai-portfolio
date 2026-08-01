"""
self_attention.py - the core of every transformer, in pure NumPy (runnable, no GPU).

    python self_attention.py

WHAT : implements scaled dot-product self-attention with a causal mask, and a tiny character-level
       bigram language model that generates text - so 'attention' and 'next-token prediction' stop
       being abstract.
WHY  : every LLM (GPT, Claude, Llama) is attention stacked in layers. Build it once by hand.
HOW  : tokens -> Q,K,V -> scores=QK^T/sqrt(d) -> causal-mask the future -> softmax -> mix V.
WHERE: the concept layer. mini_gpt.py is the production layer (a trainable GPT in PyTorch).
       See notebooks/06_mini_gpt_from_scratch.ipynb for the attention/causal-mask heatmaps.
"""
from __future__ import annotations
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def self_attention(X, seed=0):
    """X: (T, d) token vectors. Returns (output, attention_weights)."""
    T, d = X.shape
    rng = np.random.default_rng(seed)
    Wq, Wk, Wv = (rng.standard_normal((d, d)) * 0.5 for _ in range(3))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = (Q @ K.T) / np.sqrt(d)                       # how much each token attends to each
    mask = np.triu(np.ones((T, T)), k=1).astype(bool)     # True above the diagonal = the future
    scores[mask] = -1e9                                   # causal mask: can't look ahead
    attn = softmax(scores, axis=-1)                       # each row sums to 1
    return attn @ V, attn


def demo_attention():
    word = "attention"
    rng = np.random.default_rng(0)
    X = rng.standard_normal((len(word), 16))              # pretend embeddings for each letter
    out, attn = self_attention(X)
    print(f"[attn] sequence length {len(word)} -> attention matrix {attn.shape}")
    assert np.allclose(attn.sum(axis=1), 1.0), "each attention row must sum to 1"
    assert np.allclose(np.triu(attn, k=1), 0.0), "causal mask must zero out the future"
    print("[attn] OK - rows sum to 1 and the future is masked (upper triangle = 0)")


def demo_language_model():
    corpus = ("the cat sat on the mat. the cat ran to the man. "
              "the dog sat on the log. the dog ran to the den. ") * 6
    chars = sorted(set(corpus)); stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}; V = len(chars)
    counts = np.ones((V, V))                              # add-one smoothing (no zero probabilities)
    for a, b in zip(corpus, corpus[1:]):
        counts[stoi[a], stoi[b]] += 1
    P = counts / counts.sum(axis=1, keepdims=True)        # P[i] = distribution of the next char
    rng = np.random.default_rng(0); i = stoi["t"]; out = ["t"]
    for _ in range(80):
        i = rng.choice(V, p=P[i]); out.append(itos[i])    # sample the next character
    print("[lm] bigram sample (1-char memory -> babble, which motivates attention):")
    print("    " + "".join(out))


if __name__ == "__main__":
    demo_attention()
    demo_language_model()
