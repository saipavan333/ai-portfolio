# Lesson 06 - Mini-GPT: Build a Transformer From Scratch

> Read with `self_attention.py` open; run `notebooks/06_mini_gpt_from_scratch.ipynb` for the
> attention + causal-mask heatmaps. This is the architecture behind every modern LLM.

## 0. What you'll be able to do after this
- Explain **self-attention**, **causal masking**, and **next-token prediction**.
- Run self-attention + a tiny language model by hand (`python self_attention.py`).
- Read a real, trainable **mini-GPT** (`mini_gpt.py`) and recognise every part.

## 1. The big picture (why this project exists)
Everything in Projects 07-15 (RAG, fine-tuning, agents) runs on pretrained transformers. If you
build the core - self-attention - once by hand, LLMs stop being magic and become engineering you
can reason about and debug.

## 2. Foundations from scratch (the basics)
**Tokens.** Text is chopped into pieces (here, single characters), each mapped to an id.

**Language modelling.** The model predicts the **next token** from the previous ones. Generating
text = predict next token, append it, repeat. That's the whole objective.

**Self-attention (the core).** For each token we make a **Query**, and compare it to every token's
**Key** to decide how much to attend to each; we then mix their **Values** by those weights. This
is how a token gathers context from the rest of the sequence. In `self_attention.py`:
```
scores = Q @ K.T / sqrt(d)      # token-to-token relevance
scores[future] = -1e9           # causal mask: you may not see the future
attn = softmax(scores)          # each row sums to 1
out  = attn @ V                 # each token = weighted mix of values
```

**Causal mask.** When generating left-to-right, a token must only attend to itself and **earlier**
tokens. The mask zeroes the "future" (upper triangle), which the demo asserts.

**Positional information.** Attention has no built-in order, so we add **positional embeddings** so
the model knows token positions.

**A transformer block.** attention + a small MLP, each wrapped with a **residual connection** and
**layer-norm**. Stack N of these, add token+positional embeddings and a final linear head -> a GPT.
`mini_gpt.py` is exactly this.

**Why the bigram babbles.** `self_attention.py`'s bigram model has only **1 token of memory**, so
it produces local-but-incoherent text. That limitation is precisely what attention (long context)
fixes - which is the whole point of transformers.

## 3. How the pieces fit - the flow
```
        (concept - runs on CPU)                         (production - needs a GPU)
  letters -> Q,K,V -> causal attention -> mix      tokens+positions -> N transformer blocks
            (self_attention.py)                              (mini_gpt.py)
  bigram model -> sample next char (babble)        -> next-token logits -> train -> generate
```

## 4. Code reading order
1. `self_attention.py` - `self_attention()` and the two demos (run it first).
2. `notebooks/06_mini_gpt_from_scratch.ipynb` - attention + causal-mask heatmaps.
3. `mini_gpt.py` - `Block` (attention+MLP+residual+LN) and `MiniGPT` (embeddings+blocks+head).

## 5. Common mistakes (and how to avoid them)
- **No causal mask** -> the model "cheats" by seeing the answer; generation breaks.
- **Forgetting positional embeddings** -> the model can't tell word order.
- **No residual/layer-norm** -> deep transformers fail to train.
- **Learning rate too high** -> loss diverges to NaN.

## 6. Check your understanding
1. In one sentence, what does self-attention compute for each token?
2. Why must the upper triangle of the attention matrix be zero during generation?
3. What does a residual connection do for a deep transformer?
4. Why does a 1-token-memory bigram babble, and how does attention fix it?

## 7. Mini-glossary (full versions in /GLOSSARY.md)
token, language modelling, self-attention, query/key/value, causal mask, positional embedding,
transformer block, residual, layer-norm, sampling/temperature.

## 8. Going deeper
- Karpathy, "Let's build GPT" + nanoGPT: https://github.com/karpathy/nanoGPT
- "Attention Is All You Need": https://arxiv.org/abs/1706.03762
