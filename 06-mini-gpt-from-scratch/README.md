# Project 06 - Build a mini-GPT (Transformer from Scratch)

> **New here? Start with [LESSON.md](LESSON.md)** - it teaches the concepts from scratch, then read the heavily-commented, runnable code in this folder.
> Implement a small GPT-style decoder (attention, positional encoding, blocks) and train it to generate text. This demystifies every LLM you will ever use.
**Phase 3: NLP & Transformers**  |  **Difficulty:** Intermediate-Advanced  |  **Est. time:** Week 9-11 (~45 hrs)

---

## Why (the point of this project)
Everything in Phase 4-6 is built on the transformer decoder. If you build a small one yourself - self-attention, multi-head, residual blocks, a training loop - LLMs stop being magic and become debuggable engineering. This is the single best investment for truly understanding (and interviewing about) modern AI.

## What you will build
A compact GPT (a few million parameters) implemented in PyTorch and trained on a small corpus (e.g. TinyShakespeare or a domain text you like) to generate coherent text. Modeled on Karpathy's nanoGPT but written and understood by you, with your own notes.

## Key concepts (learn these as you go)
| Concept | What it means |
|---|---|
| **Self-attention** | Each token attends to others via query/key/value projections, mixing context. The core operation of transformers. |
| **Multi-head attention** | Several attention 'heads' learn different relationships in parallel, then combine. |
| **Positional encoding** | Transformers have no inherent order, so position info is injected (learned or sinusoidal). |
| **Decoder block & causal masking** | Residual + LayerNorm + attention + MLP; a causal mask stops a token from seeing the future (needed for generation). |
| **Language modeling objective** | Predict the next token; cross-entropy over the vocabulary. Sampling (temperature/top-k) controls generation. |
| **Scaling intuition** | More data/params/compute -> lower loss. You will feel this directly on a small budget. |

## How - step by step
1. Prepare a small text corpus; build a character or BPE tokenizer.
2. Implement a single self-attention head, then multi-head attention.
3. Stack into a decoder block (attention + MLP + residual + LayerNorm) and a full model with positional embeddings.
4. Write the training loop with causal masking and cross-entropy; log loss to W&B.
5. Train on Kaggle GPU; watch validation loss; add a sampling function (temperature, top-k).
6. Generate text at several checkpoints to see quality improve as loss drops.
7. Document the architecture with a diagram and your own explanation of attention.
8. Compare your loss/samples to nanoGPT to sanity-check correctness.

## Tech stack
PyTorch, tiktoken/custom tokenizer, W&B, Kaggle GPU

## Where it lives (your tools)
| Tool | How you use it here |
|---|---|
| **Kaggle** | Free GPU to train the model for a few hours. |
| **GitHub** | Your annotated implementation + architecture diagram + sample outputs. |
| **Weights & Biases** | Track loss curves and log generated-text samples per checkpoint. |
| **Hugging Face** | Optional: push the toy model + a Space that generates text live. |

## Portfolio artifact
GitHub repo (deep-understanding showpiece) + W&B report with samples. Resume line: 'Implemented and trained a GPT-style transformer from scratch (attention, causal masking, sampling).'

## Definition of Done
- [ ] Model trains; validation loss decreases steadily and samples become coherent.
- [ ] Self-attention and causal masking implemented by you (not imported).
- [ ] W&B logs loss curves + text samples across checkpoints.
- [ ] README explains attention in your own words with a diagram.
- [ ] Generation supports temperature and top-k sampling.

## Stretch goals
- Add RoPE positional encoding and compare.
- Scale up params/data and plot a mini scaling curve.
- Fine-tune your mini-GPT on a niche corpus and show style transfer.

## Resources
- [Karpathy - Let's build GPT (video + nanoGPT)](https://github.com/karpathy/nanoGPT)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Results (fill this in as you build)
- **Headline metric:** _e.g. AUC / F1 / accuracy / p95 latency_
- **Artifact links:** GitHub _ | HF _ | Kaggle _ | W&B _ | Demo _
- **One thing that broke and how I fixed it:** _..._
- **Build-in-public post:** _link_
