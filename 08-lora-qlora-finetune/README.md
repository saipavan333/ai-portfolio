# Project 08 - Fine-tune an LLM with LoRA / QLoRA (PEFT)

> **New here? Start with [LESSON.md](LESSON.md)** - it teaches the concepts from scratch, then read the heavily-commented, runnable code in this folder.
> Adapt an open LLM to a specific task/domain/style on a single GPU using parameter-efficient fine-tuning, then publish the adapter to the Hub.
**Phase 4: LLMs, RAG & Fine-tuning**  |  **Difficulty:** Advanced  |  **Est. time:** Week 13-15 (~45 hrs)

---

## Why (the point of this project)
When prompting/RAG isn't enough (specialized format, domain tone, a skill the base model lacks), you fine-tune. LoRA/QLoRA make this possible on ONE consumer/Kaggle GPU instead of a cluster. 'LLM fine-tuning' is a top-3 demanded skill in 2026, and most engineers have never actually done it. You will.

## What you will build
A small open base model (1-8B, e.g. a Llama/Qwen/Gemma-class model) fine-tuned with LoRA/QLoRA on an instruction or domain dataset, evaluated against the base model, with the adapter pushed to the HF Hub and a before/after demo.

## Key concepts (learn these as you go)
| Concept | What it means |
|---|---|
| **Parameter-efficient fine-tuning (PEFT)** | Train a tiny set of new weights while freezing the base model - cheap, fast, shippable. |
| **LoRA** | Inject low-rank adapter matrices into attention/MLP layers; train only those. Tiny files, big effect. |
| **QLoRA & quantization** | Load the base model in 4-bit to slash memory, then train LoRA on top - fine-tune big models on small GPUs. |
| **Instruction tuning & chat templates** | Format data as instruction/response with the model's chat template so it learns to follow prompts. |
| **Catastrophic forgetting & eval** | Fine-tuning can degrade general ability; always evaluate against the base model on held-out tasks. |

## How - step by step
1. Pick a concrete task (e.g. convert SQL<->text, summarize tickets in your style, a domain Q/A). Define success.
2. Build/curate an instruction dataset (a few hundred to a few thousand examples); format with the chat template.
3. Load a small open base model in 4-bit (bitsandbytes); attach LoRA adapters with PEFT/TRL.
4. Fine-tune with TRL SFTTrainer on Kaggle GPU; track loss + samples in W&B.
5. Evaluate vs the base model on a held-out set (task metric + qualitative side-by-side).
6. Merge or keep the adapter; push to the HF Hub with a model card describing data + limits.
7. Build a small before/after demo (Space) showing the improvement.
8. Write up cost: GPU-hours, dataset size, and what moved the needle.

## Tech stack
transformers, PEFT, TRL, bitsandbytes, datasets, accelerate, Hugging Face Hub, W&B, Kaggle GPU

## Where it lives (your tools)
| Tool | How you use it here |
|---|---|
| **Hugging Face** | Base model + dataset + adapter hosting + demo Space. |
| **Kaggle** | Free GPU for QLoRA training (4-bit fits small models). |
| **Weights & Biases** | Track training + a sweep over rank/alpha/LR. |
| **GitHub** | Training scripts, eval harness, dataset card, results. |

## Portfolio artifact
HF adapter + before/after Space + GitHub repo + W&B report. Resume line: 'Fine-tuned an open LLM with QLoRA on a single GPU, improving task metric from X to Y over the base model.'

## Definition of Done
- [ ] Fine-tuned model beats the base model on your defined task metric.
- [ ] Trained with QLoRA (4-bit) on a single GPU; adapter pushed to the Hub.
- [ ] Held-out evaluation includes a base-vs-tuned comparison (quant + qualitative).
- [ ] W&B logs training + at least one hyperparameter sweep.
- [ ] Model card documents data, intended use, and limitations honestly.

## Stretch goals
- Add DPO/preference tuning on top of SFT and compare.
- Quantize the merged model to GGUF and run it locally with llama.cpp/Ollama.
- Measure forgetting on a general benchmark before/after.

## Resources
- [HF PEFT docs](https://huggingface.co/docs/peft)
- [TRL (SFT/DPO) docs](https://huggingface.co/docs/trl)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)

## Results (fill this in as you build)
- **Headline metric:** _e.g. AUC / F1 / accuracy / p95 latency_
- **Artifact links:** GitHub _ | HF _ | Kaggle _ | W&B _ | Demo _
- **One thing that broke and how I fixed it:** _..._
- **Build-in-public post:** _link_
