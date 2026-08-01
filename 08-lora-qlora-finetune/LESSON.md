# Lesson 08 - Fine-tuning LLMs with LoRA / QLoRA

> Read with `src/lora_demo.py` open; run `notebooks/08_lora_qlora_finetuning.ipynb` for the
> parameter-count, low-rank, and quantization diagrams.

## 0. What you'll be able to do after this
- Explain **PEFT**, **LoRA**, **quantization**, and **QLoRA** - and why they fit a big model on one GPU.
- Run the NumPy proof (`python -m src.lora_demo`) and read the production trainer (`qlora_finetune.py`).

## 1. The big picture (why this project exists)
When prompting and RAG aren't enough (special format, domain tone, a skill the base model lacks),
you **fine-tune**. Full fine-tuning of a billion-parameter model needs a cluster. LoRA + QLoRA let
you do it on a single free GPU - and 'LLM fine-tuning' is a top-demanded 2026 skill most engineers
have never actually done.

## 2. Foundations from scratch (the basics)
- **Full fine-tuning** updates *every* weight - billions of numbers, huge memory.
- **PEFT** freezes the giant model and trains a *tiny* set of new weights instead.
- **LoRA:** for a weight matrix W (e.g. 4096x4096), instead of a full update it learns two skinny
  matrices A (4096 x r) and B (r x 4096); the update is `A @ B`. The demo shows r=8 trains **256x
  fewer** numbers. This works because fine-tuning updates are approximately **low-rank** - the demo
  reconstructs a matrix with rank-5 at ~4% error.
- **Quantization:** store each weight in fewer bits. The demo shows 4-bit = **8x** less memory than
  fp32, with small rounding error.
- **QLoRA:** load the frozen base model in **4-bit** (saves memory) and train **LoRA** adapters on
  top. That's the whole trick. `src/qlora_finetune.py` does it with PEFT + TRL.

## 3. How the pieces fit - the flow
```
        (concept - runs on CPU)                          (production - qlora_finetune.py)
  count params: full d*d  vs  LoRA 2*d*r           base model loaded in 4-bit (quantized)
  low-rank: A@B approximates a full update                 + LoRA adapters (trainable)
  quantize: fp32 -> 4-bit (8x smaller)                            |
              (lora_demo.py)                          SFTTrainer on instruction data -> push adapter
```

## 4. Code reading order
1. `src/lora_demo.py` - param counts, low-rank reconstruction, quantization (run it first).
2. `notebooks/08_lora_qlora_finetuning.ipynb` - the rendered diagrams.
3. `src/qlora_finetune.py` - 4-bit load + LoRA config + SFTTrainer + push_to_hub.

## 5. Common mistakes (and how to avoid them)
- **Rank too small** -> underfits; too large -> wastes the savings. Pick r where the demo's error curve flattens.
- **Skipping evaluation vs the base model** -> you can't claim the fine-tune helped.
- **Catastrophic forgetting** -> fine-tuning can hurt general ability; re-check on general tasks.
- **Wrong chat template** -> format data with the model's template or it won't learn to follow prompts.

## 6. Check your understanding
1. Why does training A (d x r) + B (r x d) save so much vs a full d x d update?
2. What property of fine-tuning updates makes LoRA work?
3. What does the 'Q' in QLoRA add, and why does it matter for memory?
4. Why must you always compare the fine-tuned model to the base model?

## 7. Mini-glossary (full versions in /GLOSSARY.md)
fine-tuning, PEFT, LoRA, rank, quantization, 4-bit/nf4, QLoRA, adapter, instruction tuning,
catastrophic forgetting.

## 8. Going deeper
- Hugging Face PEFT: https://huggingface.co/docs/peft
- QLoRA paper: https://arxiv.org/abs/2305.14314
