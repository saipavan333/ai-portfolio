# Project 13 - Train a Reasoning Model with RL (GRPO) - Frontier Project

> **New here? Start with [LESSON.md](LESSON.md)** - it teaches the concepts from scratch, then read the heavily-commented, runnable code in this folder.
> Teach a small model to reason step-by-step on verifiable tasks (math/logic/code) using reinforcement learning with verifiable rewards (GRPO).
**Phase 5: Agentic AI & Multimodal**  |  **Difficulty:** Expert / Stretch  |  **Est. time:** Week 21-22 (~35 hrs)

---

## Why (the point of this project)
Reasoning models (DeepSeek-R1 and successors) are the headline AI story of 2025-2026 - 'a shift from fluent output to structured reasoning.' Doing even a small RL reasoning fine-tune puts you in rare company and shows you understand the current frontier, not just last year's playbook. This is your 'I track the bleeding edge' project.

## What you will build
A small open base model trained with GRPO (Group Relative Policy Optimization) or similar RLVR on a task with automatically checkable answers (e.g. grade-school math, a small code task, or a puzzle), where the reward = 'is the final answer correct + well formatted,' producing visible chain-of-thought and improved accuracy.

## Key concepts (learn these as you go)
| Concept | What it means |
|---|---|
| **Reasoning models & chain-of-thought** | Models trained to 'think' in intermediate steps before answering, boosting accuracy on hard tasks. |
| **RL with verifiable rewards (RLVR)** | Use an automatic checker (answer matches, tests pass) as the reward signal - no human labels needed. |
| **GRPO** | A simplified, critic-free policy-optimization method popularized for reasoning training; efficient on modest hardware. |
| **Reward design & reward hacking** | Rewards must capture true correctness + format; models will exploit loopholes (you'll see it). |
| **Sample efficiency & evaluation** | Track pass@1/accuracy on held-out problems; watch for instability and length blowup. |

## How - step by step
1. Pick a task with automatic verification (e.g. GSM8K-style math, a constrained code task, or a logic puzzle).
2. Write a robust reward function (parse the model's final answer; check correctness + format).
3. Use a small base model + a GRPO trainer (TRL's GRPO or an open implementation) with LoRA to fit your GPU.
4. Train on Kaggle or a rented GPU; log reward, accuracy, and response length to W&B.
5. Inspect samples over time: does visible reasoning emerge? watch for reward hacking.
6. Evaluate pass@1 on held-out problems vs the base model.
7. Document compute, instability you hit, and what the model learned (and gamed).
8. Push the adapter + a short report; this is a talking-point project in interviews.

## Tech stack
TRL (GRPO), transformers, PEFT/LoRA, a verifiable task dataset, W&B, GPU (Kaggle/cloud)

## Where it lives (your tools)
| Tool | How you use it here |
|---|---|
| **Hugging Face** | Base model + dataset + adapter + report. |
| **Kaggle** | GPU for the RL run (keep the model small). |
| **Weights & Biases** | Track reward, accuracy, and response-length dynamics. |
| **GitHub** | Reward function, training script, eval, and your write-up. |

## Portfolio artifact
HF adapter + W&B report + GitHub repo + a blog post. Resume line: 'Trained a small reasoning model with GRPO/RLVR, improving verifiable-task accuracy from X to Y.'

## Definition of Done
- [ ] A working automatic reward function on a verifiable task.
- [ ] GRPO training run completes and improves held-out accuracy over the base model.
- [ ] W&B logs reward + accuracy + response length over training.
- [ ] You identify at least one reward-hacking behavior and discuss it.
- [ ] A clear write-up of method, compute, and results.

## Stretch goals
- Add format + length rewards and study their effect on the chain-of-thought.
- Compare GRPO vs simple rejection-sampling fine-tuning.
- Distill the reasoning behavior into a smaller model.

## Resources
- [TRL GRPO trainer](https://huggingface.co/docs/trl/grpo_trainer)
- [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948)
- [Hugging Face Open-R1 project](https://github.com/huggingface/open-r1)

## Results (fill this in as you build)
- **Headline metric:** _e.g. AUC / F1 / accuracy / p95 latency_
- **Artifact links:** GitHub _ | HF _ | Kaggle _ | W&B _ | Demo _
- **One thing that broke and how I fixed it:** _..._
- **Build-in-public post:** _link_
