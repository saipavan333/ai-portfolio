# Lesson 13 - Reasoning Models & Reinforcement Learning (GRPO)

> Read with `src/grpo_demo.py` open; run it (`python -m src.grpo_demo`). See
> `notebooks/13_reasoning_model_grpo.ipynb` for the reward + reward-hacking diagrams.

## 0. What you'll be able to do after this
- Explain **RL with verifiable rewards (RLVR)** and **GRPO** in plain terms.
- Run a toy GRPO trainer that learns to reason, and see **reward hacking** happen.
- Read the production TRL `GRPOTrainer` code (`src/grpo_train.py`).

## 1. The big picture (why this project exists)
Reasoning models (DeepSeek-R1 and successors) are the 2025-2026 headline: models trained to **think
step-by-step**. The training trick is RL where the reward is simply "is the final answer correct?" -
checkable automatically, no human labels. That one signal is enough to teach reasoning.

## 2. Foundations from scratch (the basics)
- **Policy:** the model's strategy for producing an answer (here, a probability over a few
  strategies: guess / short reasoning / step-by-step).
- **Verifiable reward:** +1 if the parsed final answer is correct. Perfect for maths/code.
- **GRPO:** for each problem, sample a **group** of answers, score them, and compute each one's
  **advantage = reward - group mean**. Increase the probability of above-average answers. Being
  group-relative means no separate value network - efficient and popular for reasoning.
  `train()` implements exactly this; the demo's policy converges on **step_by_step** (correctness 0.95).
- **Reward hacking:** models optimise what you reward. The demo cranks the length bonus, and the
  policy learns to be long-but-wrong (`padded_wrong`) - reward looks high, true correctness drops to
  0.20 (asserted). Verifiable correctness must dominate the reward.

## 3. How the pieces fit - the flow
```
        (concept - runs on CPU)                          (production - grpo_train.py)
  sample group -> verifiable reward -> advantage     model samples answers -> reward_correct() parses
   -> push policy toward above-average               & checks #### answer -> GRPOTrainer updates
              (grpo_demo.py)                          (GSM8K maths; watch reward AND length)
```

## 4. Code reading order
1. `src/grpo_demo.py` - `train()` (the GRPO loop) and the reward-hacking comparison (run it).
2. `notebooks/13_reasoning_model_grpo.ipynb` - reward curve + hacking diagrams.
3. `src/grpo_train.py` - `reward_correct` + the TRL `GRPOTrainer` on GSM8K.

## 5. Common mistakes (and how to avoid them)
- **Reward that's gameable** -> the model hacks it; keep correctness dominant (the demo shows why).
- **Tiny group size** -> noisy advantages; use a reasonable group.
- **Not tracking response length** -> you miss reward hacking creeping in.
- **No held-out pass@1** -> you can't prove real improvement over the base model.

## 6. Check your understanding
1. Why are maths/code ideal tasks for RL with verifiable rewards?
2. What is an "advantage" in GRPO, and why group-relative?
3. In the demo, what made the policy start producing long-but-wrong answers?
4. What two things should you track to catch reward hacking?

## 7. Mini-glossary (full versions in /GLOSSARY.md)
reasoning model, chain-of-thought, RLVR, GRPO, policy, advantage, reward hacking, pass@1.

## 8. Going deeper
- TRL GRPO trainer: https://huggingface.co/docs/trl/grpo_trainer
- DeepSeek-R1 paper: https://arxiv.org/abs/2501.12948
