"""
grpo_train.py - PRODUCTION: train a reasoning model with TRL's GRPO on a verifiable task.

    python -m src.grpo_train        # needs a GPU + trl/transformers

WHAT : trains a small model on maths problems where the reward = 'is the final answer correct?'
       (parsed and checked automatically). The model learns to show working and answer correctly.
WHY  : same idea as grpo_demo.py - group sampling + verifiable rewards - on a real model.
HOW  : define reward_correct(...) -> GRPOTrainer with group sampling -> train.
WHERE: production layer; grpo_demo.py explains WHY it works. Track reward AND length (catch hacking).

Requires: trl, transformers, datasets (GPU).
"""
from __future__ import annotations
import re


def reward_correct(completions, answer, **kwargs):
    """Verifiable reward: 1.0 if the parsed final answer matches, else 0.0 (no human labels)."""
    out = []
    for c, a in zip(completions, answer):
        m = re.search(r"####\s*(-?\d+)", c)            # parse the model's '#### <answer>'
        out.append(1.0 if (m and m.group(1) == str(a)) else 0.0)
    return out


def main():
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer
    data = load_dataset("openai/gsm8k", "main", split="train")   # maths with checkable answers
    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=[reward_correct],
        args=GRPOConfig(num_generations=8, per_device_train_batch_size=8, report_to="wandb"),
        train_dataset=data)
    trainer.train()
    print("done - report pass@1 (base vs trained) and watch response length for reward hacking")


if __name__ == "__main__":
    main()
