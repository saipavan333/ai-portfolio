"""
grpo_demo.py - WHY reinforcement learning makes a model reason, in pure NumPy (runnable).

    python -m src.grpo_demo

WHAT : a tiny GRPO-style trainer on a toy task with VERIFIABLE rewards. It learns to prefer the
       'step-by-step' strategy under a good reward, and demonstrates REWARD HACKING under a bad one.
WHY  : reasoning models (DeepSeek-R1 and successors) are trained with RL on checkable answers.
       Building the loop shows why 'reward correctness' teaches reasoning - and how a bad reward backfires.
HOW  : sample a GROUP of answers, reward them, compute advantage = reward - group mean, push the
       policy toward above-average strategies (group-relative => no critic network).
WHERE: the concept layer. grpo_train.py is production (TRL GRPOTrainer on a real model).
       See notebooks/13_reasoning_model_grpo.ipynb for the reward + hacking diagrams.
"""
from __future__ import annotations
import numpy as np

# strategy:        (name,            P(correct),  'length'/thoroughness)
STRAT = [("guess", 0.20, 0.1), ("short_reason", 0.60, 0.5),
         ("step_by_step", 0.95, 1.0), ("padded_wrong", 0.20, 3.0)]
NAMES = [s[0] for s in STRAT]
ACC = np.array([s[1] for s in STRAT])
LEN = np.array([s[2] for s in STRAT])


def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def train(format_weight: float, iters: int = 400, group: int = 16, lr: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    logits = np.zeros(len(STRAT))
    for _ in range(iters):
        p = softmax(logits)
        g = rng.choice(len(STRAT), size=group, p=p)         # sample a GROUP of answers
        correct = (rng.random(group) < ACC[g]).astype(float)  # VERIFIABLE reward (right/wrong)
        reward = correct + format_weight * LEN[g]            # + a length/format bonus
        advantage = reward - reward.mean()                   # GRPO: relative to the GROUP mean
        for s, a in zip(g, advantage):
            logits[s] += lr * a                              # push toward better-than-average
    return softmax(logits)


def main():
    good = train(format_weight=0.05)    # a sensible reward (correctness dominates)
    hack = train(format_weight=0.7)     # over-rewards length -> gets gamed
    print("good reward -> policy:", {n: round(float(p), 2) for n, p in zip(NAMES, good)})
    print("hacked reward -> policy:", {n: round(float(p), 2) for n, p in zip(NAMES, hack)})
    print(f"true correctness: good={float((good*ACC).sum()):.2f}  hacked={float((hack*ACC).sum()):.2f}")
    assert good[2] > 0.8, "with a good reward, RL should learn step_by_step"
    assert hack[3] > 0.5, "with a length-heavy reward, the model should hack via padded_wrong"
    assert (good * ACC).sum() > (hack * ACC).sum(), "reward hacking should lower true correctness"
    print("[grpo] OK - RL learned to reason; a bad reward got hacked (correctness fell)")


if __name__ == "__main__":
    main()
