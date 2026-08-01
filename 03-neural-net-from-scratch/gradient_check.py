"""
gradient_check.py - PROVE the hand-written backprop is correct.

    python gradient_check.py

WHY: backprop is easy to get subtly wrong (a transpose, a missing /n). We compare each
     analytic gradient from backward() to a NUMERICAL gradient computed by nudging one weight:
         numeric = (loss(w + eps) - loss(w - eps)) / (2 * eps)
     If analytic and numeric agree to ~1e-6, the math is right. This is exactly how engineers
     debug custom neural-network layers in the real world.
"""
import numpy as np
from numpy_net import MLP, one_hot, cross_entropy


def main():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((8, 64))
    Y = one_hot(rng.integers(0, 10, 8), 10)

    net = MLP(64, 16, 10, seed=2)
    net.forward(X)
    grads = net.backward(Y)                    # the analytic gradients we want to verify

    eps, worst = 1e-5, 0.0
    for name in ["W1", "b1", "W2", "b2"]:
        flat = getattr(net, name).ravel()      # a view: editing flat edits the real weight
        g = grads[name].ravel()
        for i in rng.choice(flat.size, size=min(20, flat.size), replace=False):
            orig = flat[i]
            flat[i] = orig + eps; lp = cross_entropy(net.forward(X), Y)
            flat[i] = orig - eps; lm = cross_entropy(net.forward(X), Y)
            flat[i] = orig                      # restore
            numeric = (lp - lm) / (2 * eps)
            rel = abs(numeric - g[i]) / max(1e-8, abs(numeric) + abs(g[i]))
            worst = max(worst, rel)
    print(f"worst relative error: {worst:.2e}  ->", "PASS" if worst < 1e-4 else "FAIL")


if __name__ == "__main__":
    main()
