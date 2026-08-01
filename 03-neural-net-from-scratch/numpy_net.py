"""
numpy_net.py - a neural network with NOTHING but NumPy: forward pass + hand-written
backpropagation + gradient descent.

    python numpy_net.py

It trains a 2-layer MLP to recognise 8x8 handwritten digits (scikit-learn's load_digits, no
download) and prints test accuracy (expect ~97%). After reading this you will understand
exactly what PyTorch's .backward() does under the hood - the single best thing you can do to
stop treating deep learning as magic.

WHAT : implement neurons, layers, a loss, and the gradients by hand, then train with SGD.
WHY  : every later project uses PyTorch's autograd; doing it once by hand makes those projects
       debuggable instead of mysterious.
HOW  : forward (inputs -> probabilities) then backward (chain rule -> gradients) then step.
WHERE: this file is the whole project; gradient_check.py proves the backward() math is correct.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# === MATH HELPERS ==========================================================
def relu(z):
    # the nonlinearity: keep positives, zero out negatives. Without it, stacking linear
    # layers would collapse into a single linear layer (no extra power).
    return np.maximum(0.0, z)


def softmax(z):
    # turn raw scores into probabilities that sum to 1.
    # subtract the row max first for numerical stability (exp of big numbers overflows).
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y, k):
    o = np.zeros((y.size, k))
    o[np.arange(y.size), y] = 1.0
    return o


def cross_entropy(P, Y):
    # average negative log-probability of the correct class. 0 = perfect, grows as we're wrong.
    return -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))


class MLP:
    """Architecture: input -> Linear(W1,b1) -> ReLU -> Linear(W2,b2) -> Softmax."""

    def __init__(self, d_in, d_hidden, d_out, seed=0):
        rng = np.random.default_rng(seed)
        # He initialization (variance 2/fan_in) keeps signal magnitudes stable through ReLU.
        # Bad init is a top reason nets fail to train - this line matters more than it looks.
        self.W1 = rng.standard_normal((d_in, d_hidden)) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.standard_normal((d_hidden, d_out)) * np.sqrt(2.0 / d_hidden)
        self.b2 = np.zeros(d_out)

    # --- FORWARD: inputs -> predictions (cache values backprop will need) ---
    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1      # pre-activation of hidden layer (a weighted sum)
        self.A1 = relu(self.Z1)              # hidden activations
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.P = softmax(self.Z2)            # output: class probabilities
        return self.P

    # --- BACKWARD: chain rule -> gradient of the loss wrt every weight ------
    def backward(self, Y):
        n = Y.shape[0]
        # For softmax + cross-entropy, d(loss)/d(Z2) simplifies to (P - Y)/n. This elegant
        # result is WHY these two are always paired. Everything below is the chain rule.
        dZ2 = (self.P - Y) / n               # (n, d_out)
        dW2 = self.A1.T @ dZ2                # how output weights should change
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T                # push the error back to the hidden activations
        dZ1 = dA1 * (self.Z1 > 0)           # ReLU's derivative: 1 where input was positive, else 0
        dW1 = self.X.T @ dZ1                # how input weights should change
        db1 = dZ1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads, lr):
        # gradient descent: nudge each weight a little DOWNHILL on the loss surface.
        self.W1 -= lr * grads["W1"]; self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]; self.b2 -= lr * grads["b2"]

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


def main():
    # === STEP 1: data (8x8 digit images, flattened to 64 numbers) ==========
    digits = load_digits()
    X = digits.data / 16.0                    # scale pixel values to [0,1] - networks like small inputs
    y = digits.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    Ytr = one_hot(ytr, 10)

    # === STEP 2: model =====================================================
    net = MLP(d_in=64, d_hidden=64, d_out=10)

    # === STEP 3: training loop (mini-batch stochastic gradient descent) =====
    lr, epochs, batch = 0.5, 60, 64
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        order = rng.permutation(len(Xtr))     # reshuffle every epoch so batches vary
        for s in range(0, len(Xtr), batch):
            b = order[s:s + batch]
            net.forward(Xtr[b])               # 1) predict on this batch
            grads = net.backward(Ytr[b])      # 2) compute gradients by hand
            net.step(grads, lr)               # 3) update the weights
        if (ep + 1) % 10 == 0:
            loss = cross_entropy(net.forward(Xtr), Ytr)
            acc = (net.predict(Xte) == yte).mean()
            print(f"epoch {ep + 1:3d}  train_loss {loss:.3f}  test_acc {acc:.3f}")

    # === STEP 4: final score ==============================================
    print(f"FINAL test accuracy: {(net.predict(Xte) == yte).mean():.3f}")
    # YOUR TURN: add a second hidden layer; try lr=0.1 vs 1.0; add L2 weight decay.


if __name__ == "__main__":
    main()
