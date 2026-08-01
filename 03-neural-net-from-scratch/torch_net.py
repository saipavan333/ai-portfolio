"""
torch_net.py - the SAME network as numpy_net.py, but in PyTorch.

    python torch_net.py        (needs: pip install torch scikit-learn)

The point of this file is the contrast: PyTorch's autograd computes the gradients you wrote
by hand in numpy_net.py. Notice you NEVER write backward() yourself - loss.backward() builds
the gradients automatically from the forward computation graph. Everything you learned by
hand is still happening; PyTorch just does the bookkeeping.
"""
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # exactly the same shape as the NumPy version: 64 -> 64 -> 10
        self.net = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x)                     # note: no softmax here - CrossEntropyLoss adds it


def main():
    digits = load_digits()
    X = torch.tensor(digits.data / 16.0, dtype=torch.float32)
    y = torch.tensor(digits.target, dtype=torch.long)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)   # Adam = a smarter gradient-descent rule
    loss_fn = nn.CrossEntropyLoss()                      # softmax + cross-entropy in one

    for ep in range(60):
        net.train()
        opt.zero_grad()                 # clear old gradients
        logits = net(Xtr)               # forward
        loss = loss_fn(logits, ytr)     # how wrong are we
        loss.backward()                 # AUTOGRAD: compute all gradients (your backward(), automated)
        opt.step()                      # update weights
        if (ep + 1) % 10 == 0:
            acc = (net(Xte).argmax(1) == yte).float().mean().item()
            print(f"epoch {ep + 1:3d}  loss {loss.item():.3f}  test_acc {acc:.3f}")

    acc = (net(Xte).argmax(1) == yte).float().mean().item()
    print(f"FINAL test accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
