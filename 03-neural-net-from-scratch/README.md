# Project 03 - Neural Network from Scratch, then PyTorch

> **New here? Start with [LESSON.md](LESSON.md)** - it teaches the concepts from scratch, then read the heavily-commented code in `src/`.
> Build a working neural net with only NumPy (forward + backprop by hand), then rebuild it in PyTorch. After this, deep learning is no longer a black box.
**Phase 2: Deep Learning Core**  |  **Difficulty:** Intermediate  |  **Est. time:** Week 3-4 (~35 hrs)

---

## Why (the point of this project)
You will use PyTorch for the rest of the program. Spending one project understanding what `.backward()` actually does - gradients, the chain rule, optimization - pays off every time training misbehaves later. Doing it once by hand is the difference between copying tutorials and debugging real models.

## What you will build
Two implementations of the same classifier (start with MNIST or Fashion-MNIST): (1) a 2-layer MLP in pure NumPy with hand-written forward pass, loss, and backpropagation; (2) the same network in PyTorch using autograd, an optimizer, and a DataLoader. Compare that they learn the same thing.

## Key concepts (learn these as you go)
| Concept | What it means |
|---|---|
| **Neuron, layer, activation** | A neuron is a weighted sum + nonlinearity (ReLU/sigmoid). Stacking layers lets the net model nonlinear patterns. |
| **Forward pass & loss** | Push inputs through the net to get predictions; the loss (cross-entropy) measures how wrong they are. |
| **Backpropagation & gradients** | The chain rule computes how each weight affects the loss; gradients say which way to nudge weights. |
| **Gradient descent & learning rate** | Iteratively step weights downhill on the loss surface; the learning rate sets step size (too big diverges, too small crawls). |
| **Autograd & optimizers** | PyTorch builds a computation graph and computes gradients automatically; optimizers (SGD, Adam) apply update rules. |
| **Overfitting, batches, epochs** | An epoch is one pass over data, split into mini-batches; regularization/early-stopping prevent memorizing the training set. |

## How - step by step
1. Load MNIST. Normalize pixels. Visualize a few digits so you trust the data.
2. NumPy net: implement forward (linear -> ReLU -> linear -> softmax), cross-entropy loss, and backprop deriving each gradient. Train with plain SGD.
3. Verify gradients numerically (finite differences) - this teaches you to trust/debug backprop.
4. Plot the training loss curve; confirm accuracy climbs above ~95% on MNIST.
5. Rebuild in PyTorch: nn.Module, DataLoader, Adam, .backward(). Reach similar/better accuracy in far less code.
6. Log both runs to W&B and put the loss curves side by side.
7. Write a short 'what I learned about backprop' note in the README - this is gold in interviews.

## Tech stack
Python, NumPy, PyTorch, torchvision, Matplotlib, W&B

## Where it lives (your tools)
| Tool | How you use it here |
|---|---|
| **Kaggle** | Use free GPU notebooks for the PyTorch version; MNIST/Fashion-MNIST are built in. |
| **GitHub** | Repo with numpy_net.py and torch_net.py side by side + the learnings note. |
| **Weights & Biases** | Log loss/accuracy curves for both implementations. |
| **Docker** | Optional: a CUDA-enabled Dockerfile to standardize the training environment. |

## Portfolio artifact
GitHub repo (the 'I understand the fundamentals' proof) + a W&B report. Resume line: 'Implemented backpropagation from scratch in NumPy and reproduced it in PyTorch.'

## Definition of Done
- [ ] NumPy net trains to >=95% MNIST accuracy with hand-written backprop.
- [ ] Numerical gradient check passes (analytic vs finite-difference gradients match).
- [ ] PyTorch version matches or beats the NumPy version.
- [ ] Both runs logged to W&B with comparable loss curves.
- [ ] README explains backprop in your own words with one diagram.

## Stretch goals
- Add dropout + batch norm and measure the effect.
- Try SGD vs Adam vs RMSprop and compare convergence.
- Extend to a small CNN and beat the MLP.

## Resources
- [3Blue1Brown Neural Networks](https://www.3blue1brown.com/topics/neural-networks)
- [Andrej Karpathy - micrograd / makemore](https://github.com/karpathy)
- [PyTorch 60-min blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

## Results

A 2-layer neural network built **from scratch in NumPy** — forward pass, cross-entropy loss, and
**backpropagation written by hand** — trained on scikit-learn's 8×8 handwritten digits.

| Metric | Value |
|---|---|
| Final test accuracy | **97.5%** |
| Training loss (epoch 10 → 60) | 0.080 → 0.009 |
| Gradient check (analytic vs numerical) | worst rel. error **2.51e-09 → PASS** |

The gradient check is the key result: it nudges each weight and compares the numerically-estimated
gradient to the one `backward()` computes. Agreement to ~1e-9 means the hand-derived backprop is
correct — the exact technique used to debug custom layers in production.

**What this demonstrates:** precisely what PyTorch's `loss.backward()` automates. Building it by
hand once makes every later project (transformers, fine-tuning, agents) debuggable instead of magic.

**Artifacts:** `numpy_net.py` (the net), `gradient_check.py` (the proof), `torch_net.py` (the
PyTorch twin).

**Next:** add a second hidden layer / dropout and compare; run the PyTorch twin on a GPU.
