# Lesson 03 - Neural Networks & Backpropagation, from scratch

> This is the most important "basics" lesson in the whole program. Every later project (BERT,
> GPT, RAG, fine-tuning, agents) runs on neural networks trained by backpropagation. Build one
> by hand ONCE and none of it will ever be magic again.

## 0. What you'll be able to do after this
- Explain a neuron, a layer, an activation, a loss, and gradient descent.
- Explain backpropagation as "the chain rule, applied layer by layer".
- Read `numpy_net.py` and know what every line computes - and why PyTorch's `.backward()`
  does the same thing automatically (`torch_net.py`).

## 1. The big picture (why this project exists)
A neural network is just a big function with millions of tunable numbers (weights). "Training"
means automatically adjusting those numbers so the function's output matches the right answers.
The adjusting algorithm is **gradient descent**, and the trick that makes it efficient for
deep networks is **backpropagation**. That's the entire game. Our network learns to read 8x8
images of digits and hits ~97.5% accuracy - written in pure NumPy.

## 2. Foundations from scratch (the basics)

**A neuron.** Takes inputs, multiplies each by a weight, adds them up (plus a bias), and passes
the sum through a nonlinearity. In code that "weighted sum" is the matrix multiply
`X @ W + b`.

**Activation function (ReLU).** `relu(z) = max(0, z)`. Why needed? If you stack linear layers
with no nonlinearity, the whole stack collapses into ONE linear layer - no extra power. The
nonlinearity is what lets networks model curves and complex patterns.

**A layer.** Many neurons in parallel = one matrix multiply. Our net has two layers:
`input(64) -> Linear -> ReLU -> Linear -> 10 scores`.

**Softmax.** Turns the 10 raw output scores into 10 probabilities that sum to 1, so we can read
"the model thinks this is a 7 with 92% confidence".

**Loss function (cross-entropy).** A single number measuring how wrong the predictions are.
It's the average negative log-probability of the correct class: small when the model is
confident and right, large when it's confident and wrong. Training = making this number small.

**The forward pass.** Run inputs through the layers to get predictions (and the loss). That's
the `forward()` method. We cache intermediate values (`Z1`, `A1`) because backprop needs them.

**Gradients & gradient descent.** A gradient is the slope of the loss with respect to a weight:
"if I increase this weight a tiny bit, does the loss go up or down, and how fast?". Gradient
descent nudges every weight a small step in the DOWNHILL direction:
`weight -= learning_rate * gradient`. Repeat thousands of times and the loss falls.

**The learning rate.** The step size. Too big and training diverges (overshoots); too small and
it crawls. Ours is 0.5 for the NumPy net. (try 0.1 and 1.0 in the code.)

**Backpropagation = the chain rule.** To update a weight we need its gradient. The chain rule
lets us compute gradients layer by layer, from the output back to the input, reusing work.
The single most beautiful line in the code:
```
dZ2 = (P - Y) / n          # gradient of (softmax + cross-entropy) wrt the output scores
```
For softmax+cross-entropy this simplifies to "prediction minus truth" - which is WHY those two
are always paired. From there it's mechanical:
```
dW2 = A1.T @ dZ2           # output-layer weight gradients
dA1 = dZ2 @ W2.T           # push the error back to the hidden layer
dZ1 = dA1 * (Z1 > 0)       # ReLU's gradient: pass through where the input was positive
dW1 = X.T @ dZ1            # input-layer weight gradients
```

**Epochs and batches.** An *epoch* is one pass over all training data. We don't use all data at
once; we split it into small *batches* (64 rows) and update after each - "mini-batch SGD"
(stochastic gradient descent). Shuffling each epoch keeps batches varied.

**Autograd (the punchline).** `torch_net.py` is the same network in PyTorch. You write only the
forward pass; `loss.backward()` computes ALL those gradients automatically by tracking the
operations you did. Everything you wrote by hand still happens - PyTorch just does the
bookkeeping. That's why understanding this lesson makes PyTorch transparent.

## 3. How the pieces fit - the flow (one training step)

```
        forward                                   backward (chain rule)
  X --> Z1=XW1+b1 --> A1=relu(Z1) --> Z2=A1W2+b2 --> P=softmax(Z2) --> loss
                                                                          |
   dW1 <-- dZ1 <-- dA1 <----------------- dW2,db2 <----- dZ2=(P-Y)/n <----+
                                                                          |
                       step(): W -= lr * dW   (every weight moves downhill)
```

Repeat this forward->backward->step for every batch, every epoch. The loss falls; accuracy rises.

## 4. Code reading order
1. `numpy_net.py` math helpers (`relu`, `softmax`, `cross_entropy`) - the building blocks.
2. `MLP.__init__` - weight shapes and He initialization.
3. `MLP.forward` - the prediction path (match it to the flow diagram).
4. `MLP.backward` - the chain rule; read each line against the diagram, right to left.
5. `main()` - the training loop (epochs, batches, step).
6. `gradient_check.py` - the proof that backward() is correct.
7. `torch_net.py` - the same thing with autograd; notice what you DON'T have to write.

## 5. Why gradient_check.py matters
It nudges one weight up and down by a tiny epsilon and measures the loss change to estimate the
gradient numerically, then compares to your analytic gradient. Ours agrees to 2.5e-09 (PASS).
If you ever change `backward()` and this fails, your math is wrong - this is your safety net.

## 6. Common mistakes (and how to avoid them)
- **Forgetting the nonlinearity** -> the network can only draw straight lines.
- **Learning rate too high** -> loss explodes to NaN. Lower it.
- **Bad weight init** -> training stalls. He-init (used here) fixes it for ReLU.
- **Forgetting to zero gradients in PyTorch** (`opt.zero_grad()`) -> gradients accumulate and
  training goes haywire. (No analog in the NumPy version because we recompute from scratch.)
- **Not shuffling batches** -> the model can learn the data order instead of the pattern.

## 7. Check your understanding
1. Why does removing ReLU make a deep network no more powerful than a single linear layer?
2. In words, what does the gradient of a weight tell you?
3. Why does softmax+cross-entropy give the clean gradient `(P - Y)/n`?
4. What goes wrong if the learning rate is 50? If it's 0.00001?
5. What does PyTorch's `loss.backward()` replace from `numpy_net.py`?

## 8. Mini-glossary (full versions in /GLOSSARY.md)
neuron, weight/bias, activation (ReLU), softmax, loss, cross-entropy, forward pass, gradient,
backpropagation, gradient descent, learning rate, epoch, batch, autograd.

## 9. Going deeper
- 3Blue1Brown "Neural networks" series (watch this - the visuals are unmatched).
- Andrej Karpathy "The spelled-out intro to backprop / micrograd" (YouTube + GitHub).
- The PyTorch 60-minute blitz: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
