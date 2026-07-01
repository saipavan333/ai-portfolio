# GLOSSARY - every key term in the program, in plain English

Terms are grouped by theme. The "(Pxx)" tag shows where the term is first taught in depth.

## Machine-learning basics
- **Model (P1):** a function learned from data that maps inputs (features) to an output (a
  prediction). "Training" = finding the function; "inference" = using it.
- **Features / target (P1):** features are the input columns (X); the target is what you
  predict (y).
- **Supervised learning (P1):** learning from labeled examples (you know the right answer).
  Classification predicts categories; regression predicts numbers.
- **Train / validation / test split (P1):** slices of data used to fit, tune, and finally
  judge a model. The test set is touched exactly once, at the end.
- **Cross-validation (P1):** rotating the validation slice over k folds and averaging, for a
  more stable score than a single split.
- **Data leakage (P1):** information from the answer (or the future) sneaking into features
  or preprocessing, giving fake-good scores that collapse in production.
- **Baseline (P1):** the dumbest reasonable model (majority class / mean). Everything must
  beat it, or your fancy model adds nothing.
- **Overfitting / underfitting (P1, P3):** memorizing the training data vs. being too simple
  to capture the pattern. Both hurt test performance.
- **Regularization (P3):** techniques (dropout, weight decay, early stopping) that discourage
  overfitting.
- **Hyperparameters (P1):** settings you choose before training (learning rate, tree depth),
  as opposed to parameters the model learns.

## Metrics
- **Accuracy (P1):** fraction correct. Misleading when classes are imbalanced.
- **Precision / recall / F1 (P1, P5):** precision = of predicted positives, how many were
  right; recall = of actual positives, how many you caught; F1 = their harmonic mean.
- **ROC-AUC (P1):** probability the model ranks a random positive above a random negative;
  0.5 = random, 1.0 = perfect.
- **RMSE / MAE (P1):** typical size of regression error (squared vs. absolute).
- **Confusion matrix (P1):** table of predicted-vs-actual counts.

## Pipelines & MLOps
- **scikit-learn Pipeline (P1):** preprocessing + model bundled into one object so transforms
  fit only on training folds (kills leakage).
- **Feature engineering (P2):** creating informative inputs (ratios, aggregates, time windows)
  from raw columns.
- **Feature store (P2):** a place that serves identical feature logic to training and
  production, preventing train/serve skew.
- **Data-centric AI (P2):** improving the dataset (labels, features, coverage) rather than
  only the model - often the biggest win.
- **DVC (P2):** git-like version control for large data and model files.
- **CI/CD (P2, P14):** automation that lints, tests, builds, and deploys on every push.
- **Reproducibility (P2):** anyone can recreate your result from a commit (fixed data, code,
  seeds, environment).

## Deep learning
- **Neuron (P3):** a weighted sum of inputs passed through a nonlinearity.
- **Activation function (P3):** the nonlinearity (ReLU, sigmoid) that lets nets model curves.
- **Forward pass (P3):** running inputs through the network to get a prediction.
- **Loss function (P3):** a number measuring how wrong the prediction is (cross-entropy, MSE).
- **Backpropagation (P3):** using the chain rule to compute how each weight affects the loss.
- **Gradient descent (P3):** nudging weights downhill on the loss surface; the learning rate
  sets the step size.
- **Optimizer (P3):** the update rule (SGD, Adam) that applies gradients.
- **Epoch / batch (P3):** one full pass over the data; a batch is a small chunk processed at
  once.
- **Autograd (P3):** automatic gradient computation (PyTorch builds a graph and differentiates).
- **CNN (P4):** convolutional neural network - exploits local spatial structure in images.
- **Transfer learning / fine-tuning (P4, P8):** start from a model pretrained on lots of data
  and adapt it to your smaller task.
- **Data augmentation (P4):** random transforms (flip, crop) that expand image data.

## NLP & transformers
- **Tokenization (P5):** splitting text into subword tokens mapped to integer ids.
- **Embedding (P5, P7):** a vector representation of text where similar meaning = nearby
  vectors.
- **Transformer (P6):** the architecture behind modern AI; uses attention to mix context.
- **Self-attention (P6):** each token looks at others (via query/key/value) to build
  context-aware representations.
- **Positional encoding (P6):** position info injected because attention has no built-in order.
- **Causal mask (P6):** stops a token from "seeing the future" so the model can generate text.
- **Language modeling (P6):** predicting the next token; the training objective of GPTs.

## LLMs, RAG & fine-tuning
- **LLM (P7):** large language model - a big transformer trained on huge text.
- **Prompt (P7):** the input text/instructions you give an LLM.
- **RAG (P7):** Retrieval-Augmented Generation - fetch relevant documents and let the LLM
  answer grounded in them (reduces hallucination, adds citations).
- **Vector database (P7):** stores embeddings and finds nearest neighbors fast (FAISS, Chroma,
  pgvector).
- **Chunking (P7):** splitting documents into passages sized for retrieval.
- **Hybrid search / re-ranking (P7):** combine keyword + vector search, then reorder with a
  cross-encoder for precision.
- **Hallucination (P7, P9):** an LLM confidently stating something false.
- **PEFT / LoRA / QLoRA (P8):** parameter-efficient fine-tuning - train tiny adapter weights
  (LoRA), optionally on a 4-bit "quantized" base model (QLoRA), so you can fine-tune on one GPU.
- **Quantization (P8, P14):** storing weights in fewer bits (e.g. 4-bit) to save memory.
- **Instruction tuning (P8):** fine-tuning on instruction/response pairs so a model follows
  prompts.
- **LLM-as-a-judge (P9):** using a strong model to grade outputs against a rubric.
- **Guardrails (P9):** input/output checks (PII, prompt-injection, schema) that keep a system
  safe.
- **Observability / tracing (P9, P14):** logging every prompt, token, latency, and cost to
  debug and monitor.

## Agents, multimodal & reasoning
- **Agent (P10):** an LLM that reasons, calls tools, observes results, and loops until done.
- **Tool / function calling (P10):** exposing functions with typed schemas the model can call.
- **ReAct (P10):** interleaving Reasoning and Acting in the agent loop.
- **Multi-agent orchestration (P11):** several specialized agents (planner, researcher, critic)
  collaborating via shared state and control flow (LangGraph, CrewAI).
- **VLM (P12):** vision-language model - understands images and text together.
- **Reasoning model (P13):** a model trained to "think" step-by-step before answering.
- **RLVR / GRPO (P13):** reinforcement learning with verifiable rewards; GRPO is an efficient
  policy-optimization method used to train reasoning.
- **Reward hacking (P13):** a model exploiting loopholes in the reward instead of doing the task.

## Serving & deployment
- **Inference server (P14):** a service that runs a model for many requests (vLLM, TGI, Triton).
- **Continuous batching / paged attention (P14):** vLLM tricks that serve many LLM requests
  efficiently.
- **API gateway (P14):** the front door (auth, rate limiting, logging) before your model.
- **Container / Docker image (P2, P14):** a packaged, reproducible environment that runs the
  same anywhere.
- **SLO (P14):** service-level objective - a target like "p95 latency < 800 ms".

## Your tools
- **GitHub:** code, history, branches, pull requests, CI (GitHub Actions).
- **Kaggle:** free GPUs (Notebooks), datasets, competitions.
- **Hugging Face:** models, datasets, and live demos (Spaces); the transformers/datasets libs.
- **Weights & Biases (W&B):** experiment tracking, sweeps, reports, model registry.
- **Docker:** containerization for reproducibility and deployment.
