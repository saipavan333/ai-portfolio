# AI Engineering Portfolio

A hands-on portfolio documenting my move from data engineering into AI. Fifteen projects across the
modern stack - classical ML, deep learning, NLP & transformers, LLMs (RAG, fine-tuning, evaluation),
agents, multimodal, reasoning/RL, and production serving/MLOps. Each project ships a public artifact
(a repo, a notebook, a model, or a live demo).

**About me:** ~10 years in ETL / data engineering, currently pursuing an Executive M.Tech at IIT
Jodhpur and moving into AI engineering - with a focus on making models **reliable and deployed**,
not just demoed.

**Tools used across the projects:** Kaggle - GitHub - Hugging Face - Weights & Biases - Docker

## How this repo is organized
Every project folder contains four layers:
- a **README** - the goal, approach, step list, and definition of done;
- a **LESSON.md** - the concepts from scratch, with a flow diagram and a code reading-order;
- runnable, heavily-commented code in **`src/`** (with a zero-setup `--demo` mode where practical); and
- an executed **notebook** in **`notebooks/`** - every cell explained, with diagrams.

Start with **[ROADMAP.md](ROADMAP.md)**, then work the projects in order.

## Projects
| # | Project | Phase | Focus |
|---|---------|-------|-------|
| 01 | [End-to-End Tabular ML Pipeline](01-tabular-ml-pipeline/README.md) | 1 | Classical ML, leakage-safe pipelines |
| 02 | [Reproducible, Feature-Engineered ML Pipeline (Data-Centric AI)](02-feature-pipeline-mlops/README.md) | 1 | Feature engineering, DVC, CI |
| 03 | [Neural Network from Scratch, then PyTorch](03-neural-net-from-scratch/README.md) | 2 | Neural nets & backprop from scratch |
| 04 | [Computer Vision with Transfer Learning + Live Demo](04-cv-transfer-learning/README.md) | 2 | CNNs, transfer learning, HF Space |
| 05 | [Fine-tune a Transformer for Text Classification (BERT)](05-finetune-bert-classifier/README.md) | 3 | Transformers, fine-tuning BERT |
| 06 | [Build a mini-GPT (Transformer from Scratch)](06-mini-gpt-from-scratch/README.md) | 3 | Self-attention, a mini-GPT |
| 07 | [Production RAG System (Retrieval-Augmented Generation)](07-production-rag/README.md) | 4 | Retrieval-augmented generation |
| 08 | [Fine-tune an LLM with LoRA / QLoRA (PEFT)](08-lora-qlora-finetune/README.md) | 4 | LoRA/QLoRA fine-tuning |
| 09 | [LLM Evaluation & Guardrails (LLMOps)](09-llm-eval-guardrails/README.md) | 4 | LLM evaluation & guardrails |
| 10 | [Single Agent with Tool Use (ReAct + Function Calling)](10-tool-using-agent/README.md) | 5 | Tool-using agents (ReAct) |
| 11 | [Multi-Agent Workflow (LangGraph / CrewAI)](11-multi-agent-workflow/README.md) | 5 | Multi-agent orchestration |
| 12 | [Multimodal App (Vision-Language Model)](12-multimodal-vlm-app/README.md) | 5 | Vision-language models |
| 13 | [Train a Reasoning Model with RL (GRPO) - Frontier Project](13-reasoning-model-rl/README.md) | 5 | Reasoning & RL (GRPO) |
| 14 | [LLMOps Deployment (Docker + vLLM Serving + Monitoring + CI/CD)](14-llmops-deployment/README.md) | 6 | Serving & MLOps (Docker + vLLM) |
| 15 | [Capstone: End-to-End Agentic AI Product](15-capstone-agentic-product/README.md) | 6 | End-to-end agentic product |

## Also in this repo
- **[ROADMAP.md](ROADMAP.md)** - the full plan: principles, phases, and timeline.
- **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)** - how the README / lesson / code / notebook layers fit together.
- **[GLOSSARY.md](GLOSSARY.md)** - every key term across the projects, in plain English.
- **[NOTEBOOKS.md](NOTEBOOKS.md)** - an index of the 15 teaching notebooks.
- **[SETUP.md](SETUP.md)** - environment + tooling setup.
