# ETL to AI Engineer: A Portfolio-First Blueprint
### 15 projects from classical ML to agentic & reasoning AI (2026 edition)

**For:** Pavan - 10 yrs ETL/Data Engineering, Executive M.Tech @ IIT Jodhpur  
**Pace:** ~20 hrs/week, 4-6 months (intensive track)  
**Tools:** Kaggle - GitHub - Hugging Face - Weights & Biases - Docker  
**Working branch:** `ai-portfolio`

A portfolio-first path from ETL/data engineering to AI engineer. 15 projects, each shipping a public artifact. Start with **[ROADMAP.md](ROADMAP.md)** and **[SETUP.md](SETUP.md)**, then work the projects in order.

## Learn the basics (new learning layer)

This repo now teaches, not just lists. Read **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)** to see
how to use it as a multi-year reference, and **[GLOSSARY.md](GLOSSARY.md)** for every term in
plain English. Each upgraded project has a **`LESSON.md`** (foundations from scratch + a flow
diagram + code reading-order) and full, runnable, heavily-commented code you can run with a
zero-setup `--demo` mode. Executed, self-explanatory **Kaggle notebooks** (every cell explained in plain English, with real diagrams) are available for **all 15 projects** in each project's `notebooks/` folder. The full `LESSON.md` + runnable `src/` upgrade is complete for **all 15 projects**.

## Project index

| # | Project | Phase | Difficulty | Primary artifact |
|---|---|---|---|---|
| 01 | [End-to-End Tabular ML Pipeline](01-tabular-ml-pipeline/README.md) | 1 | Beginner | GitHub repo + Kaggle notebook |
| 02 | [Reproducible, Feature-Engineered ML Pipeline (Data-Centric AI)](02-feature-pipeline-mlops/README.md) | 1 | Beginner-Intermediate | GitHub repo (CI + DVC) |
| 03 | [Neural Network from Scratch, then PyTorch](03-neural-net-from-scratch/README.md) | 2 | Intermediate | GitHub repo + W&B report |
| 04 | [Computer Vision with Transfer Learning + Live Demo](04-cv-transfer-learning/README.md) | 2 | Intermediate | Live HF Space + model |
| 05 | [Fine-tune a Transformer for Text Classification (BERT)](05-finetune-bert-classifier/README.md) | 3 | Intermediate | HF model + Space |
| 06 | [Build a mini-GPT (Transformer from Scratch)](06-mini-gpt-from-scratch/README.md) | 3 | Intermediate-Advanced | GitHub repo + W&B samples |
| 07 | [Production RAG System (Retrieval-Augmented Generation)](07-production-rag/README.md) | 4 | Intermediate-Advanced | RAG service + demo |
| 08 | [Fine-tune an LLM with LoRA / QLoRA (PEFT)](08-lora-qlora-finetune/README.md) | 4 | Advanced | HF adapter + demo |
| 09 | [LLM Evaluation & Guardrails (LLMOps)](09-llm-eval-guardrails/README.md) | 4 | Advanced | Eval-in-CI repo |
| 10 | [Single Agent with Tool Use (ReAct + Function Calling)](10-tool-using-agent/README.md) | 5 | Advanced | Agent demo |
| 11 | [Multi-Agent Workflow (LangGraph / CrewAI)](11-multi-agent-workflow/README.md) | 5 | Advanced | Multi-agent demo |
| 12 | [Multimodal App (Vision-Language Model)](12-multimodal-vlm-app/README.md) | 5 | Advanced | Multimodal HF Space |
| 13 | [Train a Reasoning Model with RL (GRPO) - Frontier Project](13-reasoning-model-rl/README.md) | 5 | Expert / Stretch | HF adapter + report |
| 14 | [LLMOps Deployment (Docker + vLLM Serving + Monitoring + CI/CD)](14-llmops-deployment/README.md) | 6 | Advanced | Deployed service + dashboards |
| 15 | [Capstone: End-to-End Agentic AI Product](15-capstone-agentic-product/README.md) | 6 | Expert / Flagship | Flagship live product |

## How to use this repo
1. Read `ROADMAP.md` (philosophy, tooling, timeline).
2. Do `SETUP.md` once (accounts, the `ai-portfolio` branch, tooling).
3. For each project: read its README, build in `src/`, track in W&B, ship the artifact, tick the Definition of Done.
4. Write a short build-in-public post per project.
