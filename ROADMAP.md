# ROADMAP - The Full Blueprint

*ETL to AI Engineer: A Portfolio-First Blueprint - 15 projects from classical ML to agentic & reasoning AI (2026 edition)*

## Mindset (read this first)

### You are not starting from zero
You have spent 10 years doing the single hardest, least glamorous, most in-demand part of AI: moving data reliably. In 2026 employers screen for "production deployment evidence first" and the #1 bottleneck in real AI teams is data and operations, not model math. Your ETL background (SQL, orchestration, pipelines, reliability, monitoring) is a genuine head start. The plan below deliberately leans into that edge instead of pretending you are a fresh grad.

### Portfolio-first, not course-first
Tutorials are consumed and forgotten; artifacts get you hired. Every project here ends with a public, clickable proof-of-work: a GitHub repo, a live Hugging Face Space, a Kaggle notebook, a W&B report, or a Docker image. By the end you have ~15 such artifacts plus one flagship capstone. A hiring manager should be able to click and see working systems within 30 seconds.

### Build in public
After each project, write a short (300-500 word) post on LinkedIn or a blog: what you built, one thing that broke, one thing you learned. This compounds. Recruiters find the posts; the posts link to the repos; the repos prove the skill. Three months of this is worth more than any certificate.

### Spaced difficulty, with a spine
The 15 projects escalate: tabular ML -> deep learning -> NLP/transformers -> LLMs/RAG/fine-tuning -> agents/multimodal -> production & capstone. Each phase reuses skills from the last. Resist the urge to skip to agents on day one; the foundations are what make the flashy stuff debuggable.

### Definition of Done
Every project has an explicit checklist. AI work is a swamp of 'I could tune this forever.' The checklist tells you when to stop, ship, and move on. Shipping 15 'good enough and public' projects beats polishing 2 forever.

## Your five tools, mapped to the journey

| Tool | Role | How you use it across the program |
|---|---|---|
| **GitHub** | Code + history | Home base for all code. You will create one dedicated branch, ai-portfolio (commands in SETUP), and one folder per project. Use it for clean READMEs, GitHub Actions CI (lint + test on every push), and as the URL you put on your resume. A green-commit history that tells a story is itself a portfolio signal. |
| **Kaggle** | Data + free GPUs | Your free compute and dataset source. Kaggle Notebooks give ~30 GPU hrs/week (T4/P100) at no cost - enough for most of this plan. Use Datasets for inputs, Notebooks for experiments you want to show, and 2-3 Competitions to benchmark yourself against a public leaderboard (a great, objective resume line). |
| **Hugging Face** | Models + demos | The center of gravity of open AI. You will (a) pull models/datasets with the transformers & datasets libraries, (b) push your fine-tuned models and datasets to the Hub, and (c) deploy live demos as Spaces (Gradio). A working Space is the single most impressive thing a non-technical hiring manager can click. |
| **Weights & Biases** | Experiment tracking | Your lab notebook. Log every training run (loss, metrics, configs, system stats), run hyperparameter Sweeps, register models, and publish W&B Reports. 'I track and compare experiments rigorously' is exactly the MLOps maturity employers want to see, and it maps directly onto the discipline you already have from data pipelines. |
| **Docker** | Reproducibility + deploy | Your reproducibility and shipping layer - and a natural extension of ETL ops. You will containerize training and serving so 'works on my machine' dies, then deploy containers (HF Spaces, a cloud run service, or local). By the capstone you will have a multi-container app with an API, a vector DB, and a model server. |

## The six phases

| Phase | Theme | Weeks | Goal |
|---|---|---|---|
| 1 | **Foundations & Data-Centric ML** | Weeks 1-3 | Play to your strength. Ship two production-grade tabular ML pipelines and lock in MLOps habits (tracking, reproducibility) from day one. |
| 2 | **Deep Learning Core** | Weeks 3-7 | Earn your deep-learning fundamentals: build a net from scratch, then do real computer vision with transfer learning and a live demo. |
| 3 | **NLP & Transformers** | Weeks 7-11 | Go from classical NLP to the architecture that runs modern AI. Fine-tune BERT, then build a mini-GPT so transformers stop being magic. |
| 4 | **LLMs, RAG & Fine-tuning** | Weeks 11-17 | The commercial core of 2026. Production RAG, parameter-efficient fine-tuning (LoRA/QLoRA), and LLM evaluation/guardrails (LLMOps). |
| 5 | **Agentic AI & Multimodal** | Weeks 17-22 | The 2026 frontier. Tool-using agents, multi-agent orchestration, a vision-language app, and a taste of training reasoning models with RL. |
| 6 | **Production & Capstone** | Weeks 22-24 | Tie it together like an engineer. Full LLMOps deployment, then a flagship end-to-end agentic product that combines everything. |

## Timeline (intensive ~20 hrs/week, ~4-6 months)

| When | Phase | Projects | Focus |
|---|---|---|---|
| Weeks 1-3 | Phase 1 | Projects 1-2 | Tabular pipeline + reproducible/feature-engineered pipeline. Set up all 5 tools. |
| Weeks 3-7 | Phase 2 | Projects 3-4 | NN from scratch + PyTorch; CV transfer learning with a live HF Space. |
| Weeks 7-11 | Phase 3 | Projects 5-6 | Fine-tune BERT classifier; build & train a mini-GPT from scratch. |
| Weeks 11-17 | Phase 4 | Projects 7-9 | Production RAG; LoRA/QLoRA fine-tune; LLM eval + guardrails. |
| Weeks 17-22 | Phase 5 | Projects 10-13 | Tool agent; multi-agent workflow; multimodal app; reasoning-model RL. |
| Weeks 22-24 | Phase 6 | Projects 14-15 | LLMOps deployment; capstone agentic product. Polish, write-ups, resume. |

## The 15 projects at a glance


### Phase 1: Foundations & Data-Centric ML

**01. End-to-End Tabular ML Pipeline** (Beginner, Week 1-2 (~35 hrs))  
Predict a real-world outcome from tabular data with a clean, reproducible scikit-learn pipeline. Your ETL skills shine here.  
-> [Full blueprint](01-tabular-ml-pipeline/README.md)

**02. Reproducible, Feature-Engineered ML Pipeline (Data-Centric AI)** (Beginner-Intermediate, Week 2-3 (~35 hrs))  
Turn project 1 into a versioned, automated pipeline. This is the project that says 'I am the data person every AI team is desperate for.'  
-> [Full blueprint](02-feature-pipeline-mlops/README.md)


### Phase 2: Deep Learning Core

**03. Neural Network from Scratch, then PyTorch** (Intermediate, Week 3-4 (~35 hrs))  
Build a working neural net with only NumPy (forward + backprop by hand), then rebuild it in PyTorch. After this, deep learning is no longer a black box.  
-> [Full blueprint](03-neural-net-from-scratch/README.md)

**04. Computer Vision with Transfer Learning + Live Demo** (Intermediate, Week 4-7 (~45 hrs))  
Fine-tune a pretrained vision model on your own image classes and ship it as a live Hugging Face Space anyone can try in the browser.  
-> [Full blueprint](04-cv-transfer-learning/README.md)


### Phase 3: NLP & Transformers

**05. Fine-tune a Transformer for Text Classification (BERT)** (Intermediate, Week 7-9 (~40 hrs))  
Use the Hugging Face Transformers stack end-to-end: tokenize text, fine-tune a BERT-family model, evaluate properly, and ship a demo.  
-> [Full blueprint](05-finetune-bert-classifier/README.md)

**06. Build a mini-GPT (Transformer from Scratch)** (Intermediate-Advanced, Week 9-11 (~45 hrs))  
Implement a small GPT-style decoder (attention, positional encoding, blocks) and train it to generate text. This demystifies every LLM you will ever use.  
-> [Full blueprint](06-mini-gpt-from-scratch/README.md)


### Phase 4: LLMs, RAG & Fine-tuning

**07. Production RAG System (Retrieval-Augmented Generation)** (Intermediate-Advanced, Week 11-13 (~45 hrs))  
Build a question-answering system over your own documents: chunk, embed, store in a vector DB, retrieve, and generate grounded answers with citations.  
-> [Full blueprint](07-production-rag/README.md)

**08. Fine-tune an LLM with LoRA / QLoRA (PEFT)** (Advanced, Week 13-15 (~45 hrs))  
Adapt an open LLM to a specific task/domain/style on a single GPU using parameter-efficient fine-tuning, then publish the adapter to the Hub.  
-> [Full blueprint](08-lora-qlora-finetune/README.md)

**09. LLM Evaluation & Guardrails (LLMOps)** (Advanced, Week 15-17 (~40 hrs))  
Build the unglamorous layer that separates demos from products: systematic LLM evaluation, safety guardrails, and observability.  
-> [Full blueprint](09-llm-eval-guardrails/README.md)


### Phase 5: Agentic AI & Multimodal

**10. Single Agent with Tool Use (ReAct + Function Calling)** (Advanced, Week 17-18 (~35 hrs))  
Build an LLM agent that reasons, calls tools (search, calculator, your APIs), observes results, and loops until the task is done.  
-> [Full blueprint](10-tool-using-agent/README.md)

**11. Multi-Agent Workflow (LangGraph / CrewAI)** (Advanced, Week 18-20 (~40 hrs))  
Orchestrate several specialized agents (planner, researcher, coder, critic) that collaborate to solve a task no single agent handles well.  
-> [Full blueprint](11-multi-agent-workflow/README.md)

**12. Multimodal App (Vision-Language Model)** (Advanced, Week 20-22 (~40 hrs))  
Build an app that understands images AND text together - visual Q&A, document/screenshot understanding, or image-grounded chat.  
-> [Full blueprint](12-multimodal-vlm-app/README.md)

**13. Train a Reasoning Model with RL (GRPO) - Frontier Project** (Expert / Stretch, Week 21-22 (~35 hrs))  
Teach a small model to reason step-by-step on verifiable tasks (math/logic/code) using reinforcement learning with verifiable rewards (GRPO).  
-> [Full blueprint](13-reasoning-model-rl/README.md)


### Phase 6: Production & Capstone

**14. LLMOps Deployment (Docker + vLLM Serving + Monitoring + CI/CD)** (Advanced, Week 22-23 (~40 hrs))  
Serve a model like a real production service: high-throughput inference, an API gateway, autoscaling-ready containers, monitoring, and CI/CD.  
-> [Full blueprint](14-llmops-deployment/README.md)

**15. Capstone: End-to-End Agentic AI Product** (Expert / Flagship, Week 23-24+ (~50 hrs))  
Combine everything into one real product: a deployed, monitored, agentic application solving a genuine problem - ideally tied to your domain.  
-> [Full blueprint](15-capstone-agentic-product/README.md)

## After the program (turning this into a job)

- **Resume:** lead with the capstone (P15) + P7 (RAG) + P14 (LLMOps). Each bullet = artifact link + a metric.
- **Pin** your 6 best repos on GitHub; keep the `ai-portfolio` branch clean and documented.
- **Spaces:** keep 3-4 live HF Spaces - these are what non-technical interviewers click.
- **Narrative:** 'ETL engineer who can put AI in production' - lean on P2, P7, P9, P14 (data + ops).
- **Kaggle:** a couple of competition ranks make great objective resume lines.
- **Write-ups:** one post per project; a final 'what I built in 6 months' summary post.
