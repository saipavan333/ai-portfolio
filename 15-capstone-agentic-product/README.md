# Project 15 - Capstone: End-to-End Agentic AI Product

> **New here? Start with [LESSON.md](LESSON.md)** - it teaches the concepts from scratch, then read the heavily-commented, runnable code in this folder.
> Combine everything into one real product: a deployed, monitored, agentic application solving a genuine problem - ideally tied to your domain.
**Phase 6: Production & Capstone**  |  **Difficulty:** Expert / Flagship  |  **Est. time:** Week 23-24+ (~50 hrs)

---

## Why (the point of this project)
One deep, complete product beats ten toys. The capstone is the flagship: it demonstrates taking a real problem from data to a deployed, evaluated, monitored, agentic system - the full stack employers pay for.  It is tied to a data-engineering / enterprise-data background.

## What you will build
A complete application that integrates the program's skills: a RAG knowledge base (P7) + a multi-agent or tool-using core (P10/P11) + an optional fine-tuned/specialized model (P8) + evaluation & guardrails (P9) + production deployment & monitoring (P14), with a polished UI and a public demo. Example: an 'enterprise data assistant' that answers questions over a data warehouse's docs, writes/validates SQL via agents, runs it, and explains results - directly leveraging a data-engineering background.

## Key concepts (learn these as you go)
| Concept | What it means |
|---|---|
| **System design for AI products** | Compose retrieval, agents, models, guardrails, and serving into one coherent, observable architecture. |
| **End-to-end evaluation** | Evaluate the whole product (task success, faithfulness, safety, latency, cost), not just components. |
| **Reliability & cost engineering** | Budgets, caching, fallbacks, retries, and graceful degradation for a real user-facing system. |
| **Product thinking** | Scope to a real user + job-to-be-done; a focused product beats a kitchen-sink demo. |
| **Storytelling** | A README + demo video + write-up that makes a busy hiring manager 'get it' in 60 seconds. |

## How - step by step
1. Choose a real problem (ideally domain-tied: enterprise data assistant, compliance doc agent, analytics copilot).
2. Write a 1-page design doc: users, scope, architecture, success metrics, risks.
3. Assemble the pieces: ingestion/RAG (P7) + agent core (P10/P11) + guardrails/eval (P9).
4. Add a clean UI (Gradio/Streamlit or a light web front end) with streaming + visible agent steps.
5. Deploy with the P14 stack (containers, API, monitoring); put it behind auth.
6. Run end-to-end evaluation (task success, faithfulness, safety, latency, cost); log to W&B.
7. Record a 2-3 min demo video; write a strong README + an architecture diagram.
8. Publish: live demo link, repo, write-up/blog, and a LinkedIn launch post.

## Tech stack
Everything: RAG + agents + (optional) fine-tuned model + guardrails + vLLM/Docker + monitoring + a UI

## Where it lives (your tools)
| Tool | How you use it here |
|---|---|
| **Hugging Face** | Models + the live demo Space (or link to your deployed app). |
| **GitHub** | The flagship repo - clean, documented, tested, CI/CD. |
| **Docker** | Production deployment of the whole multi-service system. |
| **Weights & Biases** | End-to-end evaluation report + monitoring. |

## Portfolio artifact
Live product demo + flagship GitHub repo + demo video + blog post + LinkedIn launch. Resume headline: 'Designed, built, evaluated, and deployed an end-to-end agentic AI product (RAG + multi-agent + monitored serving).'

## Definition of Done
- [ ] Solves a real, scoped problem end-to-end with a working deployed demo.
- [ ] Integrates retrieval + agents + guardrails + monitored serving.
- [ ] End-to-end evaluation (success/faithfulness/safety/latency/cost) is reported.
- [ ] Polished README + architecture diagram + 2-3 min demo video.
- [ ] Public launch: live link + repo + write-up + LinkedIn post.

## Stretch goals
- Add a feedback loop that logs user corrections to improve the system.
- Add multi-tenant auth + usage metering.
- Write a short 'lessons learned / what I'd do next' post-mortem.

## Resources
- [Hugging Face Spaces (deploy)](https://huggingface.co/docs/hub/spaces)
- [Streamlit (UI option)](https://docs.streamlit.io/)
- [Google - People + AI Guidebook (product)](https://pair.withgoogle.com/guidebook/)

## Results (fill this in as you build)
- **Headline metric:** _e.g. AUC / F1 / accuracy / p95 latency_
- **Artifact links:** GitHub _ | HF _ | Kaggle _ | W&B _ | Demo _
- **One thing that broke and how I fixed it:** _..._
- **Build-in-public post:** _link_
