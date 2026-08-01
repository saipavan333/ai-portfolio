# Project 14 - LLMOps Deployment (Docker + vLLM Serving + Monitoring + CI/CD)

> **New here? Start with [LESSON.md](LESSON.md)** - it teaches the concepts from scratch, then read the heavily-commented, runnable code in this folder.
> Serve a model like a real production service: high-throughput inference, an API gateway, autoscaling-ready containers, monitoring, and CI/CD.
**Phase 6: Production & Capstone**  |  **Difficulty:** Advanced  |  **Est. time:** Week 22-23 (~40 hrs)

---

## Why (the point of this project)
This is the project that most directly turns a decade of ops/ETL experience into production AI value. 'MLOps/deployment - Docker, Kubernetes, vLLM, Triton - production muscle' is explicitly what 2026 employers want, and many ML engineers are weak here, which makes it a strong differentiator.

## What you will build
A production-style serving stack: an open model served with vLLM (continuous batching, OpenAI-compatible API) behind a FastAPI gateway (auth, rate limiting, logging), fully containerized with docker compose, with Prometheus/Grafana-style monitoring and a CI/CD pipeline that builds, tests, and pushes the image.

## Key concepts (learn these as you go)
| Concept | What it means |
|---|---|
| **High-throughput LLM serving** | vLLM/TGI use paged attention + continuous batching to serve many requests efficiently - far beyond a naive loop. |
| **API gateway concerns** | Auth, rate limiting, request validation, structured logging, and graceful degradation. |
| **Containerization & compose** | Multi-service apps (model server + gateway + monitoring) wired with docker compose; GPU passthrough. |
| **Monitoring & SLOs** | Track latency (p50/p95), throughput, errors, GPU/memory, and cost; define and watch SLOs - the ops DNA. |
| **CI/CD for models** | Automated build -> test -> scan -> push image; reproducible, versioned deploys. |

## How - step by step
1. Pick a model from an earlier project (your fine-tune from P8, or an open chat model).
2. Serve it with vLLM exposing an OpenAI-compatible endpoint.
3. Put a FastAPI gateway in front: API keys, rate limiting, request/response logging, health checks.
4. Write a multi-service docker-compose (model server + gateway + Prometheus + Grafana).
5. Add metrics (latency, throughput, errors, tokens, cost) and a Grafana dashboard.
6. Load-test (Locust/k6); record p50/p95 latency and max throughput; tune batching.
7. Add CI/CD (GitHub Actions): build image, run tests, scan, push to a registry.
8. Document the architecture, SLOs, and a runbook (how to deploy/roll back).

## Tech stack
vLLM (or TGI), FastAPI, Docker + docker compose, Prometheus + Grafana, Locust/k6, GitHub Actions, a container registry

## Where it lives (your tools)
| Tool | How you use it here |
|---|---|
| **Docker** | The star here - multi-container serving stack with GPU passthrough. |
| **GitHub** | CI/CD pipeline + infra-as-code + runbook; an ops showpiece. |
| **Hugging Face** | Source the model weights; optionally deploy the gateway as a Space. |
| **Weights & Biases** | Optional: log serving metrics / a load-test report. |

## Portfolio artifact
GitHub repo (architecture + runbook) + a published image + dashboard screenshots. Resume line: 'Deployed an LLM with vLLM behind a FastAPI gateway (auth, rate limiting, monitoring, CI/CD); p95 latency Xms at N req/s.'

## Definition of Done
- [ ] Model served via vLLM with an OpenAI-compatible API.
- [ ] Gateway enforces auth + rate limiting and logs structured requests.
- [ ] `docker compose up` brings up server + gateway + monitoring together.
- [ ] Grafana dashboard shows latency/throughput/errors; a load test report exists.
- [ ] CI/CD builds, tests, and pushes the image; a runbook documents deploy/rollback.

## Stretch goals
- Add a Kubernetes manifest + HPA for autoscaling.
- Add canary/blue-green deploy and a rollback demo.
- Add semantic caching to cut cost/latency and measure savings.

## Resources
- [vLLM docs](https://docs.vllm.ai/)
- [FastAPI docs](https://fastapi.tiangolo.com/)
- [Grafana + Prometheus](https://prometheus.io/docs/visualization/grafana/)

## Results (fill this in as you build)
- **Headline metric:** _e.g. AUC / F1 / accuracy / p95 latency_
- **Artifact links:** GitHub _ | HF _ | Kaggle _ | W&B _ | Demo _
- **One thing that broke and how I fixed it:** _..._
- **Build-in-public post:** _link_
