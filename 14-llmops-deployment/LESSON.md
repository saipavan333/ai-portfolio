# Lesson 14 - Production Serving & LLMOps (Docker + vLLM)

> Read with `src/serving_sim.py` open; run it (`python -m src.serving_sim`). See
> `notebooks/14_serving_mlops.ipynb` for the throughput/latency/architecture diagrams.

## 0. What you'll be able to do after this
- Explain why **batching** is the key to LLM-serving throughput, and what **SLOs** are.
- Explain an **API gateway**'s job (auth, rate limiting, logging).
- Run serving simulations and read the production **FastAPI gateway** + **Docker Compose** stack.

## 1. The big picture (why this project exists)
This is the project that turns your data/ops background into an AI paycheck. Anyone can run a model
once; few can serve it to thousands of users fast and cheaply. Production serving (vLLM batching), a
gateway (auth + rate limits), and monitoring (SLOs) are what make AI a product. 2026 employers
screen for exactly this "production deployment evidence".

## 2. Foundations from scratch (the basics)
- **Throughput vs latency:** throughput = requests/sec; latency = time per request. Serving maximises
  throughput while keeping latency acceptable.
- **Continuous batching (vLLM):** a GPU step has fixed overhead, so processing many requests together
  amortises it. `throughput()` shows batch 1 -> 64 going ~18 -> 173 req/s (asserted).
- **SLOs and the tail:** you watch **p95/p99**, not the mean, because a few slow requests ruin UX.
  `latency_percentiles()` shows the p99 well above the median.
- **API gateway:** the front door - **auth** (API keys), **rate limiting** (a token bucket;
  `TokenBucket` throttles a burst), logging - before traffic reaches the model. `gateway/app.py` is
  the real FastAPI version.
- **Containers:** package server + gateway + monitoring with `docker-compose.yml` so it runs
  identically anywhere - a data/ops instinct, applied to AI.

## 3. How the pieces fit - the flow
```
  client -> API gateway (auth + rate limit + logging) -> vLLM server (batched GPU) -> response
                         (gateway/app.py)                    (docker-compose.yml)
                                   \___ metrics ___> Prometheus + Grafana (SLO dashboards)
            serving_sim.py simulates batching, the latency tail, and rate limiting on CPU.
```

## 4. Code reading order
1. `src/serving_sim.py` - throughput, latency percentiles, token bucket (run it first).
2. `notebooks/14_serving_mlops.ipynb` - the rendered curves + architecture diagram.
3. `gateway/app.py` - the production FastAPI gateway (auth + rate limit + proxy).
4. `docker-compose.yml`, `loadtest/locustfile.py`, `.github/workflows/deploy.yml` - the full stack + CI/CD.

## 5. Common mistakes (and how to avoid them)
- **Serving one request at a time** -> terrible throughput; use vLLM's batching.
- **Watching the average latency** -> hides the tail; track p95/p99.
- **No auth/rate limit** -> abuse and runaway cost; the gateway enforces both.
- **No runbook** -> nobody can deploy/roll back safely; write one.

## 6. Check your understanding
1. Why does batching raise throughput on the same GPU?
2. Why report p95/p99 instead of the mean latency?
3. What three jobs does the API gateway do before the model sees a request?
4. What does `docker compose up` bring up here?

## 7. Mini-glossary (full versions in /GLOSSARY.md)
throughput, latency, continuous batching, p50/p95/p99, SLO, API gateway, rate limiting/token bucket,
vLLM, Docker Compose, CI/CD.

## 8. Going deeper
- vLLM: https://docs.vllm.ai/
- FastAPI: https://fastapi.tiangolo.com/
