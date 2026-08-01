# Project 09 - LLM Evaluation & Guardrails (LLMOps)

> **New here? Start with [LESSON.md](LESSON.md)** - it teaches the concepts from scratch, then read the heavily-commented, runnable code in this folder.
> Build the unglamorous layer that separates demos from products: systematic LLM evaluation, safety guardrails, and observability.
**Phase 4: LLMs, RAG & Fine-tuning**  |  **Difficulty:** Advanced  |  **Est. time:** Week 15-17 (~40 hrs)

---

## Why (the point of this project)
Anyone can get an LLM demo working once; shipping one you can trust requires evaluation, guardrails, and monitoring. This LLMOps discipline maps directly onto data-engineering skills (testing, data quality, observability) - and it is a fast-growing, under-supplied skill. It makes every other project on this list production-credible.

## What you will build
An evaluation + guardrail toolkit applied to your RAG (P7) or fine-tuned model (P8): an automated eval suite (correctness, faithfulness, format, safety), input/output guardrails (PII, injection, schema validation), and a tracing/monitoring dashboard - all in CI so regressions are caught before deploy.

## Key concepts (learn these as you go)
| Concept | What it means |
|---|---|
| **LLM evaluation methods** | Reference-based metrics, rubric scoring, and LLM-as-judge; build a fixed eval set so changes are measurable. |
| **LLM-as-a-judge (and its pitfalls)** | Use a strong model to grade outputs against a rubric; control for bias/variance with calibration. |
| **Guardrails** | Validate/sanitize inputs and outputs: PII redaction, prompt-injection detection, JSON-schema enforcement, refusal of unsafe requests. |
| **Observability/tracing** | Log every prompt, retrieval, token, latency, and cost per request to debug and monitor in production. |
| **Regression testing for prompts** | Treat prompts/models like code: a test suite that fails CI when quality drops. |

## How - step by step
1. Define an eval set (inputs + expected behavior) for your P7/P8 system; version it.
2. Implement metrics: exact/semantic correctness, faithfulness, format/schema validity, and a safety check.
3. Add LLM-as-judge scoring with a clear rubric; spot-check judge reliability against your own labels.
4. Add input guardrails (PII detection, injection patterns) and output guardrails (schema validation, toxicity).
5. Instrument tracing (latency, tokens, cost) - e.g. with an open observability tool; build a dashboard.
6. Wire the eval suite into GitHub Actions so PRs run it and block on regression.
7. Run an adversarial/red-team pass (jailbreaks, prompt injection) and document mitigations.
8. Publish an evaluation report (W&B) with scorecards across versions.

## Tech stack
pytest, an eval framework (e.g. DeepEval/Promptfoo/RAGAS), guardrails tooling (e.g. Guardrails AI/Presidio), OpenTelemetry/Langfuse-style tracing, GitHub Actions, W&B

## Where it lives (your tools)
| Tool | How you use it here |
|---|---|
| **GitHub** | Eval-in-CI workflow; this repo proves 'I ship safely.' |
| **Weights & Biases** | Versioned evaluation scorecards and trend charts. |
| **Docker** | Containerize the eval + tracing stack for reproducibility. |
| **Hugging Face** | Use small open models as judges/classifiers for guardrails. |

## Portfolio artifact
GitHub repo with eval-in-CI + a public evaluation report. Resume line: 'Built an LLM evaluation + guardrail suite (LLM-as-judge, PII/injection defense, eval-in-CI) catching regressions before deploy.'

## Definition of Done
- [ ] Versioned eval set + automated metrics produce a scorecard.
- [ ] Input + output guardrails block PII, injection, and malformed outputs in tests.
- [ ] Eval suite runs in GitHub Actions and fails on a seeded regression.
- [ ] Tracing dashboard shows latency, token, and cost per request.
- [ ] A red-team report lists attacks tried and mitigations applied.

## Stretch goals
- Add automatic prompt optimization (e.g. DSPy) and measure eval gains.
- Add cost/latency budgets that fail CI when exceeded.
- Build a small public 'LLM eval cookbook' from what you learned.

## Resources
- [Promptfoo (eval & red-team)](https://www.promptfoo.dev/)
- [Microsoft Presidio (PII)](https://microsoft.github.io/presidio/)
- [Langfuse (LLM observability)](https://langfuse.com/docs)

## Results (fill this in as you build)
- **Headline metric:** _e.g. AUC / F1 / accuracy / p95 latency_
- **Artifact links:** GitHub _ | HF _ | Kaggle _ | W&B _ | Demo _
- **One thing that broke and how I fixed it:** _..._
- **Build-in-public post:** _link_
