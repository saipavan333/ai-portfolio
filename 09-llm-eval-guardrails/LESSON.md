# Lesson 09 - LLM Evaluation & Guardrails (LLMOps)

> Read with `src/evaluate.py` and `src/guardrails.py` open. This whole project is **fully runnable**
> (no GPU) - run the demos and the CI tests. See `notebooks/09_llm_eval_guardrails.ipynb` for the
> scorecard + dashboard diagrams.

## 0. What you'll be able to do after this
- Build an **evaluation harness** (exact-match + semantic similarity) that catches bad answers.
- Build **guardrails**: PII redaction, prompt-injection detection, output-schema validation.
- Wire both into **CI** so quality/safety regressions can't ship.

## 1. The big picture (why this project exists)
Anyone can get an LLM demo working once; shipping one you can trust needs the unglamorous layer:
evaluation (is it good?) and guardrails (is it safe?). This is a data-engineering discipline - testing, data
quality, monitoring - applied to AI, and it's exactly what 2026 employers mean by "production-ready".

## 2. Foundations from scratch (the basics)
- **Why eval is hard:** there's often no single right answer, so use a fixed **eval set** + multiple
  metrics. `src/evaluate.py` scores **exact match** (strict) and **semantic similarity** (meaning).
  Its `main()` asserts the harness FAILS the wrong "2+2=5" answer - proof it actually catches errors.
- **LLM-as-a-judge:** in production, a strong model grades answers against a rubric (powerful but
  biased - calibrate it). The TF-IDF similarity here is a dependency-free stand-in.
- **Guardrails** (`src/guardrails.py`):
  - **PII redaction:** regex finds emails/phones/cards and replaces them with tags.
  - **Prompt-injection detection:** flag inputs like "ignore previous instructions".
  - **Schema validation:** reject malformed structured output so downstream code can't crash.
- **Eval-in-CI:** `evals/test_evals.py` runs on every pull request (`.github/workflows/evals.yml`),
  so a prompt/model change that lowers quality or safety fails the build.

## 3. How the pieces fit - the flow
```
  user input --> [ INPUT GUARDRAILS ] --> your LLM/RAG system --> [ OUTPUT GUARDRAILS ] --> answer
                  PII redact + injection flag                       schema + safety validate
                                            \                      /
                                     [ EVALUATION HARNESS scores answers vs references ]
                                                       |
                                        runs in CI (evals/test_evals.py) -> blocks bad deploys
```

## 4. Code reading order
1. `src/evaluate.py` - the scorecard (run it; watch it catch the wrong answers).
2. `src/guardrails.py` - PII / injection / schema (run it; all checks assert true).
3. `evals/test_evals.py` - the same checks as CI tests (`python -m pytest evals`).
4. `notebooks/09_llm_eval_guardrails.ipynb` - the scorecard + guardrail-dashboard diagrams.

## 5. Common mistakes (and how to avoid them)
- **Vibe-checking instead of measuring** -> use a fixed eval set and a threshold.
- **Regex-only PII in production** -> graduate to Microsoft Presidio (more robust).
- **Trusting LLM-as-judge blindly** -> spot-check it against your own labels.
- **No CI gate** -> a prompt tweak silently regresses quality; run evals on every PR.

## 6. Check your understanding
1. Why use semantic similarity in addition to exact match?
2. Name two input guardrails and one output guardrail and what each prevents.
3. What does "eval-in-CI" stop from happening?
4. Why is LLM-as-a-judge powerful but risky?

## 7. Mini-glossary (full versions in /GLOSSARY.md)
eval set, exact match, semantic similarity, LLM-as-judge, guardrail, PII, prompt injection,
schema validation, eval-in-CI, observability/tracing.

## 8. Going deeper
- Promptfoo (eval + red-team): https://www.promptfoo.dev/
- Microsoft Presidio (PII): https://microsoft.github.io/presidio/
