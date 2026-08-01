# Lesson 15 - Capstone: An End-to-End Agentic AI Product

> Read with `src/capstone.py` open; run it (`python -m src.capstone`) - it's **fully runnable** and
> proves the whole pipeline end-to-end. See `notebooks/15_capstone_end_to_end.ipynb` for the
> architecture + scorecard diagrams, and `README_DESIGN.md` for the design-doc template.

## 0. What you'll be able to do after this
- Explain how the program's pieces **compose** into one product (system design).
- Run a complete pipeline (RAG + agent + guardrails + eval) and read the integration code.
- Know how to scope, build, evaluate, deploy, and present a flagship project.

## 1. The big picture (why this project exists)
One deep, complete product beats ten toys. The capstone is the flagship to lead a resume and interviews
with: proof you can take a real problem from data to a deployed, evaluated, monitored, agentic system -
the full stack employers pay for. Tie it to a data-engineering background for maximum impact.

## 2. Foundations from scratch (the basics)
- **Compose, don't reinvent:** a product is the *integration* of components you already built -
  retrieval (P07), an agent with a tool (P10), guardrails (P09), evaluation (P09), and serving (P14).
  `src/capstone.py` wires them into one `assistant()` pipeline.
- **The pipeline:** input guardrail -> retrieve -> the agent decides (answer directly, or use the
  calculator) -> cited answer -> output guardrail. The demo answers a docs question, computes a
  yearly cost (40x12 = 480) by combining RAG + a tool, and blocks an injection attempt.
- **End-to-end evaluation:** score the WHOLE product on real tasks, not just parts. The demo asserts
  a 4/4 end-to-end score and that injection is blocked.
- **Observability:** every stage is logged into a trace so you can debug and improve.

## 3. How the pieces fit - the flow
```
  question -> input guardrail -> retrieve (RAG) -> agent (+calculator tool) -> cited answer
                                                          |                          |
                                                 output guardrail <-----------------+
                                                          |
                                          end-to-end evaluation (scorecard)  ->  deploy (P14)
          (src/capstone.py - fully runnable; productionise each box with its project's real version)
```

## 4. Code reading order
1. `src/capstone.py` - `assistant()` (the integrated pipeline) + the end-to-end eval (run it first).
2. `notebooks/15_capstone_end_to_end.ipynb` - the architecture + scorecard diagrams.
3. `README_DESIGN.md` - the 1-page design-doc template (users, scope, metrics, risks).

## 5. How to ship it (the playbook)
1. **Scope** (design doc): users, job-to-be-done, success metrics, risks.
2. **Build for real:** neural RAG (P07), an LLM agent (P10/P11), full guardrails + eval (P09).
3. **Deploy:** containerize with the Docker + vLLM stack (P14); add a UI; host a demo Space.
4. **Prove:** run the end-to-end eval in CI; record a 2-3 min demo video; write an honest README.
5. **Present:** live demo + repo + blog post + LinkedIn launch.

## 6. Check your understanding
1. Which earlier projects does the capstone integrate, and what does each contribute?
2. How did the assistant answer a question that needed both a document AND a calculation?
3. What does "end-to-end evaluation" measure that component tests don't?
4. What's the headline you'd put on a resume for this project?

## 7. Mini-glossary (full versions in /GLOSSARY.md)
system design, end-to-end evaluation, observability, integration, capstone.

## 8. Going deeper
- Google People + AI Guidebook: https://pair.withgoogle.com/guidebook/
- Hugging Face Spaces (deploy): https://huggingface.co/docs/hub/spaces
