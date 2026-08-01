# Project 11 - Multi-Agent Workflow (LangGraph / CrewAI)

> **New here? Start with [LESSON.md](LESSON.md)** - it teaches the concepts from scratch, then read the heavily-commented, runnable code in this folder.
> Orchestrate several specialized agents (planner, researcher, coder, critic) that collaborate to solve a task no single agent handles well.
**Phase 5: Agentic AI & Multimodal**  |  **Difficulty:** Advanced  |  **Est. time:** Week 18-20 (~40 hrs)

---

## Why (the point of this project)
'Multi-agent orchestration: LangGraph and CrewAI replaced ad-hoc loops' is a defining 2026 shift. Real automation comes from teams of agents with roles, handoffs, and a shared state - and the hard part is the ENGINEERING (state, control flow, reliability), which is familiar engineering territory. This is a standout, current portfolio piece.

## What you will build
A multi-agent system that decomposes a goal across role-specialized agents with explicit control flow and shared state - e.g. a 'research analyst' crew (planner -> researcher(s) -> writer -> critic/verifier) that produces a cited report, or a 'data analyst' crew that plans, queries, analyzes, and summarizes.

## Key concepts (learn these as you go)
| Concept | What it means |
|---|---|
| **Agent roles & specialization** | Different system prompts/tools per agent (planner, researcher, coder, critic) for division of labor. |
| **Orchestration patterns** | Supervisor/router, sequential pipeline, and debate/critic loops; choose by task structure. |
| **Shared state & handoffs** | A typed shared state object passed between nodes; explicit edges define who runs when (LangGraph graphs). |
| **Verification/critic agents** | A dedicated checker reduces errors by reviewing other agents' output before finalizing. |
| **Reliability & cost control** | Multi-agent systems multiply cost and failure surface; need budgets, retries, and termination conditions. |

## How - step by step
1. Pick a task that genuinely needs multiple roles (multi-step research report, or analyze-a-dataset-and-report).
2. Design the graph: nodes (agents), edges (control flow), and the shared state schema.
3. Implement agents with distinct prompts + tools (reuse tools from P10; retrieval from P7).
4. Add a supervisor/router to assign work and a critic to verify before finishing.
5. Add termination + budget guardrails (max iterations, cost ceiling) and structured tracing.
6. Evaluate end-to-end on several goals; measure quality, cost, and latency vs a single-agent baseline.
7. Visualize the agent graph and a sample run trace in the README.
8. Deploy a demo (Space) where users enter a goal and watch the crew work.

## Tech stack
LangGraph and/or CrewAI, an LLM (function calling), your P7 retrieval + P10 tools, Gradio, Docker, W&B

## Where it lives (your tools)
| Tool | How you use it here |
|---|---|
| **GitHub** | Graph definition, agent prompts, traces, eval vs single-agent baseline. |
| **Hugging Face** | Demo Space visualizing the multi-agent run. |
| **Weights & Biases** | Track quality/cost/latency across orchestration designs. |
| **Docker** | Containerize the multi-agent service. |

## Portfolio artifact
GitHub repo + live demo + W&B comparison report. Resume line: 'Built a multi-agent system (planner/researcher/critic) with LangGraph, outperforming a single-agent baseline on quality at controlled cost.'

## Definition of Done
- [ ] 3+ role-specialized agents collaborate via an explicit graph + shared state.
- [ ] A critic/verifier step measurably improves output quality.
- [ ] Budget + termination guardrails prevent runaway cost/loops.
- [ ] Eval shows the multi-agent system beats a single-agent baseline on the task.
- [ ] README includes a graph diagram and an annotated run trace.

## Stretch goals
- Add parallel agents (e.g. multiple researchers) and merge results.
- Add a human approval node for the final output.
- A/B two orchestration patterns (supervisor vs debate) and report.

## Resources
- [LangGraph multi-agent](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [CrewAI docs](https://docs.crewai.com/)
- [AutoGen (Microsoft)](https://microsoft.github.io/autogen/)

## Results (fill this in as you build)
- **Headline metric:** _e.g. AUC / F1 / accuracy / p95 latency_
- **Artifact links:** GitHub _ | HF _ | Kaggle _ | W&B _ | Demo _
- **One thing that broke and how I fixed it:** _..._
- **Build-in-public post:** _link_
