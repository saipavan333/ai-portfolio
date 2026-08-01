# Project 10 - Single Agent with Tool Use (ReAct + Function Calling)

> **New here? Start with [LESSON.md](LESSON.md)** - it teaches the concepts from scratch, then read the heavily-commented, runnable code in this folder.
> Build an LLM agent that reasons, calls tools (search, calculator, your APIs), observes results, and loops until the task is done.
**Phase 5: Agentic AI & Multimodal**  |  **Difficulty:** Advanced  |  **Est. time:** Week 17-18 (~35 hrs)

---

## Why (the point of this project)
Agentic AI is THE defining trend of 2026 - Gartner projects 40% of enterprise apps will embed agents by end of 2026. Before orchestrating many agents (P11), you must deeply understand one: the reason->act->observe loop, tool schemas, and the failure modes (loops, hallucinated tools, runaway cost). This is the foundation of everything agentic.

## What you will build
A single agent that, given a goal, decides which tool to call, executes it, reads the result, and iterates to a final answer. Tools include web/search, a calculator, and at least one custom tool (e.g. query your RAG from P7 or a database). Built on the function-calling/ReAct pattern with proper guardrails on steps and cost.

## Key concepts (learn these as you go)
| Concept | What it means |
|---|---|
| **Agent loop (ReAct)** | Interleave Reasoning and Acting: the model thinks, picks a tool, acts, observes, repeats until done. |
| **Tools & function calling** | Expose functions with typed schemas; the model emits structured calls the runtime executes. |
| **Planning vs reacting** | Plan-then-execute vs react-step-by-step; trade-offs in reliability and latency. |
| **Memory & state** | Short-term scratchpad vs longer-term memory across steps/sessions. |
| **Agent guardrails** | Max-steps, tool allow-lists, cost/timeout budgets, and human-in-the-loop for risky actions. |

## How - step by step
1. Pick a useful task (e.g. 'research question -> cited answer', or 'query my data -> summarize').
2. Define 3-4 tools with clean typed schemas (search, calculator, custom RAG/db tool).
3. Implement the agent loop using a framework (LangGraph) or hand-rolled function calling.
4. Add guardrails: max steps, tool allow-list, per-run cost/time budget, and structured logging.
5. Trace every step (thought, tool, args, observation) for debuggability.
6. Test on a suite of tasks; measure success rate, steps, and cost; log to W&B.
7. Wrap in a chat UI (Gradio) that shows the agent's intermediate steps.
8. Document failure modes you hit (loops, wrong tool, hallucinated args) and your fixes.

## Tech stack
LangGraph (or hand-rolled), an LLM with function calling, your tools/APIs, Gradio, W&B

## Where it lives (your tools)
| Tool | How you use it here |
|---|---|
| **GitHub** | Agent code + tool definitions + trace logs + eval suite. |
| **Hugging Face** | Optional Space showing the agent's step-by-step reasoning. |
| **Weights & Biases** | Track success rate, step count, and cost across task runs. |
| **Docker** | Containerize the agent + tools for consistent runs. |

## Portfolio artifact
GitHub repo + live agent demo + W&B run report. Resume line: 'Built a tool-using ReAct agent (function calling, step/cost guardrails, full tracing) with an XX% task success rate.'

## Definition of Done
- [ ] Agent selects and calls the right tools to complete multi-step tasks.
- [ ] Guardrails enforce max-steps and a cost/time budget (no runaway loops).
- [ ] Every step is traced (thought/tool/args/observation).
- [ ] An evaluation suite reports success rate over multiple tasks.
- [ ] Demo UI shows intermediate reasoning, not just the final answer.

## Stretch goals
- Add long-term memory (vector store) across sessions.
- Add human-in-the-loop approval for risky tools.
- Compare a 2026 reasoning model vs a standard model as the agent's brain.

## Resources
- [LangGraph docs](https://langchain-ai.github.io/langgraph/)
- [ReAct paper](https://arxiv.org/abs/2210.03629)
- [Anthropic - Building effective agents](https://www.anthropic.com/research/building-effective-agents)

## Results (fill this in as you build)
- **Headline metric:** _e.g. AUC / F1 / accuracy / p95 latency_
- **Artifact links:** GitHub _ | HF _ | Kaggle _ | W&B _ | Demo _
- **One thing that broke and how I fixed it:** _..._
- **Build-in-public post:** _link_
