# Lesson 11 - Multi-Agent Workflows

> Read with `src/crew_demo.py` open; run it (`python -m src.crew_demo`). See
> `notebooks/11_multi_agent_workflow.ipynb` for the agent-graph + coverage diagrams.

## 0. What you'll be able to do after this
- Explain **agent roles**, **shared state**, **orchestration**, and the value of a **critic**.
- Run a working crew and read the production LangGraph version (`src/crew_langgraph.py`).

## 1. The big picture (why this project exists)
Real automation comes from a **team** of specialised agents - planner, researcher, writer, critic -
that hand work to each other, not one do-everything agent. The hard part is the **engineering**
(state, control flow, reliability), which is exactly your strength.

## 2. Foundations from scratch (the basics)
- **Role specialisation:** each agent has one job and its own instructions. Focused agents make
  fewer mistakes. `crew_demo.py` has planner/researcher/writer/critic.
- **Shared state:** one object (a dict here) carries the goal, plan, findings, and draft between
  agents.
- **Orchestration:** the control flow. Here it's a pipeline with a **loop-back**: the critic sends
  the draft back to the writer until every planned topic is covered. The demo's first draft
  deliberately omits a topic; the critic catches it; the revision fixes it (asserted complete).
- **Critic / verifier:** a dedicated checker before finalising - one of the biggest reliability
  wins in agentic systems.
- **Guardrail:** cap the number of revisions so the loop can't run forever.

## 3. How the pieces fit - the flow
```
  planner -> researcher -> writer -> critic --(missing topics? REVISE)--> writer
                                       |
                                       +--(all covered? APPROVE)--> final brief
            (crew_demo.py - shared state dict; crew_langgraph.py - real LLM agents in a StateGraph)
```

## 4. Code reading order
1. `src/crew_demo.py` - the four agents + `run_crew` loop (run it; read the log).
2. `notebooks/11_multi_agent_workflow.ipynb` - the agent graph + coverage bar.
3. `src/crew_langgraph.py` - the production StateGraph with a conditional loop-back edge.

## 5. Common mistakes (and how to avoid them)
- **No critic/verifier** -> errors slip through; add a checker before finalising.
- **Unbounded loops** -> cap revisions and budget cost.
- **One overloaded agent** -> split into focused roles; it's more reliable and debuggable.
- **No shared schema** -> agents miscommunicate; use a typed state.

## 6. Check your understanding
1. Why does role specialisation beat one do-everything agent?
2. What does the shared state carry, and who reads/writes it?
3. What did the critic catch in the demo, and how was it fixed?
4. Why cap the number of revisions?

## 7. Mini-glossary (full versions in /GLOSSARY.md)
role specialisation, shared state, handoff, orchestration, supervisor, critic/verifier, loop-back.

## 8. Going deeper
- LangGraph multi-agent: https://langchain-ai.github.io/langgraph/concepts/multi_agent/
- CrewAI: https://docs.crewai.com/
