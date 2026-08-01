# Lesson 10 - AI Agents: Tool Use & the ReAct Loop

> Read with `src/agent_demo.py` open; run it (`python -m src.agent_demo`). See
> `notebooks/10_tool_using_agent.ipynb` for the ReAct + trace diagrams.

## 0. What you'll be able to do after this
- Explain the **agent loop** (Thought -> Action -> Observation -> repeat -> Answer).
- Explain **tools / function calling** and why agents beat a bare LLM on multi-step tasks.
- Run a working agent and read the production LangGraph version (`src/agent_llm.py`).

## 1. The big picture (why this project exists)
A bare LLM only predicts text - it can't reliably do arithmetic or look things up. An **agent**
gives the model **tools** and a **loop**: think, call a tool, read the result, repeat until done.
Agentic AI is the defining 2026 trend; master one agent before orchestrating many (Project 11).

## 2. Foundations from scratch (the basics)
- **Tool:** a normal function the agent can call (calculator, search, a DB query). Each has a name
  and described input so the model knows when/how to use it. `agent_demo.py` has two tools.
- **Function calling:** the model emits a structured "call tool X with these args"; your code runs
  it and feeds the result back.
- **The ReAct loop:** **Reason** (decide) -> **Act** (call a tool) -> **Observe** (read result) ->
  repeat. `run_agent()` implements exactly this; the demo chains `calculator` then `letter_count`
  to reach **448** (asserted).
- **Guardrails:** a **max-steps** cap (in `run_agent`) stops infinite loops; production adds a
  cost/time budget and a tool allow-list.
- **The 'brain':** `decide()` is a rule-based stand-in so the demo runs key-free. In production an
  **LLM** makes these decisions (`agent_llm.py`) - the loop is identical.

## 3. How the pieces fit - the flow
```
        (concept - runs on CPU)                         (production - agent_llm.py)
  goal -> decide() -> tool -> observe -> loop           goal -> LLM decides -> tool -> observe -> loop
            (agent_demo.py, max_steps guardrail)         (LangGraph create_react_agent + budgets)
                       |
                final answer (448)
```

## 4. Code reading order
1. `src/agent_demo.py` - tools, `decide`, `run_agent` (run it first; read the trace).
2. `notebooks/10_tool_using_agent.ipynb` - the ReAct loop + trace diagrams.
3. `src/agent_llm.py` - the real LLM agent via LangGraph function calling.

## 5. Common mistakes (and how to avoid them)
- **No step limit** -> infinite loops / runaway cost. Always cap steps (and cost).
- **Vague tool descriptions** -> the model calls the wrong tool. Describe inputs precisely.
- **No trace** -> impossible to debug. Log every (thought, tool, args, observation).
- **Letting tools do dangerous actions** -> allow-list tools; require approval for risky ones.

## 6. Check your understanding
1. Why can an agent answer "23*19 + ..." reliably when a bare LLM can't?
2. What three things happen in each iteration of the ReAct loop?
3. What does the max-steps guardrail protect against?
4. In production, what replaces the rule-based `decide()`?

## 7. Mini-glossary (full versions in /GLOSSARY.md)
agent, tool, function calling, ReAct, scratchpad/memory, guardrail, trace, allow-list.

## 8. Going deeper
- LangGraph: https://langchain-ai.github.io/langgraph/
- "ReAct" paper: https://arxiv.org/abs/2210.03629
