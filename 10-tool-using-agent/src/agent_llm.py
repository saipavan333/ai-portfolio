"""
agent_llm.py - PRODUCTION agent: a real LLM with function calling (LangGraph).

    python -m src.agent_llm        # needs langgraph + an LLM key

WHAT : replaces the rule-based brain with an LLM that decides which tool to call; LangGraph runs
       the ReAct loop and the guardrails.
WHY  : the loop mechanics are identical to agent_demo.py - only the decision-maker is a real model.
HOW  : define @tool functions -> create_react_agent(llm, tools) -> invoke with a goal.
WHERE: production layer; agent_demo.py teaches the loop. Add step/cost budgets + tracing + an eval suite.

Requires: langgraph, langchain, langchain-openai (or another provider).
"""
from __future__ import annotations


def build_agent():
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI

    @tool
    def calculator(expression: str) -> float:
        """Evaluate a math expression like '23*19'."""
        return eval(expression, {"__builtins__": {}}, {})

    @tool
    def web_search(query: str) -> str:
        """Search the web and return a short snippet."""
        raise NotImplementedError("wire up a search API")

    llm = ChatOpenAI(model="gpt-4o-mini")                # the 'brain' that decides actions
    return create_react_agent(llm, tools=[calculator, web_search])


def main():
    agent = build_agent()
    out = agent.invoke({"messages": [("user", "What is 23*19 plus the population of France?")]})
    print(out["messages"][-1].content)
    # TODO: max-steps + cost budget, tool allow-list, structured tracing, task-success eval.


if __name__ == "__main__":
    main()
