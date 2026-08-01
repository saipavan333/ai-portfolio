"""
crew_langgraph.py - PRODUCTION multi-agent: real LLM agents in a LangGraph StateGraph.

    python -m src.crew_langgraph        # needs langgraph + an LLM key

WHAT : the same plan -> research -> write -> critique flow as crew_demo.py, but each node is a real
       LLM agent and the critic conditionally loops back to the writer.
WHY  : identical structure to the demo; only the agents are real models now.
HOW  : define a typed State + node functions -> add nodes/edges -> conditional edge for the loop.
WHERE: production layer; crew_demo.py teaches the orchestration.

Requires: langgraph, langchain-openai.
"""
from __future__ import annotations
from typing import TypedDict, List


class State(TypedDict):
    goal: str
    plan: List[str]
    findings: List[str]
    draft: str
    review: str


def build_app():
    from langgraph.graph import StateGraph, END
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")

    def planner(s):    s["plan"] = llm.invoke(f"List 3 angles for: {s['goal']}").content.split("\n"); return s
    def researcher(s): s["findings"] = [llm.invoke(f"Research: {a}").content for a in s["plan"]]; return s
    def writer(s):     s["draft"] = llm.invoke(f"Write a brief from: {s['findings']}").content; return s
    def critic(s):     s["review"] = llm.invoke(f"Critique; say APPROVED or REVISE: {s['draft']}").content; return s

    g = StateGraph(State)
    for name, fn in [("planner", planner), ("researcher", researcher), ("writer", writer), ("critic", critic)]:
        g.add_node(name, fn)
    g.set_entry_point("planner")
    g.add_edge("planner", "researcher"); g.add_edge("researcher", "writer"); g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", lambda s: "writer" if "REVISE" in s["review"] else END)  # loop-back
    return g.compile()


def main():
    app = build_app()
    print(app.invoke({"goal": "Write a brief on renewable energy."})["draft"])
    # TODO: budget iterations + cost, run researchers in parallel, add a human-approval node.


if __name__ == "__main__":
    main()
