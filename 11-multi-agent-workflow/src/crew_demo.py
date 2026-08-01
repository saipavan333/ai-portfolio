"""
crew_demo.py - a working multi-agent crew (runnable, no API key).

    python -m src.crew_demo

WHAT : four specialised agents (planner, researcher, writer, critic) share a state dict and pass
       work along a pipeline. The critic loops the draft back for revision until it covers every
       planned topic - and the demo asserts the final draft is complete.
WHY  : real automation comes from a TEAM of focused agents with handoffs and a verifier - not one
       do-everything agent. The hard part is the engineering (state, control flow) - familiar engineering territory.
HOW  : planner -> researcher -> (writer -> critic) loop, capped at a few attempts (a guardrail).
WHERE: the concept layer. crew_langgraph.py is production (real LLM agents in a LangGraph StateGraph).
       See notebooks/11_multi_agent_workflow.ipynb for the agent-graph + coverage diagrams.
"""
from __future__ import annotations

KB = {
    "benefits":   "Renewable energy cuts greenhouse emissions and lowers long-run costs.",
    "challenges": "Its main challenges are intermittency and high upfront capital.",
    "examples":   "Solar and wind are the fastest-growing renewable sources worldwide.",
}


def planner(state):
    state["subtasks"] = ["benefits", "challenges", "examples"]
    state["log"].append(("planner", f"plan = {state['subtasks']}"))


def researcher(state):
    state["findings"] = {t: KB.get(t, "(no data)") for t in state["subtasks"]}
    state["log"].append(("researcher", f"gathered {len(state['findings'])} findings"))


def writer(state, revision):
    # the FIRST draft (revision 0) deliberately forgets the last topic, so the critic can catch it
    topics = state["subtasks"] if revision else state["subtasks"][:-1]
    state["draft"] = " ".join(state["findings"][t] for t in topics)
    state["log"].append(("writer", f"draft covers {len(topics)}/{len(state['subtasks'])} topics"))


def critic(state):
    missing = [t for t in state["subtasks"] if state["findings"][t] not in state["draft"]]
    state["log"].append(("critic", "approved" if not missing else f"missing {missing}"))
    return not missing


def run_crew(goal: str, max_revisions: int = 3):
    state = {"goal": goal, "subtasks": [], "findings": {}, "draft": "", "log": []}
    planner(state)
    researcher(state)
    for attempt in range(max_revisions):
        writer(state, revision=attempt)
        if critic(state):
            break
    return state


def main():
    state = run_crew("Write a short brief on renewable energy.")
    for who, what in state["log"]:
        print(f"  {who:11s}: {what}")
    print("\nFINAL BRIEF:\n" + state["draft"])
    covered = all(state["findings"][t] in state["draft"] for t in state["subtasks"])
    assert covered, "the critic loop should yield a draft covering all planned topics"
    print("\n[crew] OK - critic loop produced a complete draft")


if __name__ == "__main__":
    main()
