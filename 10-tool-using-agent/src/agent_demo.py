"""
agent_demo.py - a working tool-using agent loop (runnable, no API key).

    python -m src.agent_demo

WHAT : an agent that solves a multi-step task by reasoning, calling tools, observing results, and
       looping (the ReAct pattern), with a max-steps guardrail and a full trace.
WHY  : agentic AI is the defining 2026 trend. Before orchestrating many agents you must master ONE:
       the reason -> act -> observe loop and its guardrails.
HOW  : a rule-based 'brain' (stand-in for the LLM) decides the next tool; the loop executes it.
WHERE: the concept layer. agent_llm.py is production (a real LLM decides via function calling).
       See notebooks/10_tool_using_agent.ipynb for the ReAct + trace diagrams.
"""
from __future__ import annotations


def calculator(expr: str):
    return eval(expr, {"__builtins__": {}}, {})     # restricted eval (no builtins)


def letter_count(word: str):
    return len(word)


TOOLS = {"calculator": calculator, "letter_count": letter_count}


def decide(scratch: dict):
    """The 'brain'. A real LLM does this via function calling; here it's rule-based.
    Returns (thought, action, argument). action='final' ends the loop."""
    if "product" not in scratch:
        return ("First compute 23 * 19.", "calculator", "23*19")
    if "letters" not in scratch:
        return ("Now count letters in 'transformer'.", "letter_count", "transformer")
    return ("Combine the two results.", "final", scratch["product"] + scratch["letters"])


def run_agent(goal: str, max_steps: int = 6):
    scratch, trace = {}, []
    for step in range(1, max_steps + 1):
        thought, action, arg = decide(scratch)
        if action == "final":
            trace.append((step, thought, "FINAL", arg))
            return arg, trace
        obs = TOOLS[action](arg)                                 # ACT
        scratch["product" if action == "calculator" else "letters"] = obs   # OBSERVE
        trace.append((step, thought, f"{action}('{arg}')", obs))
    return None, trace                                           # guardrail: stop


def main():
    goal = "Compute 23 * 19, then add the number of letters in 'transformer'."
    answer, trace = run_agent(goal)
    print("GOAL:", goal)
    for step, thought, action, obs in trace:
        print(f"  step {step}: {thought}  ->  {action}  ->  {obs}")
    print("FINAL ANSWER:", answer)
    assert answer == 23 * 19 + len("transformer") == 448, "agent should chain tools to 448"
    print("[agent] OK - the agent chained two tools correctly")


if __name__ == "__main__":
    main()
