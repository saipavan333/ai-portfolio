"""
evaluate.py - an LLM evaluation harness (fully runnable, no GPU).

    python -m src.evaluate

WHAT : scores model answers against references with exact-match AND semantic similarity, and prints
       a pass/fail scorecard. This is the test suite you run before every deploy.
WHY  : 'it looked fine once' is not evidence. A fixed eval set + metrics turns quality into a number
       you can defend and track over time.
HOW  : for each (question, reference, answer): exact match + TF-IDF cosine similarity; pass if >= threshold.
WHERE: pair with guardrails.py; wire both into CI via evals/test_evals.py.
"""
from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

THRESHOLD = 0.30

EVALSET = [
    {"q": "Capital of France?",       "ref": "Paris",                "ans": "Paris"},
    {"q": "Capital of Japan?",        "ref": "Tokyo",                "ans": "The capital of Japan is Tokyo."},
    {"q": "What is 2 + 2?",           "ref": "4",                    "ans": "5"},
    {"q": "Largest planet?",          "ref": "Jupiter",              "ans": "Jupiter is the largest planet."},
    {"q": "Refund processing time?",  "ref": "5 to 7 business days", "ans": "about a month"},
]


def score(evalset=EVALSET, threshold=THRESHOLD):
    vec = TfidfVectorizer().fit([d["ref"] for d in evalset] + [d["ans"] for d in evalset])
    results = []
    for d in evalset:
        exact = d["ans"].strip().lower() == d["ref"].strip().lower()
        sem = float(cosine_similarity(vec.transform([d["ref"]]), vec.transform([d["ans"]]))[0, 0])
        results.append({**d, "exact": exact, "sem": sem, "pass": sem >= threshold})
    return results


def main():
    results = score()
    for r in results:
        print(f"pass={r['pass']!s:5} exact={r['exact']!s:5} sem={r['sem']:.2f}  Q: {r['q']}")
    passed = sum(r["pass"] for r in results)
    print(f"\nSCORECARD: {passed}/{len(results)} passed")
    # the '2+2=5' and 'about a month' answers must FAIL - proving the harness catches errors
    assert not any(r["pass"] for r in results if r["q"].startswith("What is 2")), "must catch a wrong answer"
    print("[eval] OK - the harness catches wrong answers")


if __name__ == "__main__":
    main()
