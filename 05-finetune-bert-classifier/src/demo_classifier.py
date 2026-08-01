"""
demo_classifier.py - a real, runnable text classifier (no GPU, no downloads).

    python -m src.demo_classifier

WHAT : trains a TF-IDF + logistic-regression sentiment classifier on a tiny built-in dataset and
       classifies new sentences. This is your strong BASELINE - the bar a fine-tuned BERT must beat.
WHY  : before reaching for a giant transformer, prove a simple model's score. Often it's already
       good, and it tells you whether the transformer was worth the compute.
HOW  : tokenize text into TF-IDF vectors -> logistic regression -> predict probabilities.
WHERE: the baseline layer. finetune_bert.py is the production transformer layer.
       See notebooks/05_nlp_transformers.ipynb for the word-importance + embedding diagrams.
"""
from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TEXTS = [
    "i love this product it works great", "absolutely fantastic and easy to use",
    "wonderful experience highly recommend", "best purchase i have made",
    "great quality and fast delivery", "happy with it works perfectly",
    "excellent value really pleased", "amazing support smooth setup",
    "terrible it broke after one day", "awful quality very disappointed",
    "worst purchase i regret it", "hate it nothing works",
    "bad experience slow and buggy", "horrible support painful setup",
    "useless complete waste of money", "frustrating and unreliable",
]
LABELS = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]   # 1 = positive, 0 = negative


def build_classifier() -> Pipeline:
    # TF-IDF turns each sentence into a weighted word-count vector; logistic regression separates them
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("lr", LogisticRegression(max_iter=1000)),
    ])


def main() -> None:
    clf = build_classifier()
    clf.fit(TEXTS, LABELS)
    new = ["this is great i am very happy", "terrible and a waste i regret buying",
           "the setup was smooth and reliable", "buggy slow and disappointing"]
    print("[baseline] predictions on unseen sentences:")
    for s, p in zip(new, clf.predict_proba(new)[:, 1]):
        print(f"  P(positive)={p:.2f} -> {'POSITIVE' if p > 0.5 else 'NEGATIVE'} | {s}")
    # the first two should be positive/negative respectively
    probs = clf.predict_proba(new)[:, 1]
    assert probs[0] > 0.5 and probs[1] < 0.5, "baseline should classify clear sentences correctly"
    print("[baseline] OK - sanity checks passed")


if __name__ == "__main__":
    main()
