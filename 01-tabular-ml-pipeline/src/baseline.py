"""
baseline.py - the bar every real model must clear.

WHY: a score is meaningless without a reference. Is AUC 0.78 good? Only if the dumb
     baseline is ~0.50 and a linear model is ~0.72. Baselines turn "looks fine" into
     "measurably better", and occasionally reveal that your fancy model adds nothing.
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


def baseline_models():
    # name -> estimator; train.py wraps each in the SAME leakage-safe pipeline
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),   # predicts the common class
        "logistic_regression": LogisticRegression(max_iter=1000),      # strong, simple linear bar
    }
