"""Small but REAL tests, so CI checks something meaningful (run: python -m pytest -q)."""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from src.pipeline import build_pipeline
from src import data as D


def test_build_pipeline_returns_object():
    assert build_pipeline(["a"], ["b"], LogisticRegression()) is not None


def test_pipeline_learns_signal():
    # the synthetic data contains real signal, so logistic regression should beat 0.70 AUC
    df = D.make_demo_data(1500)
    X, y, num, cat = D.split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0)
    pipe = build_pipeline(num, cat, LogisticRegression(max_iter=1000))
    pipe.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    assert auc > 0.70, f"expected real signal, got AUC={auc:.2f}"
