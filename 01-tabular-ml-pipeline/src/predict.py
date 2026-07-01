"""
predict.py - load the saved pipeline and score new rows.

WHY: training and serving MUST use identical preprocessing. Because we saved the whole
     Pipeline (preprocess + model), inference is just predict_proba - there is no way to
     accidentally apply a different scaler/encoder at serve time. This is why we bundled.
Run: python -m src.predict      (run train.py first so the artifact exists)
"""
import joblib
from src import data as D


def predict(df, model_path: str = "artifacts/model.joblib"):
    """Return churn probability for each row of df."""
    model = joblib.load(model_path)
    return model.predict_proba(df)[:, 1]


if __name__ == "__main__":
    demo = D.make_demo_data(5)              # pretend these are 5 new customers
    X = demo.drop(columns=[D.TARGET])
    print("churn probabilities:", predict(X).round(3))
