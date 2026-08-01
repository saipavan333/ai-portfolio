"""
train.py - measure the DATA-CENTRIC lift: SAME model, raw features vs engineered features.

    python -m src.train --demo

WHY  : this project's thesis is that improving the DATA beats fiddling with the model. So we
       FREEZE the model (one logistic regression) and change only the features, then compare
       cross-validated AUC. If engineered features win, we've proven the data-centric point.
HOW  : 1) make raw data  2) run the data-quality gate  3) CV-score raw features
       4) CV-score engineered features  5) report the lift  6) save metrics.json
WHERE: the orchestrator - calls demo_data, data_quality, features, modeling.
"""
import argparse, json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from src.demo_data import make_raw_demo
from src.data_quality import validate
from src.features import make_features
from src.modeling import split_xy, build_pipeline


def cv_auc(df):
    # fixed model on purpose: only the FEATURES differ between the two calls
    X, y, num, cat = split_xy(df)
    pipe = build_pipeline(num, cat, LogisticRegression(max_iter=1000))
    return float(cross_val_score(pipe, X, y, cv=5, scoring="roc_auc").mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use built-in synthetic data")
    ap.parse_args()

    # === STEP 1: get raw data ==============================================
    raw = make_raw_demo()

    # === STEP 2: data-quality gate (fail fast on bad input) ================
    validate(raw,
             required_cols=["tenure", "MonthlyCharges", "TotalCharges", "Contract", "Churn"],
             ranges={"tenure": (0, 80), "MonthlyCharges": (0, 200)})

    # === STEP 3+4: same model, raw vs engineered features ==================
    auc_raw = cv_auc(raw)
    auc_eng = cv_auc(make_features(raw))

    # === STEP 5: report the data-centric lift ==============================
    lift = auc_eng - auc_raw
    print(f"[result] AUC raw features        = {auc_raw:.3f}")
    print(f"[result] AUC engineered features = {auc_eng:.3f}")
    print(f"[result] data-centric lift       = {lift:+.3f}  "
          f"({'features helped' if lift > 0 else 'no gain - iterate on features'})")

    # === STEP 6: persist metrics (DVC tracks this file) ====================
    with open("metrics.json", "w") as f:
        json.dump({"auc_raw": round(auc_raw, 4),
                   "auc_engineered": round(auc_eng, 4),
                   "lift": round(lift, 4)}, f, indent=2)
    print("[save] wrote metrics.json")


if __name__ == "__main__":
    main()
