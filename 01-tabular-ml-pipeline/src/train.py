"""
train.py - the training entry point. Run me three ways:

    python -m src.train --demo                      # zero-setup synthetic run (learn the flow)
    python -m src.train --data data/telco.csv       # the real Kaggle dataset
    python -m src.train --demo --wandb              # also log to Weights & Biases

WHAT : load data -> build baselines + a gradient-boosting model -> score them with
       cross-validation -> pick the best -> evaluate ONCE on a held-out test set ->
       save the fitted pipeline + metrics.
WHY  : this is THE canonical supervised-ML flow. Every later project reuses this shape:
       data -> split -> candidates -> validate -> select -> test once -> persist.
HOW  : follow the STEP banners below, in order.
WHERE: the orchestrator - it calls data.py, pipeline.py, and baseline.py.
"""
from __future__ import annotations
import argparse, json, os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from src import data as D
from src.pipeline import build_pipeline
from src.baseline import baseline_models


def get_gbm():
    """Return (name, model) for the gradient-boosting candidate.
    why: XGBoost is the Kaggle favourite, but may not be installed. Fall back to
         scikit-learn's HistGradientBoosting so this script ALWAYS runs."""
    try:
        from xgboost import XGBClassifier
        return "xgboost", XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, eval_metric="logloss")
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return "hist_gbm", HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=400)


def maybe_wandb(use: bool):
    """Start a W&B run if asked AND available; otherwise return None and carry on."""
    if not use:
        return None
    try:
        import wandb
        wandb.init(project="ai-portfolio-p01", config={"task": "telco-churn"})
        return wandb
    except Exception as e:
        print(f"[wandb] disabled ({e})")
        return None


def main():
    ap = argparse.ArgumentParser(description="Train a churn classifier.")
    ap.add_argument("--data", help="path to a Telco churn CSV. Omit to use --demo data")
    ap.add_argument("--demo", action="store_true", help="use built-in synthetic data")
    ap.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    args = ap.parse_args()

    # === STEP 1: load data =================================================
    if args.data:
        df = D.clean_telco(D.load_raw(args.data))
    else:
        if not args.demo:
            print("No --data given; using --demo synthetic data.")
        df = D.make_demo_data()
    X, y, num_cols, cat_cols = D.split_xy(df)
    D.basic_checks(X, y)

    # === STEP 2: stratified train/test split ===============================
    # why: stratify keeps the churn rate identical in train & test; the test set is the
    #      "final exam" and is scored exactly ONCE, at the very end.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    run = maybe_wandb(args.wandb)

    # === STEP 3: assemble candidate models (baselines + GBM) ===============
    candidates = baseline_models()
    gbm_name, gbm = get_gbm()
    candidates[gbm_name] = gbm

    # === STEP 4: cross-validate each candidate on TRAIN ONLY ===============
    cv_scores = {}
    for name, model in candidates.items():
        pipe = build_pipeline(num_cols, cat_cols, model)            # fresh, leakage-safe pipe
        auc = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc")
        cv_scores[name] = float(auc.mean())
        print(f"[cv] {name:22s} AUC = {auc.mean():.3f} +/- {auc.std():.3f}")
        if run:
            run.log({f"cv_auc/{name}": auc.mean()})

    # === STEP 5: pick the best by CV, refit on ALL training data ===========
    best_name = max(cv_scores, key=cv_scores.get)
    print(f"[select] best by cross-validation = {best_name}")
    best = build_pipeline(num_cols, cat_cols, candidates[best_name])
    best.fit(X_tr, y_tr)

    # === STEP 6: evaluate ONCE on the held-out test set ====================
    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba > 0.5).astype(int)
    test_auc = roc_auc_score(y_te, proba)
    print(f"[test] {best_name} AUC = {test_auc:.3f}")
    print(classification_report(y_te, pred, digits=3))
    print("confusion matrix [[TN FP][FN TP]]:\n", confusion_matrix(y_te, pred))

    # === STEP 7: persist the model + metrics ===============================
    os.makedirs("artifacts", exist_ok=True)
    import joblib
    joblib.dump(best, "artifacts/model.joblib")     # the WHOLE pipeline (preprocess + model)
    metrics = {"best_model": best_name, "test_auc": round(test_auc, 4),
               "cv_auc": {k: round(v, 4) for k, v in cv_scores.items()}}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[save] wrote artifacts/model.joblib and metrics.json")
    if run:
        run.log({"test_auc": test_auc})
        run.finish()

    # Extend: (1) tune the GBM with Optuna; (2) add a SHAP summary plot;
    #            (3) try class_weight / threshold tuning for the imbalance.


if __name__ == "__main__":
    main()
