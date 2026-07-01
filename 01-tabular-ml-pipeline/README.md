# Project 01 - End-to-End Tabular ML Pipeline

> **New here? Start with [LESSON.md](LESSON.md)** - it teaches the concepts from scratch, then read the heavily-commented code in `src/`.
> Predict a real-world outcome from tabular data with a clean, reproducible scikit-learn pipeline. Your ETL skills shine here.
**Phase 1: Foundations & Data-Centric ML**  |  **Difficulty:** Beginner  |  **Est. time:** Week 1-2 (~35 hrs)

---

## Why (the point of this project)
This is the fastest way to turn existing data skills into an AI artifact. 80% of enterprise ML is still tabular (churn, fraud, pricing, demand). You already understand data better than most ML beginners, so start where you are strongest and get a win on the board. It also forces the habits the rest of the program relies on: train/validation/test discipline, leakage avoidance, metrics, and reproducibility.

## What you will build
A repo + Kaggle notebook that takes a raw tabular dataset and produces a trained, evaluated model behind a single `predict()` function. You will compare a linear baseline, a random forest, and gradient boosting (XGBoost/LightGBM), pick the best by a proper validation scheme, and explain it with feature importance / SHAP.

## Key concepts (learn these as you go)
| Concept | What it means |
|---|---|
| **Supervised learning** | Learn a function from labeled examples (X -> y). Classification (categories) vs regression (numbers). |
| **Train/val/test split & cross-validation** | Hold out data the model never sees in training so your score estimates real-world performance. CV averages over multiple folds. |
| **Data leakage** | When info from the future/target sneaks into features (e.g. fitting a scaler on the whole dataset). The #1 cause of 'great in notebook, terrible in prod'. Your ETL eye for lineage helps here. |
| **Gradient boosting** | Ensembles of small trees added sequentially, each fixing the last one's errors. XGBoost/LightGBM dominate tabular Kaggle. |
| **Evaluation metrics** | Accuracy is often misleading. Use ROC-AUC/F1 for imbalanced classification, RMSE/MAE for regression. Pick the metric the business cares about. |
| **Model explainability (SHAP)** | Attribute a prediction to each feature so stakeholders trust the model. |

## How - step by step
1. Pick a dataset on Kaggle with a clear target (suggested: Telco Customer Churn, Home Credit, or a UCI dataset). Frame the business question in one sentence.
2. Do EDA in a notebook: missing values, distributions, target balance, obvious leakage. Write down 3 hypotheses about what predicts the target.
3. Build a scikit-learn `Pipeline` (imputer -> encoder -> scaler -> model) so preprocessing is fit ONLY on training folds. This single habit prevents most leakage.
4. Establish a dumb baseline (majority class / mean) and a logistic/linear baseline. Everything later must beat these.
5. Train RandomForest and XGBoost/LightGBM. Use cross-validation, not a single split.
6. Tune the best model with Optuna or RandomizedSearchCV. Log every trial to W&B.
7. Evaluate on the held-out test set ONCE. Report the business metric + a confusion matrix / residual plot.
8. Explain the model with SHAP; sanity-check that top features make business sense.
9. Wrap inference in `src/predict.py` with a clean `predict(df)` API and save the fitted pipeline with joblib.

## Tech stack
Python, pandas, scikit-learn, XGBoost/LightGBM, Optuna, SHAP, Weights & Biases

## Where it lives (your tools)
| Tool | How you use it here |
|---|---|
| **Kaggle** | Source the dataset; publish a polished public notebook of your EDA + modeling. |
| **GitHub** | Repo with src/, notebooks/, tests/, and a results table in the README. |
| **Weights & Biases** | Log every CV run and tuning trial; embed a W&B comparison chart in the README. |
| **Docker** | Optional here - add a Dockerfile that runs training end-to-end for reproducibility practice. |

## Portfolio artifact
GitHub repo (pinned) + public Kaggle notebook + a W&B report link. Resume line: 'Built a reproducible tabular ML pipeline (AUC X.XX) with leakage-safe preprocessing and experiment tracking.'

## Definition of Done
- [ ] Best model beats both the dumb and linear baselines on a held-out test set.
- [ ] Zero data leakage: all preprocessing lives inside the sklearn Pipeline / CV folds.
- [ ] Every experiment is logged to W&B and the README has a results table.
- [ ] `python -m src.train` reproduces the model from scratch; `predict()` loads and scores new rows.
- [ ] README explains the problem, approach, results, and 2 limitations in plain English.

## Stretch goals
- Enter the matching Kaggle competition and record your leaderboard rank.
- Add a fairness/slice analysis (does the model underperform on a subgroup?).
- Serve predictions via a tiny FastAPI endpoint.

## Resources
- [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)
- [Kaggle: Intro to ML & Intermediate ML](https://www.kaggle.com/learn)
- [W&B quickstart](https://docs.wandb.ai/quickstart)

## Results (fill this in as you build)
- **Headline metric:** _e.g. AUC / F1 / accuracy / p95 latency_
- **Artifact links:** GitHub _ | HF _ | Kaggle _ | W&B _ | Demo _
- **One thing that broke and how I fixed it:** _..._
- **Build-in-public post:** _link_
