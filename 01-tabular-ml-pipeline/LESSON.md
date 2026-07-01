# Lesson 01 - Tabular Machine Learning, from scratch

> Read this with the code open beside you. By the end you'll understand *every* line in
> `src/` and *why* it's there.

## 0. What you'll be able to do after this
- Explain supervised learning, train/test splits, cross-validation, and data leakage.
- Read a tabular dataset, build a leakage-safe model, and judge it with the right metric.
- Run a complete ML pipeline (`python -m src.train --demo`) and read its output.

## 1. The big picture (why this project exists)
Most business ML is still **tabular**: rows = things (customers, transactions), columns =
facts about them, and you predict one column (will this customer churn?). You already think
in tables and pipelines from 10 years of ETL, so this is the perfect on-ramp: same data
instincts, new goal (prediction instead of transport).

The job here: given a customer's details, output the **probability they will churn** (leave).
That's **binary classification**.

## 2. Foundations from scratch (the basics)

**Supervised learning.** We learn from *labeled* examples - rows where we already know the
answer (did they churn: Yes/No). The model finds a function `f(features) -> answer`.
"Training" = finding `f`. "Inference" = using `f` on new rows.

**Features (X) and target (y).** Features are the input columns (tenure, charges, contract).
The target is the column we predict (`Churn`). We convert Yes/No to 1/0 so math works.

**Why we split the data.** If you test a model on the same rows it trained on, of course it
looks great - it memorized them. So we hold out a **test set** it never sees during training.
Its score there estimates real-world performance. The test set is the *final exam*: you look
at it **once**, at the very end. (In code: `train_test_split(..., stratify=y)`.)

- **Stratify** keeps the churn rate the same in train and test, so the exam is fair.

**Cross-validation (CV).** A single train/validation split is noisy. CV rotates the
validation slice over k=5 "folds" and averages the score - a far steadier estimate. We use
CV on the *training* data to choose between models, keeping the test set untouched.
(In code: `cross_val_score(pipe, X_tr, y_tr, cv=5)`.)

**Data leakage - the #1 killer.** Leakage is when information you wouldn't have at prediction
time sneaks into training. Classic example: scaling/encoding using statistics computed over
the *whole* dataset (including validation rows). Your score looks amazing, then collapses in
production. **The fix:** put preprocessing *inside* the pipeline so it's re-fit on each
training fold only. That's the entire reason `pipeline.py` exists. (Your ETL eye for data
lineage is exactly the instinct that prevents this.)

**Baselines.** Before celebrating, compare against the dumbest models:
- *Majority-class* dummy (always predict "won't churn") - sets the floor (AUC ~0.50).
- *Logistic regression* - a strong, simple linear bar.
A fancy model only earns its place by beating these.

**Gradient boosting (XGBoost / HistGBM).** An ensemble of small decision trees added one at
a time, each correcting the previous trees' mistakes. It dominates tabular problems. (Our
code uses XGBoost if installed, else scikit-learn's HistGradientBoosting - so it always runs.)

**Metrics - pick the one that matches the problem.**
- *Accuracy* (fraction correct) is **misleading** when classes are imbalanced: if only 27%
  churn, predicting "nobody churns" is 73% accurate and useless.
- *Precision* = of customers we flagged, how many really churned. *Recall* = of churners,
  how many we caught. *F1* balances them.
- *ROC-AUC* = the probability the model ranks a random churner above a random non-churner.
  0.5 = coin flip, 1.0 = perfect. It's threshold-independent, so we optimize for it here.
- *Confusion matrix* = the 2x2 count of right/wrong by class.

## 3. How the pieces fit - the flow

```
                   +-------------+
   raw CSV  or ->  |  data.py    |  load + clean + split into X / y + column types
   --demo data     +------+------+
                          |  X, y, num_cols, cat_cols
                          v
                   +-------------+
                   | train.py    |  STEP 2 split -> STEP 4 cross-validate candidates
                   |             |  uses pipeline.py to wrap EACH model leakage-safely
                   +------+------+      ^                         ^
                          |             | build_pipeline()        | baseline_models()
                          |        +----+------+            +-----+------+
                          |        | pipeline. |            | baseline.  |
                          |        |   py      |            |   py       |
                          |        +-----------+            +------------+
                          |  STEP 6 evaluate best on test set (ONCE)
                          v
                 artifacts/model.joblib  +  metrics.json
                          |
                          v
                   +-------------+
                   | predict.py  |  load the saved pipeline, score new customers
                   +-------------+
```

## 4. Code reading order (open them in this sequence)
1. `src/data.py` - where data comes from and what shape it's in.
2. `src/pipeline.py` - the leakage-safe preprocessing+model bundle (short, crucial).
3. `src/baseline.py` - the bars to beat.
4. `src/train.py` - the orchestrator; read the `# === STEP n ===` banners top-to-bottom.
5. `src/predict.py` - using the saved model.
6. `tests/test_pipeline.py` - what "correct" looks like.

## 5. Walkthrough - the key lines
- `data.make_demo_data()` bakes a *known* signal into synthetic data (month-to-month + high
  charges + low tenure => churn). Because we built the signal, we know a good model must find
  it - handy for testing.
- `pipeline.build_pipeline()` returns ONE object. When `cross_val_score` re-fits it per fold,
  the imputer/scaler/encoder learn only from that fold's training rows => no leakage.
- `train.get_gbm()` shows a production-minded pattern: try the best tool (XGBoost), fall back
  gracefully (HistGBM) so the code never hard-crashes on a missing dependency.
- `train.py` STEP 6 calls `predict_proba(...)[:, 1]` - we want the *probability* of class 1
  (churn), not a hard 0/1, because AUC needs scores and businesses set their own threshold.

## 6. Common mistakes (and how to avoid them)
- **Fitting the scaler on all data** -> leakage. Always preprocess inside the pipeline.
- **Reading the test set more than once / tuning on it** -> you overfit the exam. Use CV.
- **Trusting accuracy on imbalanced data** -> use AUC / F1.
- **Committing the dataset or `artifacts/`** -> they're in `.gitignore`; never push data.
- **Dropping rows with missing values silently** -> impute instead, and log how many.

## 7. Check your understanding
1. Why is fitting `StandardScaler` on the whole dataset a form of leakage?
2. The dummy model scores AUC 0.50. Why exactly 0.50?
3. Accuracy is 0.77 but recall for churners is 0.33. Explain why that can still be a weak model.
4. Why do we use `predict_proba` instead of `predict` for AUC?
5. What would you change if churn were only 2% of customers?

## 8. Mini-glossary (full versions in /GLOSSARY.md)
feature, target, supervised learning, train/test split, stratify, cross-validation, leakage,
baseline, gradient boosting, ROC-AUC, precision/recall/F1, confusion matrix, pipeline.

## 9. Going deeper
- scikit-learn "Common pitfalls" (leakage): https://scikit-learn.org/stable/common_pitfalls.html
- Kaggle Learn - Intro to ML & Intermediate ML: https://www.kaggle.com/learn
- StatQuest (YouTube) for gradient boosting & ROC/AUC, explained visually.
