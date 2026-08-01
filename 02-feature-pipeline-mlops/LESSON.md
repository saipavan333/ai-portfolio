# Lesson 02 - Data-Centric AI & MLOps foundations

> This is a strong differentiator. Most ML beginners can fit a model; very few can wrap it in the
> engineering rigor you already practice in ETL. This lesson turns 10 years of pipeline
> instincts into an AI superpower.

## 0. What you'll be able to do after this
- Explain data-centric AI and prove a feature change helps (measured, not vibes).
- Write PURE, row-local feature functions that don't cause train/serve skew.
- Add a data-quality gate that fails fast on bad input.
- Understand DVC (versioned data) and CI (automated tests) and why they matter.

## 1. The big picture (why this project exists)
In Project 1 you improved the MODEL. In the real world the biggest, cheapest wins usually
come from improving the DATA: better features, cleaner labels, broader coverage. That mindset
is called **data-centric AI** (vs. model-centric). And to make data improvements *trustworthy*,
you wrap them in MLOps: version the data (DVC), validate it (quality gates), and test it (CI).
This is exactly an ETL pipeline - with a model bolted on the end.

## 2. Foundations from scratch (the basics)

**Data-centric AI.** Hold the model fixed and improve the data. `train.py` literally does
this: it freezes one logistic regression and only changes the features, then compares AUC.
Here engineered features lift AUC from ~0.74 to ~0.83 - a +0.09 gain with zero model changes.

**Why feature engineering works (the key intuition).** Our synthetic churn signal is an
*interaction*: customers churn when tenure and price "match" (short+expensive OR
long+cheap). A linear model sees `tenure` and `MonthlyCharges` as separate knobs and is
near-blind to their interaction. The moment we add ONE column that encodes the interaction
(`tenure_price_match`), the model can use it and AUC jumps. Lesson: features let simple
models express patterns they otherwise can't.

**Pure, row-local features (avoiding skew).** A feature function must give the same answer
for a row whether that row is scored alone (production) or inside a big batch (training). So
features must depend ONLY on that row - never on a population statistic like a global median.
If you need a population stat (mean, median), it belongs INSIDE the fitted pipeline (the
scaler), which learns it on training data only. This is the leakage lesson from Project 1,
applied to features. `tests/test_features.py` actually checks row-locality.

**Data-quality gate.** `data_quality.validate()` checks the columns exist, nulls are under a
threshold, and numbers are in range - then raises `DataQualityError` if not. This is a data-engineering
"reject the bad batch" instinct. In production you'd run this before training AND before
serving.

**DVC (Data Version Control).** Git is bad at large files. DVC versions your data and model
files, storing lightweight pointers in git so any result is reproducible from a commit.
`dvc.yaml` declares a PIPELINE of stages (features -> train) with declared inputs (deps) and
outputs (outs). `dvc repro` re-runs a stage ONLY when its inputs change - incremental, just
like ETL. (`features_stage.py` is the code behind the "features" stage.)

**CI (Continuous Integration).** `.github/workflows/ci.yml` runs on every push: install deps,
lint with `ruff`, run `pytest`. It catches breakage before it reaches `main`. "My repo has
green CI" signals real engineering maturity to employers.

## 3. How the pieces fit - the flow

```
  data/raw/*  (DVC-tracked)
       |
       v
  +-----------------+      validate() gate (data_quality.py) -- fail fast on bad data
  | features_stage  |----> data/features/*   (DVC 'features' stage; re-runs only if inputs change)
  | (features.py)   |
  +-----------------+
       |
       v
  +-----------------+      compares: SAME model on RAW vs ENGINEERED features
  |    train.py     |----> metrics.json  (the data-centric lift; DVC-tracked metric)
  | (modeling.py)   |
  +-----------------+
       ^
       |  every push:
  GitHub Actions CI: ruff lint + pytest  (.github/workflows/ci.yml)
```

## 4. Code reading order
1. `src/demo_data.py` - the (synthetic) raw data and the hidden interaction signal.
2. `src/data_quality.py` - the fail-fast validation gate.
3. `src/features.py` - PURE, row-local feature engineering (the heart of the project).
4. `src/modeling.py` - the leakage-safe pipeline + X/y split (same idea as Project 1).
5. `src/train.py` - freezes the model, compares raw vs engineered, writes `metrics.json`.
6. `src/features_stage.py` + `dvc.yaml` - how this becomes a reproducible DVC pipeline.
7. `.github/workflows/ci.yml` - the CI that runs your tests automatically.

## 5. Walkthrough - the key idea in one paragraph
`train.cv_auc()` is called twice with the SAME logistic model - once on raw data, once on
`make_features(raw)`. Everything else is identical, so any AUC difference is caused purely by
the features. That is a clean A/B experiment for a data change - the scientific habit that
makes "I improved the data" a claim you can defend with a number.

## 6. Common mistakes (and how to avoid them)
- **Population stats inside a feature function** (e.g. `df[col].median()`): causes train/serve
  skew. Keep features row-local; let the pipeline learn population stats.
- **Improving the model and the data at once**: you can't attribute the gain. Change one thing.
- **No data validation**: a malformed upstream batch silently poisons the model. Gate it.
- **Committing data to git**: use DVC; keep `.gitignore` honest.

## 7. Check your understanding
1. Why can a linear model not capture the tenure x price "match" signal from raw columns?
2. What's the difference between a row-local feature and a population-statistic feature, and
   why does it matter at serve time?
3. What does `dvc repro` avoid re-computing, and how does it know?
4. Your CI is green but the model got worse - what kind of test is missing?
5. How would you prove a NEW feature helped, fairly?

## 8. Mini-glossary (full versions in /GLOSSARY.md)
data-centric AI, feature engineering, train/serve skew, data-quality gate, DVC, pipeline
stage, CI/CD, reproducibility, A/B feature test.

## 9. Going deeper
- DVC get-started: https://dvc.org/doc/start
- Great Expectations (heavier data validation): https://docs.greatexpectations.io/
- Andrew Ng on data-centric AI: search "A Chat with Andrew Ng - Data-Centric AI".
