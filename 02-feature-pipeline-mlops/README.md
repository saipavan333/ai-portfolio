# Project 02 - Reproducible, Feature-Engineered ML Pipeline (Data-Centric AI)

> **New here? Start with [LESSON.md](LESSON.md)** - it teaches the concepts from scratch, then read the heavily-commented code in `src/`.
> Turn project 1 into a versioned, automated pipeline. This is the project that says 'I am the data person every AI team is desperate for.'
**Phase 1: Foundations & Data-Centric ML**  |  **Difficulty:** Beginner-Intermediate  |  **Est. time:** Week 2-3 (~35 hrs)

---

## Why (the point of this project)
The 2026 market rewards 'production deployment evidence' and data-centric skills above raw modeling. This project is a strong differentiator: you take a model and wrap it in the engineering rigor you already practice in ETL - versioned data, a feature pipeline, automated retraining, and CI. Most ML beginners cannot do this; you can.

## What you will build
A pipeline where raw data + code are versioned with DVC, features are computed by a reusable module (a mini 'feature store'), training is a single reproducible command, and GitHub Actions runs tests + a sample training job on every push. Improve the model primarily by improving the DATA (cleaning, feature engineering), not the algorithm.

## Key concepts (learn these as you go)
| Concept | What it means |
|---|---|
| **Data-centric AI** | Systematically improving the dataset (labels, features, coverage) instead of only the model. Often the highest-ROI lever. |
| **Data/version control (DVC)** | Git-like versioning for large data + model files so any result is reproducible from a commit hash. |
| **Feature engineering & feature store** | Deriving informative inputs (ratios, aggregates, time windows) and storing them so train and serve use identical logic - kills training/serving skew. |
| **Pipeline orchestration** | Expressing steps (ingest -> features -> train -> evaluate) as a DAG so they re-run only when inputs change. Directly parallels ETL DAGs. |
| **CI for ML** | Automated lint + unit tests + a smoke training run on every push, via GitHub Actions. |

## How - step by step
1. Fork your project-1 repo. Initialize DVC and put the raw dataset under DVC tracking; commit the .dvc pointer to git.
2. Refactor feature logic into `src/features.py` with pure functions: `make_features(df) -> df`. Same function used in train and serve.
3. Express the flow as a DVC pipeline (dvc.yaml) with stages: prepare -> features -> train -> evaluate, each with deps/outs/metrics.
4. Add at least 5 engineered features motivated by domain reasoning; measure their lift with W&B.
5. Run a data-quality gate (Great Expectations or simple asserts): schema, ranges, null thresholds. Fail the pipeline if data is bad - pure ETL discipline.
6. Write unit tests for feature functions and a tiny smoke test for training.
7. Add a GitHub Actions workflow: install, lint (ruff), run tests, run `dvc repro` on a small sample.
8. Document the lineage in the README with a simple diagram (raw -> features -> model -> metrics).

## Tech stack
Python, DVC, pandas, scikit-learn, Great Expectations, GitHub Actions, ruff, W&B

## Where it lives (your tools)
| Tool | How you use it here |
|---|---|
| **GitHub** | Hosts code + DVC pointers; Actions runs CI. This repo is your 'MLOps maturity' showpiece. |
| **Weights & Biases** | Track each feature-set version as a run; show the lift from data-centric changes. |
| **Kaggle** | Reuse the project-1 dataset; optionally pull a larger version to stress the pipeline. |
| **Docker** | Containerize the pipeline so `docker run` reproduces training anywhere. |

## Portfolio artifact
GitHub repo with a green CI badge + a W&B report comparing feature-set versions. Resume line: 'Built a DVC-versioned, CI-tested ML pipeline with a reusable feature layer and automated data-quality gates.'

## Definition of Done
- [ ] `dvc repro` reproduces the full pipeline from raw data to metrics.
- [ ] Feature engineering improves the project-1 metric measurably (logged in W&B).
- [ ] GitHub Actions is green: lint + tests + sample training run pass on push.
- [ ] A data-quality check fails the run when given deliberately bad data.
- [ ] README shows the lineage diagram and the before/after metric table.

## Stretch goals
- Add a scheduled GitHub Action that 'retrains' weekly on refreshed data.
- Swap the mini feature store for Feast and document the trade-offs.
- Add model + data drift checks (Evidently) and alert on threshold breach.

## Resources
- [DVC getting started](https://dvc.org/doc/start)
- [Great Expectations docs](https://docs.greatexpectations.io/)
- [Made With ML (MLOps)](https://madewithml.com/)

## Results

A controlled demonstration of **data-centric AI**: the model is held fixed (one logistic
regression) and only the features change, so any score difference comes purely from the data. The
churn signal is hidden in a tenure×price **interaction** a linear model can't read from raw columns.

| Feature set | 5-fold CV ROC-AUC |
|---|---|
| Raw features | 0.744 |
| + engineered features | **0.833** |
| **Data-centric lift** | **+0.090** |

One engineered column — `tenure_price_match`, which encodes that interaction — drives the whole
+0.09 gain, with **zero** change to the model. The data-quality gate passed (4,000 rows, 5 checks)
before training, failing fast on bad input the way an ETL job would.

**Engineering practices demonstrated**
- **Pure, row-local features** (fixed thresholds, no dataset-wide statistics) → no train/serve skew.
- **Data-quality gate** that rejects malformed data before it reaches the model.
- **Reproducible pipeline** wired for DVC stages + a GitHub Actions CI workflow (`ruff` + `pytest`).

**Artifacts:** code in `src/` · (add: `dvc repro` run, CI badge, Kaggle notebook)

**Next:** run the pipeline as DVC stages (`dvc repro`), enable the CI workflow, and apply the same
feature layer to the real Telco data to confirm the lift holds.