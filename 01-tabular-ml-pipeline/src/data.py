"""
data.py - load and sanity-check the dataset.

WHAT : turns a raw CSV (or synthetic demo data) into a clean (X, y) plus the lists of
       numeric and categorical columns the pipeline needs.
WHY  : keeping data loading separate from modeling is an ETL habit that pays off - you can
       test, swap, or reuse the loader without touching training logic.
HOW  : load -> light clean -> split into X/y -> detect column types -> validate.
WHERE: the FIRST step of the flow. train.py calls these functions before anything else.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Telco Customer Churn specifics (the recommended real dataset for this project)
TARGET = "Churn"
DROP_COLS = ["customerID"]          # why: an ID carries no signal and can leak identity


def load_raw(path: str) -> pd.DataFrame:
    # why: read the file exactly as-is first; cleaning is a separate, testable step
    return pd.read_csv(path)


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    # why: in this dataset TotalCharges has blank strings for brand-new customers.
    #      Coerce to numeric so blanks become NaN; the pipeline will impute them later.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def make_demo_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Synthetic churn-like data so this project runs with ZERO downloads.
    For the real project, use load_raw() + clean_telco() on the Kaggle CSV instead."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)                                  # months as a customer
    monthly = rng.normal(70, 30, n).clip(15, 130)                   # monthly bill
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[.45, .35, .20])
    # why: bake in a realistic signal so models have something true to learn -
    #      month-to-month + high charges + low tenure + fiber => more likely to churn
    logit = (-2.0
             + 1.4 * (contract == "Month-to-month")
             + 0.015 * monthly
             - 0.04 * tenure
             + 0.6 * (internet == "Fiber optic")
             + rng.normal(0, 0.5, n))                               # noise
    churn = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)
    return pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": monthly.round(2),
        "TotalCharges": (tenure * monthly).round(2),
        "Contract": contract,
        "InternetService": internet,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })


def split_xy(df: pd.DataFrame, target: str = TARGET):
    """Return X, y, numeric-column names, categorical-column names."""
    # why: convert a Yes/No target to 1/0 so metrics like AUC work
    y = (df[target] == "Yes").astype(int) if df[target].dtype == object else df[target]
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return X, y, num_cols, cat_cols


def basic_checks(X: pd.DataFrame, y: pd.Series) -> None:
    """Cheap guards that catch the most common data bugs early (your ETL instinct)."""
    assert len(X) == len(y), "X and y must have the same number of rows"
    assert y.nunique() == 2, "this project expects a BINARY target"
    rate = float(y.mean())
    print(f"[data] rows={len(X)}  features={X.shape[1]}  positive_rate={rate:.1%}")
    if rate < 0.05 or rate > 0.95:
        print("[data] WARNING: very imbalanced - prefer AUC/F1 over accuracy")
