"""
features.py - turn raw columns into informative features (the heart of data-centric AI).

WHY  : the biggest accuracy gains often come from BETTER FEATURES, not fancier models. The
       churn signal here is an INTERACTION between tenure and price. A linear model sees those
       two columns separately and is near-blind to it. Hand it one column that encodes the
       interaction and AUC leaps - that single feature drives the lift you see in train.py.
RULE : these functions are PURE and ROW-LOCAL (each row's features depend only on that row).
       We avoid population statistics (e.g. a global median) here, because at serve time you
       score one row at a time and a global stat would cause train/serve SKEW. Population
       statistics belong INSIDE the fitted pipeline (the scaler), learned on training data
       only. (This is the leakage lesson from Project 01, reused.)
WHERE: used identically by train.py and by the DVC features stage, so there is no skew.
"""
import pandas as pd


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 1) THE big one: encode the tenure x price interaction as a single column.
    #    "match" = short-tenure&high-price OR long-tenure&low-price. Row-local, fixed thresholds.
    out["tenure_price_match"] = ((out["tenure"] < 36) == (out["MonthlyCharges"] > 70)).astype(int)
    # 2) spend per month of tenure - a ratio linear models can't form from two columns
    out["charges_per_tenure"] = out["TotalCharges"] / (out["tenure"] + 1)
    # 3) brand-new customers churn most
    out["is_new"] = (out["tenure"] < 6).astype(int)
    # 4) the classic telecom churn flag
    if "Contract" in out.columns:
        out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)
    return out
    # Extend: add more row-local features and re-measure the lift in train.py.
