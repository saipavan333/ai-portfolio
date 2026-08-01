"""
demo_data.py - synthetic RAW data so this project runs with zero downloads.

WHY: this project is about improving the DATA. We hide the churn signal inside a BALANCED
     INTERACTION that a linear model provably cannot represent from the raw columns:
     a customer churns when tenure and price "match" - either short-tenure & high-price OR
     long-tenure & low-price. Each raw column on its own says nothing (both halves are 50/50),
     so a linear model on raw features is near-blind. features.py exposes the interaction as a
     single column and accuracy jumps. That contrast is the entire point of data-centric AI.
"""
import numpy as np
import pandas as pd


def make_raw_demo(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 28, n).clip(15, 130)
    total = (tenure * monthly * rng.uniform(0.85, 1.15, n)).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20])
    a = tenure < 36                      # "short tenure"  (about half the customers)
    b = monthly > 70                     # "high price"    (about half the customers)
    match = (a == b).astype(int)         # the hidden XOR-style interaction
    logit = (-1.4
             + 0.8 * (contract == "Month-to-month")   # a weak signal raw features CAN use
             + 2.8 * match                              # the strong signal raw features CANNOT
             + rng.normal(0, 0.40, n))
    churn = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)
    return pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": monthly.round(2),
        "TotalCharges": total,
        "Contract": contract,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
