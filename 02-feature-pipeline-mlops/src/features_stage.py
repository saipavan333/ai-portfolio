"""
features_stage.py - the DVC 'features' stage entry point (see dvc.yaml).

In a DVC pipeline, each stage declares its inputs (deps) and outputs (outs). DVC re-runs a
stage ONLY when its inputs change - just like an incremental ETL job. This stage reads raw
data and writes engineered features.
Run directly: python -m src.features_stage     (expects data/raw/data.csv)
"""
import os
import pandas as pd
from src.features import make_features


def main():
    os.makedirs("data/features", exist_ok=True)
    df = pd.read_csv("data/raw/data.csv")
    make_features(df).to_csv("data/features/data.csv", index=False)
    print("[features] wrote data/features/data.csv")


if __name__ == "__main__":
    main()
