"""Tests: features are row-local, keep row count, and add the expected columns."""
import pandas as pd
from src.demo_data import make_raw_demo
from src.features import make_features


def test_features_preserve_rows_and_add_columns():
    raw = make_raw_demo(200)
    eng = make_features(raw)
    assert len(eng) == len(raw)                       # no rows lost
    assert "charges_per_tenure" in eng.columns        # the key engineered feature exists
    assert eng["charges_per_tenure"].notna().all()    # no NaNs introduced


def test_features_are_row_local():
    # row-local => scoring a single row gives the same feature as scoring it within a batch
    raw = make_raw_demo(50)
    one = make_features(raw.iloc[[0]]).iloc[0]["charges_per_tenure"]
    many = make_features(raw).iloc[0]["charges_per_tenure"]
    assert abs(one - many) < 1e-9
