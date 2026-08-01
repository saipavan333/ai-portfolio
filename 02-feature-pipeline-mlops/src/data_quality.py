"""
data_quality.py - fail fast on bad data (a data-engineering discipline, applied to ML).

WHY  : models produce confident garbage on garbage input. A data-quality GATE turns silent
       corruption into a loud, early failure - exactly like a good ETL job that refuses to
       load a malformed batch. This is one of your biggest credibility signals for AI teams.
WHAT : validate() checks required columns exist, null rates are acceptable, and numeric
       columns fall in sane ranges. It raises DataQualityError on any violation.
WHERE: called at the very start of train.py (and would gate the DVC pipeline in production).
"""
import pandas as pd


class DataQualityError(AssertionError):
    """Raised when the incoming data fails a quality check."""


def validate(df: pd.DataFrame, required_cols, max_null_frac: float = 0.2, ranges=None) -> bool:
    # 1) schema: are the columns we depend on present?
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise DataQualityError(f"missing required columns: {missing}")

    # 2) completeness: is any required column too empty to trust?
    null_frac = df[required_cols].isna().mean()
    too_empty = null_frac[null_frac > max_null_frac]
    if len(too_empty):
        raise DataQualityError(f"columns over {max_null_frac:.0%} null: {too_empty.to_dict()}")

    # 3) ranges: are numeric values physically plausible?
    for col, (lo, hi) in (ranges or {}).items():
        if col in df.columns:
            out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
            if out_of_range:
                raise DataQualityError(f"{col}: {out_of_range} values outside [{lo}, {hi}]")

    print(f"[dq] OK - {len(df)} rows passed {len(required_cols)} column checks")
    return True
