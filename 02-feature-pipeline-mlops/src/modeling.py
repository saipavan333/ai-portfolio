"""
modeling.py - the same leakage-safe pipeline idea as Project 01, kept local so this repo
stands alone. split_xy() separates inputs from target and detects column types.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_xy(df, target="Churn"):
    y = (df[target] == "Yes").astype(int) if not pd.api.types.is_numeric_dtype(df[target]) else df[target]
    X = df.drop(columns=[target])
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return X, y, num, cat


def build_pipeline(num, cat, model):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return Pipeline([("pre", pre), ("model", model)])

