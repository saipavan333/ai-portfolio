"""
pipeline.py - the leakage-safe preprocessing + model bundle.

WHAT : build_pipeline() returns ONE scikit-learn object = preprocessing + estimator.
WHY  : this is the single most important habit in tabular ML. During cross-validation the
       preprocessing (imputing, scaling, encoding) is re-fit on each TRAINING fold only.
       If instead you scaled the whole dataset up front, statistics from the validation
       rows would leak into training and your score would LIE. Bundling prevents that.
HOW  : numeric branch (impute -> scale) + categorical branch (impute -> one-hot), joined by
       a ColumnTransformer, then the model - all wrapped in a Pipeline.
WHERE: built inside train.py, then handed to cross_val_score() and .fit().
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_pipeline(num_cols, cat_cols, model) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # why: median is robust to outliers
        ("scale", StandardScaler()),                     # why: helps linear models; harmless to trees
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # why: handle_unknown='ignore' => categories unseen in training won't crash at serve time
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])
    return Pipeline([("pre", pre), ("model", model)])
