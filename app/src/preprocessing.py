import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from config import NUMERIC_FEATURES


def filter_age(df: pd.DataFrame, max_age: int = 19) -> pd.DataFrame:
    df = df.copy()
    if "Idade" not in df.columns:
        return df
    return df[df["Idade"] <= max_age]


def replace_infs(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    feats = [f for f in NUMERIC_FEATURES if f in df.columns]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[("num", numeric_pipe, feats)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
