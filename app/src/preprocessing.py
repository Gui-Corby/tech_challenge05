import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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
    if not feats:
        raise ValueError("make_preprocessor: nenhuma NUMERIC_FEATURE encontrada no dataframe")

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[("num", numeric_pipe, feats)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def check_all_nan_columns(df: pd.DataFrame,
                          numeric_features: list[str]) -> list[str]:
    """
    Retorna lista de colunas que estão 100% NaN
    """

    all_nan_cols = []

    for col in numeric_features:
        if col in df.columns:
            if df[col].isna().all():
                all_nan_cols.append(col)

    return sorted(all_nan_cols)
