import pandas as pd
import numpy as np
from config import NUMERIC_FEATURES, TARGET_COL


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feats = [f for f in NUMERIC_FEATURES if f in df.columns]
    missing = set(NUMERIC_FEATURES) - set(df.columns)
    if missing:
        print("Features ausentes:", missing)
    X = df[feats].copy()
    y = df[TARGET_COL].copy()
    return X, y


def filter_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df[(df["Idade"] <= 19)]


def treat_missing_values(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in numeric_features:
        if col not in df.columns:
            continue

        if df[col].isna().any():
            median_val = df[col].median()
            if pd.isna(median_val):  # Se mediana for NaN, usar 0
                median_val = 0
            df[col] = df[col].fillna(median_val)

        df[col] = df[col].replace([np.inf, -np.inf], 0)

    return df
