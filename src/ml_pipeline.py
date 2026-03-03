import pandas as pd

from src.config import NUMERIC_FEATURES
from src.feature_engineering import build_features_2024
from src.preprocessing import replace_infs, check_all_nan_columns

def feature_engineering_block(df: pd.DataFrame) -> pd.DataFrame:
    df = build_features_2024(df).copy()
    df = replace_infs(df, NUMERIC_FEATURES)

    nan_cols = check_all_nan_columns(df, NUMERIC_FEATURES)
    for col in nan_cols:
        df[col] = 0

    return df
