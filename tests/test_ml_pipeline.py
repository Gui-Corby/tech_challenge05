import pandas as pd
from src.ml_pipeline import feature_engineering_block

def test_feature_engineering_block_runs(df_minimo):
    out = feature_engineering_block(df_minimo.copy())
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(df_minimo)
