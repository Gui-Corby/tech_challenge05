import numpy as np
import pandas as pd

from preprocessing import filter_age, replace_infs, make_preprocessor


def test_filter_age_caps_max_age():
    df = pd.DataFrame({"Idade": [10, 19, 20], "x": [1, 2, 3]})
    out = filter_age(df, max_age=19)
    assert out["Idade"].max() <= 19
    assert len(out) == 2


def test_replace_infs_turns_into_nan():
    df = pd.DataFrame({"a": [1.0, np.inf, -np.inf], "b": [1, 2, 3]})
    out = replace_infs(df, ["a"])
    assert out["a"].isna().sum() == 2


def test_make_preprocessor_fit_transform(df_minimo):
    # Cria features primeiro porque o preprocessor seleciona NUMERIC_FEATURES
    from feature_engineering import build_features_2024
    df = build_features_2024(df_minimo)

    X = df.drop(columns=["Defasagem"])
    pre = make_preprocessor(X)
    Xt = pre.fit_transform(X)

    # Deve retornar alguma coisa com n_linhas = len(X)
    assert Xt.shape[0] == len(X)
