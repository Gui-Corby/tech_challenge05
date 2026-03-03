from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.feature_engineering import build_features_2024
from src.preprocessing import make_preprocessor


def test_full_pipeline_fit_predict_smoke(df_minimo):
    df = build_features_2024(df_minimo)

    X = df.drop(columns=["Defasagem"])
    y = df["Defasagem"]

    pipe = Pipeline(steps=[
        ("preprocess", make_preprocessor(X)),
        ("model", LogisticRegression(max_iter=200, class_weight="balanced")),
    ])

    pipe.fit(X, y)
    pred = pipe.predict(X)

    assert len(pred) == len(y)
