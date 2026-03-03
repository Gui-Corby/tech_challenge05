from __future__ import annotations

import json
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, mean_absolute_error, cohen_kappa_score
)

from src.config import (
    DF_2024, TARGET_COL,
    ARTIFACTS_DIR, MODEL_PATH, METRICS_PATH, TEST_PATH
)

from src.preprocessing import filter_age, make_preprocessor
from src.ml_pipeline import feature_engineering_block


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # 1) X/y bruto (sem feature engineering aqui)

    df_raw = DF_2024.copy()
    df_raw = filter_age(df_raw, max_age=19)

    y = df_raw[TARGET_COL].copy().dropna()
    X = df_raw.drop(columns=[TARGET_COL]).copy()
    X = X.loc[y.index]

    # 2) Split

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        used_stratify = True
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        used_stratify = False

    print(f"Split: train={len(X_train)} test={len(X_test)} | stratify={'y' if used_stratify else 'None'}")

    # 3) Feature engineering transformer

    fe = FunctionTransformer(feature_engineering_block, validate=False)

    # "Enxerga" as colunas finais para montar o preprocessor exatamente como antes
    X_train_fe = fe.fit_transform(X_train)

    preprocess = make_preprocessor(X_train_fe)  # usa NUMERIC_FEATURES presentes

    # 4) Pipeline end-to-end

    pipeline = Pipeline(steps=[
        ("feature_engineering", fe),
        ("preprocess", preprocess),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )),
    ])

    # 5) Treino + validação

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "used_stratify": bool(used_stratify),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted")),
        "mae": float(mean_absolute_error(y_test, pred)),
        "kappa": float(cohen_kappa_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

    print("\n=== Classification report (holdout) ===")
    print(classification_report(y_test, pred))
    print("\n=== Metrics (principal: f1_macro) ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # 6) Salvar artifacts

    joblib.dump(pipeline, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test
    test_df.to_csv(TEST_PATH, index=False)

    print(f"\nModelo salvo em: {MODEL_PATH}")
    print(f"Métricas salvas em: {METRICS_PATH}")
    print(f"Test set salvo em: {TEST_PATH}")


if __name__ == "__main__":
    main()
