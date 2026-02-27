from __future__ import annotations

import json

import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    cohen_kappa_score
)

from config import (
    DF_2024,
    NUMERIC_FEATURES,
    TARGET_COL,
    ARTIFACTS_DIR,
    MODEL_PATH,
    METRICS_PATH,
    TEST_PATH,
    DATA_PATH
)

print("DATA_PATH =", DATA_PATH)
print("DF_2024 shape =", DF_2024.shape)
print("DF_2024 columns sample =", list(DF_2024.columns)[:15])
print("dtypes INDE:", DF_2024[["INDE 2024", "INDE 23", "INDE 22"]].dtypes)
print("head INDE:\n", DF_2024[["INDE 2024", "INDE 23", "INDE 22"]].head())

from feature_engineering import build_features_2024
from preprocessing import filter_age, replace_infs, make_preprocessor, check_all_nan_columns


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # 1) Feature engineering + limpezas fora do sklearn
    df = build_features_2024(DF_2024).copy()
    df = filter_age(df, max_age=19)
    df = replace_infs(df, NUMERIC_FEATURES)

    nan_cols = check_all_nan_columns(df, NUMERIC_FEATURES)
    if nan_cols:
        print("Colunas 100% NaN -> preenchendo com 0:", nan_cols)
        for col in nan_cols:
            df[col] = 0

    # 2) X/y
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy().dropna()
    X = X.loc[y.index]

    # 3) Split (tenta estratificar; se falhar, cai pra None)
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

    # 4) Pipeline completo (preprocess + modelo)
    pipeline = Pipeline(steps=[
        ("preprocess", make_preprocessor(X_train)),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )),
    ])

    # 5) Treino
    pipeline.fit(X_train, y_train)

    # 6) Validação holdout
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

    # 7) Salvar artifacts
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
