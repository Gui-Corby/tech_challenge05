from __future__ import annotations

import json

import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import MODEL_PATH, TEST_PATH, TARGET_COL, ARTIFACTS_DIR


EVAL_METRICS_PATH = ARTIFACTS_DIR / "metrics_eval.json"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}. Rode train.py primeiro.")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test set não encontrado: {TEST_PATH}. Rode train.py primeiro.")

    pipeline = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if TARGET_COL not in test_df.columns:
        raise KeyError(f"Coluna target '{TARGET_COL}' não está em {TEST_PATH}")

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

    print("\n=== Classification report (conjunto teste) ===")
    print(classification_report(y_test, pred))

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    with open(EVAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nMétricas de avaliação salvas em: {EVAL_METRICS_PATH}")


if __name__ == "__main__":
    main()
