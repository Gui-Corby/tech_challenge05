import json
import pandas as pd
import numpy as np
from pathlib import Path

LOG_PATH = Path("artifacts/inference_log.jsonl")
BASELINE_PATH = Path("artifacts/test.csv")

NUMERIC_FEATURES = [
    "IAN", "IDA", "IEG", "IPS", "IPP", "IAA",
    "INDE_2024", "INDE_23", "INDE_22",
    "Mat", "Por", "Ing", "Idade"]


def calculate_psi(expected, actual, bins=10):
    expected_perc, _ = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)

    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)

    psi = np.sum((actual_perc - expected_perc) *
                 np.log((actual_perc + 1e-6) / (expected_perc + 1e-6)))

    return float(psi)


def compute_drift():
    if not LOG_PATH.exists():
        return {"status": "no inference data yet"}

    baseline_df = pd.read_csv(BASELINE_PATH)

    logs = []
    with open(LOG_PATH, "r") as f:
        for line in f:
            logs.append(json.loads(line)["features"])

    prod_df = pd.DataFrame(logs)

    results = {}

    for col in NUMERIC_FEATURES:
        if col in baseline_df.columns and col in prod_df.columns:
            psi = calculate_psi(
                baseline_df[col].dropna(),
                prod_df[col].dropna()
            )
            results[col] = psi

    return {
        "psi_per_feature": results,
        "drift_flag": any(v > 0.2 for v in results.values())
    }
