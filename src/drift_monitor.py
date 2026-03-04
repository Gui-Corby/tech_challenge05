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


def calculate_psi(expected, actual, bins=10, eps=1e-6):
    expected = pd.to_numeric(expected, errors="coerce").to_numpy()
    actual = pd.to_numeric(actual, errors="coerce").to_numpy()

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    # se não tiver dados suficientes, retorna 0 (ou np.nan)
    if expected.size == 0 or actual.size == 0:
        return 0.0

    _, bin_edges = np.histogram(expected, bins=bins)

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_perc = expected_counts / max(expected_counts.sum(), 1)
    actual_perc = actual_counts / max(actual_counts.sum(), 1)

    expected_perc = np.clip(expected_perc, eps, None)
    actual_perc = np.clip(actual_perc, eps, None)

    psi = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
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

    common_cols = sorted(set(baseline_df.columns) & set(prod_df.columns))

    numeric_cols = []
    for col in common_cols:
        # tenta converter para número (se virar tudo NaN, é categórica)
        b = pd.to_numeric(baseline_df[col], errors="coerce")
        p = pd.to_numeric(prod_df[col], errors="coerce")
        if b.notna().any() and p.notna().any():
            numeric_cols.append(col)

    results = {}
    for col in numeric_cols:
        b = pd.to_numeric(baseline_df[col], errors="coerce").dropna()
        p = pd.to_numeric(prod_df[col], errors="coerce").dropna()

        psi = calculate_psi(b, p)
        results[col] = float(psi)

    return {"n_features": len(results), "psi_by_feature": results}
