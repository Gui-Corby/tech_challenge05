import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("artifacts/inference_log.jsonl")

def log_inference(features: dict, prediction: int,
                  proba: float):
    LOG_PATH.parent.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": int(prediction),
        "proba": float(proba),
        "features": features,
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
