import numpy as np

from fastapi import APIRouter
from starlette.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.drift_monitor import calculate_psi

router = APIRouter()

@router.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@router.get("/drift")
def drift():
    try:
        rng_base = np.random.default_rng(42)
        baseline = rng_base.normal(0, 1, 1000)

        rng_cur = np.random.default_rng(123)
        current = rng_cur.normal(0, 1, 1000)

        psi = float(calculate_psi(baseline, current, bins=10))

        if psi < 0.1:
            status = "no_drift"
        elif psi < 0.25:
            status = "moderate_drift"
        else:
            status = "significant_drift"

        return {"psi": float(psi), "status": status}

    except Exception as e:
        return {"psi": None, "status": "error", "detail": repr(e)}
