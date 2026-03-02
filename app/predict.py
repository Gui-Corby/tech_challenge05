import time
import pandas as pd

from typing import List, Any, Dict
import joblib

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import MODEL_PATH

router = APIRouter(prefix="/api", tags=["predict"])

try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    pipeline = None
    load_error = e

class Student(BaseModel):
    IAN: float
    IDA: float
    IEG: float
    IPS: float
    IPP: float
    IAA: float

    INDE_2024: float
    INDE_23: float
    INDE_22: float

    Mat: float
    Por: float
    Ing: float | None = None

    Pedra_2024: str
    Pedra_23: str
    Pedra_22: str
    Pedra_21: str

    Idade: int
    Ano_ingresso: int
    N_Av: int | None = None

    Fase: str
    Fase_Ideal: str
    Genero: str

class PredictRequest(BaseModel):
    history: list[Student]


RENAME_MAP = {
    "INDE_2024": "INDE 2024",
    "INDE_23": "INDE 23",
    "INDE_22": "INDE 22",
    "Pedra_2024": "Pedra 2024",
    "Pedra_23": "Pedra 23",
    "Pedra_22": "Pedra 22",
    "Pedra_21": "Pedra 21",
    "Fase_Ideal": "Fase Ideal",
    "Genero": "Gênero",
    "Ano_ingresso": "Ano ingresso",
}

@router.post("/predict")
def predict(req: PredictRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail=f"Modelo não carregado: {repr(load_error)}")

    try:
        rows: List[Dict[str, Any]] = [s.model_dump() for s in req.history]
        df_raw = pd.DataFrame(rows).rename(columns=RENAME_MAP)

        preds = pipeline.predict(df_raw)

        return {"n_predictions": int(len(preds)), "predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao prever: {repr(e)}")
