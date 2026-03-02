import time
import pandas as pd

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

class student(BaseModel):
    IAN: float
    IDA: float
    IEG: float
    IPS: float
    IPP: float
    IAA: float
    INDE_2024: float
    Idade: int
    Idade_Ingresso: int
    Fase_num: int
    Mat: float
    Por: float
    Tem_Ingles: int
    Ingles: float


class PredictRequest(BaseModel):
    history: list[student]

@app.post("/predict")
def predict_evasion(req: PredictRequest):
    df_raw = pd.DataFrame([req.dict()])