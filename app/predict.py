from __future__ import annotations

import pandas as pd
from typing import List, Any, Dict
import joblib

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from pydantic import BaseModel

from src.config import MODEL_PATH

from app.inference_logger import log_inference

from src.drift_monitor import compute_drift

router = APIRouter(tags=["predict"])

try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    pipeline = None
    load_error = e


class Student(BaseModel):
    IAN: float
    IDA: float
    IEG: float
    IPV: float
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
    history: List[Student]


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

        # Classe prevista
        preds = pipeline.predict(df_raw)

        # Probabilidades (n_amostras x n_classes)
        probas = pipeline.predict_proba(df_raw)

        # Salvando inferências
        for i, row in df_raw.iterrows():
            log_inference(
                features=row.to_dict(),
                prediction=preds[i],
                proba=max(probas[i])
            )

        # classes_ (ordem das colunas em probas)
        classes = pipeline.classes_.tolist()

        results = []
        for i, pred in enumerate(preds):
            # índice da classe prevista dentro de classes
            pred_idx = classes.index(pred)

            results.append({
                "index": i,
                "defasagem_prevista": int(pred),
                "prob_defasagem_prevista": float(probas[i][pred_idx]),
                "probabilidades_por_defasagem": {
                    str(cls): float(probas[i][j]) for j, cls in enumerate(classes)
                }
            })

        return {
            "total_alunos": len(results),
            "resultados": results,
            "classes_modelo": [int(c) for c in classes],
        }

    except AttributeError as e:
        # Caso o modelo não tenha predict_proba (não deveria com LogisticRegression)
        raise HTTPException(status_code=500, detail=f"Pipeline não suporta predict_proba: {repr(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao prever: {repr(e)}")


@router.get("/drift")
def drift():
    return compute_drift()


@router.get("/drift-report", response_class=HTMLResponse)
def drift_report():
    data = compute_drift()
    psi_by_feature = data.get("psi_by_feature", {})

    rows = []
    significant = 0
    moderate = 0
    no_drift = 0

    for feature, psi in psi_by_feature.items():
        if psi < 0.10:
            status = "No drift"
            color = "#16a34a"
            bg = "#dcfce7"
            no_drift += 1
        elif psi < 0.25:
            status = "Moderate drift"
            color = "#ca8a04"
            bg = "#fef9c3"
            moderate += 1
        else:
            status = "Significant drift"
            color = "#dc2626"
            bg = "#fee2e2"
            significant += 1

        # limita a barra visual para não explodir layout
        bar_width = min(psi * 20, 100)

        rows.append(f"""
        <tr>
            <td>{feature}</td>
            <td>{psi:.4f}</td>
            <td>
                <div class="bar-wrapper">
                    <div class="bar" style="width:{bar_width}%; background:{color};"></div>
                </div>
            </td>
            <td>
                <span class="badge" style="color:{color}; background:{bg};">
                    {status}
                </span>
            </td>
        </tr>
        """)

    rows_html = "".join(rows) if rows else """
    <tr>
        <td colspan="4" style="text-align:center; color:#64748b;">
            No drift data available.
        </td>
    </tr>
    """

    n_features = data.get("n_features", len(psi_by_feature))

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Drift Report</title>
        <style>
            * {{
                box-sizing: border-box;
            }}

            body {{
                margin: 0;
                padding: 32px;
                font-family: Arial, sans-serif;
                background: #f8fafc;
                color: #0f172a;
            }}

            .container {{
                max-width: 1100px;
                margin: 0 auto;
            }}

            .header {{
                margin-bottom: 24px;
            }}

            .header h1 {{
                margin: 0 0 8px 0;
                font-size: 32px;
            }}

            .header p {{
                margin: 0;
                color: #475569;
            }}

            .cards {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }}

            .card {{
                background: white;
                border-radius: 14px;
                padding: 20px;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
            }}

            .card-title {{
                font-size: 14px;
                color: #64748b;
                margin-bottom: 8px;
            }}

            .card-value {{
                font-size: 28px;
                font-weight: 700;
            }}

            .panel {{
                background: white;
                border-radius: 14px;
                padding: 24px;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
            }}

            .panel h2 {{
                margin-top: 0;
                margin-bottom: 8px;
            }}

            .panel p {{
                margin-top: 0;
                color: #64748b;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}

            th, td {{
                padding: 14px 12px;
                border-bottom: 1px solid #e2e8f0;
                text-align: left;
                vertical-align: middle;
            }}

            th {{
                background: #f1f5f9;
                font-size: 14px;
                color: #334155;
            }}

            .bar-wrapper {{
                width: 180px;
                height: 12px;
                background: #e2e8f0;
                border-radius: 999px;
                overflow: hidden;
            }}

            .bar {{
                height: 100%;
                border-radius: 999px;
            }}

            .badge {{
                display: inline-block;
                padding: 6px 10px;
                border-radius: 999px;
                font-size: 13px;
                font-weight: 600;
            }}

            .legend {{
                margin-top: 18px;
                font-size: 14px;
                color: #475569;
                line-height: 1.8;
            }}

            .legend span {{
                display: inline-block;
                margin-right: 16px;
            }}

            .dot {{
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 6px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Model Drift Dashboard</h1>
                <p>Population Stability Index (PSI) monitoring by feature.</p>
            </div>

            <div class="cards">
                <div class="card">
                    <div class="card-title">Features Analyzed</div>
                    <div class="card-value">{n_features}</div>
                </div>
                <div class="card">
                    <div class="card-title">No Drift</div>
                    <div class="card-value" style="color:#16a34a;">{no_drift}</div>
                </div>
                <div class="card">
                    <div class="card-title">Moderate Drift</div>
                    <div class="card-value" style="color:#ca8a04;">{moderate}</div>
                </div>
                <div class="card">
                    <div class="card-title">Significant Drift</div>
                    <div class="card-value" style="color:#dc2626;">{significant}</div>
                </div>
            </div>

            <div class="panel">
                <h2>PSI by Feature</h2>
                <p>This dashboard summarizes distribution changes between baseline and recent production data.</p>

                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>PSI</th>
                            <th>Visual</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>

                <div class="legend">
                    <span><span class="dot" style="background:#16a34a;"></span>PSI &lt; 0.10: No drift</span>
                    <span><span class="dot" style="background:#ca8a04;"></span>0.10 ≤ PSI &lt; 0.25: Moderate drift</span>
                    <span><span class="dot" style="background:#dc2626;"></span>PSI ≥ 0.25: Significant drift</span>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

