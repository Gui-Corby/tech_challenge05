# Tech Challenge 05 --- Machine Learning Engineering

## Overview

This project implements a complete Machine Learning Engineering pipeline
designed to predict student academic defasagem (learning delay).

The system follows production-oriented best practices, including:

-   End-to-end sklearn Pipeline
-   Modular feature engineering
-   REST API for real-time inference (FastAPI)
-   Monitoring integration with Prometheus
-   Drift detection using PSI (Population Stability Index)
-   Automated testing with over 90% code coverage

------------------------------------------------------------------------

## Problem Statement

The objective is to predict whether a student presents academic
defasagem based on:

-   Academic performance indicators
-   Historical INDE scores
-   Demographic attributes
-   Pedagogical phase progression

This project simulates a real-world ML system capable of supporting
educational decision-making processes.

------------------------------------------------------------------------

## Project Structure

    tc05/
    │
    ├── src/
    │   ├── config.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── ml_pipeline.py
    │   ├── feature_engineering.py
    │   ├── preprocessing.py
    │   ├── drift_monitor.py
    │
    ├── app/
    │   ├── main.py
    │   ├── routes.py
    │
    ├── tests/
    ├── artifacts/
    ├── requirements.txt
    ├── pytest.ini
    └── README.md

------------------------------------------------------------------------

## Machine Learning Pipeline

### Model

-   Logistic Regression
-   class_weight="balanced"
-   Solver: lbfgs
-   Implemented within a full sklearn Pipeline

### Pipeline Stages

1.  Feature Engineering\
2.  Data Cleaning\
3.  Preprocessing\
4.  Model Training

Generated artifacts:

-   artifacts/pipeline.joblib\
-   artifacts/metrics.json\
-   artifacts/test.csv

------------------------------------------------------------------------

## Training

    python src/train.py

This process:

-   Performs train/test split (stratified when possible)
-   Trains the pipeline
-   Saves evaluation metrics and model artifacts

------------------------------------------------------------------------

## Evaluation

    python src/evaluate.py

Outputs:

-   Classification report
-   Confusion matrix
-   metrics_eval.json file

------------------------------------------------------------------------

## API --- Real-Time Inference

Start the server:

    uvicorn app.main:app --reload

Interactive documentation:

    http://127.0.0.1:8000/docs

### Endpoint

POST /predict

Example request:

``` json
{
  "history": [
    {
      "IAN": 7.4,
      "IDA": 6.9,
      "IEG": 7.2,
      "IPS": 7.1,
      "IPP": 6.8,
      "IAA": 7.0,
      "INDE_2024": 7.15,
      "INDE_23": 6.8,
      "INDE_22": 6.45,
      "Mat": 7.9,
      "Por": 6.7,
      "Ingles": 7.3,
      "Pedra_2024": "Ametista",
      "Pedra_23": "Ágata",
      "Pedra_22": "Ágata",
      "Pedra_21": "Quartzo",
      "Idade": 15,
      "Ano_ingresso": 2021,
      "N_Av": 4,
      "Fase": "2 - II Fase",
      "Fase_Ideal": "3 - III Fase",
      "Genero": "Menino"
    }
  ]
}
```

The response includes prediction and probability scores.

------------------------------------------------------------------------

## Monitoring

Metrics are exposed at:

    /metrics

Prometheus metrics include:

-   HTTP request count
-   Request latency histogram
-   CPU usage
-   Memory usage

------------------------------------------------------------------------

## Drift Detection

Drift detection is implemented using PSI (Population Stability Index).

Outputs:

-   Drift logs saved in artifacts/drift_log.json
-   Endpoint available at /drift

Interpretation thresholds:

-   PSI \< 0.1: No drift\
-   0.1 ≤ PSI \< 0.25: Moderate drift\
-   PSI ≥ 0.25: Significant drift

------------------------------------------------------------------------

## Testing and Quality Assurance

Run tests:

    pytest

Run with coverage:

    pytest --cov=src --cov-report=term-missing

Current coverage: above 90%

Test coverage includes:

-   Feature engineering validation
-   Preprocessing validation
-   Pipeline smoke tests
-   Drift calculation tests
-   Train and evaluate module tests

------------------------------------------------------------------------

## Engineering Practices Applied

-   Modular architecture
-   Reproducible ML pipeline
-   Clear separation between training and inference
-   Monitoring integration
-   Drift detection
-   Automated testing
-   High code coverage

------------------------------------------------------------------------

## Future Improvements

-   Docker containerization
-   Cloud deployment
-   CI/CD integration
-   Model versioning strategy

------------------------------------------------------------------------

## Author

Guilherme Corby Moreira

Machine Learning Engineering Project --- Tech Challenge 05
