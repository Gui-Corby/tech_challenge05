
## Overview

This project implements a complete Machine Learning Engineering pipeline
designed to predict student academic defasagem (learning delay).

The system follows production-oriented best practices, including:

-   End-to-end sklearn Pipeline
-   Modular feature engineering
-   REST API for real-time inference (FastAPI)
-   Monitoring integration with Prometheus
-   Drift detection using PSI (Population Stability Index)
-   Automated testing with 90% code coverage

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
    │   ├── __init__.py              # Marks src as a Python package
    │   ├── config.py                # Central configuration (paths, constants, target column)
    │   ├── feature_engineering.py   # Feature construction logic
    │   ├── preprocessing.py         # Data cleaning and preprocessing utilities
    │   ├── ml_pipeline.py           # Feature engineering block used inside sklearn Pipeline
    │   ├── train.py                 # Model training script and artifact generation
    │   ├── evaluate.py              # Holdout evaluation and metrics generation
    │   ├── drift_monitor.py         # PSI-based drift detection utilities
    │
    ├── app/
    │   ├── main.py                  # FastAPI application entrypoint
    │   ├── routes.py                # API endpoints (predict, drift, metrics)
    │
    ├── tests/
    │   ├── conftest.py              # Shared fixtures for tests
    │   ├── test_feature_engineering.py  # Unit tests for feature logic
    │   ├── test_preprocessing.py        # Tests for preprocessing utilities
    │   ├── test_ml_pipeline.py          # Tests for pipeline feature block
    │   ├── test_pipeline_smoke.py       # End-to-end pipeline smoke test
    │   ├── test_train.py                # Training execution test (artifact validation)
    │   ├── test_evaluate.py             # Evaluation script test with mocked artifacts
    │   ├── test_drift_monitor.py        # PSI drift calculation tests
    │
    ├── artifacts/                  # Saved models and evaluation outputs
    │   ├── pipeline.joblib         # Serialized sklearn pipeline
    │   ├── metrics.json            # Training metrics
    │   ├── metrics_eval.json       # Evaluation metrics
    │   ├── test.csv                # Holdout dataset used in evaluation
    │
    ├── pytest.ini                  # Pytest configuration
    ├── requirements.txt            # Production dependencies
    ├── requirements-dev.txt        # Development dependencies (testing, coverage)
    ├── notebook.ipynb              # Exploratory analysis notebook
    ├── .flake8                    
    ├── .coverage                 
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

Current coverage: 90%

Test coverage includes:

-   Feature engineering validation
-   Preprocessing validation
-   Pipeline smoke tests
-   Drift calculation tests
-   Train and evaluate module tests

------------------------------------------------------------------------

## Docker

This project can be executed in an isolated and reproducible environment using Docker.

### Prerequisites

- Docker installed (Docker Desktop or Docker Engine)

### 1. Clone the repository

```bash
git clone https://github.com/Gui-Corby/tech_challenge05
cd tech_challenge05
```

### 2. Build the Docker image
```bash
docker build -t tc05-api .
```

### 3. Run the container
```bash
docker run --rm -p 8000:8000 tc05-api
```

### 4. Access the API
open in your browser: http://localhost:8000/docs
The interactive Swagger documentation will be available for testing the endpoints.

The Docker image includes:

- The trained model artifacts

- The FastAPI application

- All required dependencies

- An isolated runtime environment

------------------------------------------------------------------------

## Engineering Practices Applied

-   Modular architecture
-   Reproducible ML pipeline
-   Clear separation between training and inference
-   Monitoring integration
-   Drift detection
-   Automated testing
-   High code coverage
- Docker environmente

------------------------------------------------------------------------
