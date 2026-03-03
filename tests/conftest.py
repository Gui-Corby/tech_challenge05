import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def df_minimo():
    # DataFrame mínimo que satisfaz todos os _require_cols do pipeline
    return pd.DataFrame({
        # create_temp_features
        "Ano ingresso": ["2022", "2023", "2021", "2020"],
        "Idade": [15, 16, 17, 18],

        # create_academic_features + consistency_features
        "Mat": [7.0, 8.0, 6.5, np.nan],
        "Por": [6.0, 9.0, 7.0, 8.0],
        "Ing": [np.nan, 7.0, np.nan, 6.0],
        "Pedra 2024": ["Quartzo", "Topázio", "Ágata", "Ametista"],
        "Pedra 23": ["Ágata", "Ametista", "Quartzo", "Topázio"],
        "Pedra 22": ["Quartzo", "Ágata", "Quartzo", "Ágata"],
        "Pedra 21": ["Quartzo", "Quartzo", "Ágata", "Quartzo"],

        # temp_evol_features
        "INDE 2024": ["7,5", "8,0", "7,0", "8,5"],  # pt-BR com vírgula
        "INDE 23": ["7.0", "7.5", "6.8", "8.0"],
        "INDE 22": [6.5, 7.0, 6.6, 7.8],

        # level_features
        "Fase": ["1 - X", "ALFA", "2 - Y", np.nan],
        "Fase Ideal": ["2 - Y", "1 - Z", "2 - Y", "3 - W"],

        # interaction_features / aggregated_features
        "IAN": [0.5, 0.7, 0.6, 0.4],
        "IDA": [0.4, 0.6, 0.5, 0.3],
        "IEG": [0.2, 0.3, 0.25, 0.1],
        "IPV": [0.1, 0.2, 0.15, 0.05],
        "IPS": [0.3, 0.4, 0.35, 0.2],
        "IPP": [0.3, 0.4, 0.33, 0.25],
        "IAA": [0.3, 0.4, 0.31, 0.22],

        # binary_categorical_features
        "Gênero": ["Menino", "Menina", "Menino", "Menina"],

        # target (para smoke tests)
        "Defasagem": [-1, 0, 1, 0],
    })
