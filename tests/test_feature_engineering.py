import numpy as np

from feature_engineering import build_features_2024, extrair_numero_fase


def test_extrair_numero_fase():
    assert extrair_numero_fase("ALFA") == 0
    assert extrair_numero_fase("1 - alguma coisa") == 1
    assert extrair_numero_fase("2") == 2
    assert np.isnan(extrair_numero_fase(np.nan))


def test_build_features_creates_expected_columns(df_minimo):
    out = build_features_2024(df_minimo)

    expected_cols = [
        "Tempo_na_PM",
        "Idade_Ingresso",
        "Media_Notas",
        "Tem_Ingles",
        "Pedra_2024_num",
        "Pedra_2023_num",
        "Pedra_2022_num",
        "Pedra_2021_num",
        "Evolucao_Pedra_23_24",
        "Evolucao_INDE_23_24",
        "Taxa_Evolução_INDE_23_24",
        "Fase_num",
        "Fase_Ideal_num",
        "Deficit_Fase",
        "INDE_x_IAN",
        "IDA_x_IEG",
        "Media_Notas_x_IAN",
        "Media_Indicadores_Psico",
        "Media_Indicadores_Desemp",
        "Genero_bin",
        "Veterano",
        "Faixa_Etaria",
        "Tendencia_Melhora",
        "Variabilidade_Notas",
    ]

    for c in expected_cols:
        assert c in out.columns
