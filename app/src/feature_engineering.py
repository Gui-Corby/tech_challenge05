import pandas as pd
import numpy as np
from config import DF_2024


# -------------------------
# Helpers
# -------------------------
def _require_cols(df: pd.DataFrame, cols: list[str], fn_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{fn_name}: faltam colunas {missing}")


def _to_float_ptbr(series: pd.Series) -> pd.Series:
    """
    Converte strings numéricas no formato pt-BR (vírgula decimal) para float.
    Mantém NaN quando não der pra converter.
    """
    if series.dtype == "object":
        return pd.to_numeric(series.astype(str).str.replace(",", "."),
                             errors="coerce")
    return series


def extrair_numero_fase(fase) -> float:
    """Converte valores tipo '1 - ...', '2', 'ALFA' em inteiro (ou NaN)."""
    if pd.isna(fase):
        return np.nan

    s = str(fase).strip().upper()

    if s == "ALFA":
        return 0

    try:
        return int(s[0])
    except Exception:
        return np.nan


# -------------------------
# Feature blocks
# -------------------------
def create_temp_features(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, ["Ano ingresso", "Idade"], "create_temp_features")
    df = df.copy()
    df["Tempo_na_PM"] = 2024 - df["Ano ingresso"]
    df["Idade_Ingresso"] = df["Idade"] - df["Tempo_na_PM"]
    return df


def create_academic_features(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, ["Mat", "Por", "Ing", "Pedra 2024", "Pedra 23"],
                  "create_academic_features")
    df = df.copy()

    df["Media_Notas"] = df[["Mat", "Por"]].mean(axis=1)
    df["Tem_Ingles"] = (~df["Ing"].isna()).astype(int)

    # Classificação Pedra como numérica
    pedra_ordem = {"Quartzo": 1, "Ágata": 2, "Ametista": 3, "Topázio": 4}
    df["Pedra_2024_num"] = df["Pedra 2024"].map(pedra_ordem)
    df["Pedra_2023_num"] = df["Pedra 23"].map(pedra_ordem)

    return df


def temp_evol_features(df: pd.DataFrame) -> pd.DataFrame:
    # aqui depende de colunas criadas em create_academic_features (Pedra_2024_num / Pedra_2023_num)
    _require_cols(
        df,
        ["Pedra_2024_num", "Pedra_2023_num", "Pedra_2022_num", "Pedra_2021_num",
         "INDE 2024", "INDE 23", "INDE 22"],
        "temp_evol_features",
    )
    df = df.copy()

    df["Evolucao_Pedra_23_24"] = df["Pedra_2024_num"] - df["Pedra_2023_num"]
    df["Evolucao_Pedra_22_23"] = df["Pedra_2023_num"] - df["Pedra_2022_num"]
    df["Evolucao_Pedra_21_22"] = df["Pedra_2022_num"] - df["Pedra_2021_num"]

    # Evolução do INDE (normaliza se vier como string)
    df["INDE 2024"] = _to_float_ptbr(df["INDE 2024"])

    df["Evolucao_INDE_23_24"] = df["INDE 2024"] - df["INDE 23"]
    df["Evolucao_INDE_22_23"] = df["INDE 23"] - df["INDE 22"]

    # Taxa de evolução (percentual)
    df["Taxa_Evolução_INDE_23_24"] = ((df["INDE 2024"] - df["INDE 23"]) / (df["INDE 23"] + 0.001)) * 100

    return df


def level_features(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, ["Fase", "Fase Ideal"], "level_features")
    df = df.copy()

    df["Fase_num"] = df["Fase"].apply(extrair_numero_fase)
    df["Fase_Ideal_num"] = df["Fase Ideal"].apply(extrair_numero_fase)
    df["Deficit_Fase"] = df["Fase_Ideal_num"] - df["Fase_num"]

    return df


def interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    # depende de Media_Notas (create_academic_features)
    _require_cols(df, ["INDE 2024", "IAN", "IDA", "IEG", "Media_Notas"], "interaction_features")
    df = df.copy()

    df["INDE_x_IAN"] = df["INDE 2024"] * df["IAN"]
    df["IDA_x_IEG"] = df["IDA"] * df["IEG"]
    df["Media_Notas_x_IAN"] = df["Media_Notas"] * df["IAN"]

    return df


def aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, ["IPS", "IPP", "IAA", "IDA", "IEG", "IPV"],
                  "aggregated_features")
    df = df.copy()

    # Média dos indicadores psico-educacionais
    df["Media_Indicadores_Psico"] = df[["IPS", "IPP", "IAA"]].mean(axis=1)

    # Média dos indicadores de desempenho
    df["Media_Indicadores_Desemp"] = df[["IDA", "IEG", "IPV"]].mean(axis=1)

    return df


def binary_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    # depende de Tempo_na_PM (create_temp_features)
    _require_cols(df, ["Gênero", "Tempo_na_PM", "Idade"],
                  "binary_categorical_features")
    df = df.copy()

    df["Genero_bin"] = (df["Gênero"] == "Menino").astype(int)
    df["Veterano"] = (df["Tempo_na_PM"] >= 2).astype(int)

    df["Faixa_Etaria"] = pd.cut(
        df["Idade"],
        bins=[0, 10, 13, 16, 100],
        labels=["Criança", "Pré-adolescente", "Adolescente", "Jovem"],
    )

    return df


def future_improvements_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Tendencia_Melhora"] = (
        (df["Evolução_Pedra_23_24"].fillna(0) > 0).astype(int) +
        (df["Evolução_Pedra_22_23"].fillna(0) > 0).astype(int)
    )

    return df


def consistency_features(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, ["Mat", "Por"], "consistency_features")
    df = df.copy()

    # Desvio padrão das notas (variabilidade no desempenho)
    df["Variabilidade_Notas"] = df[["Mat", "Por"]].std(axis=1)

    return df


# Pipeline (ordem garantida)
def build_features_2024(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline único que garante a ordem das features.
    """
    df = df.copy()

    df = create_temp_features(df)
    df = create_academic_features(df)
    df = temp_evol_features(df)
    df = level_features(df)
    df = interaction_features(df)
    df = aggregated_features(df)
    df = binary_categorical_features(df)
    df = future_improvements_features(df)
    df = consistency_features(df)

    return df


def build_df_2024_featured() -> pd.DataFrame:
    """
    Conveniência: aplica o pipeline no DF_2024 do config e devolve um df novo.
    """
    return build_features_2024(DF_2024)
