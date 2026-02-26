import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DF_2024 = pd.read_excel(BASE_DIR / "data" / "excel_file.xlsx", sheet_name="PEDE2024")

TARGET_COL = "Defasagem"

NUMERIC_FEATURES = [
    "IAN", "INDE 2024", "IEG", "IDA", "IPV", "IPS", "IPP", "IAA",
    "Media_Notas", "Mat", "Por", "Variabilidade_Notas",
    "Pedra_2024_num", "Pedra_2023_num", "Pedra_2022_num",
    "Idade", "Tempo_na_PM", "Idade_Ingresso", "Fase_num",
    "Deficit_Fase", "INDE_x_IAN", "IDA_x_IEG", "Media_Notas_x_IAN",
    "Evolucao_Pedra_23_24", "Evolucao_INDE_23_24", "Taxa_Evolucao_INDE_23_24",
    "Media_Indicadores_Psico", "Media_Indicadores_Desemp",
    "Tendencia_Melhora", "Nº Av",
    "Genero_bin", "Tem_Ingles", "Veterano",
]
