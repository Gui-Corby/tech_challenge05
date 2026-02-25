import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import DF_2024, NUMERIC_FEATURES, TARGET_COL
from feature_engineering import build_features_2024
from preprocessing import filter_age, replace_infs, make_preprocessor, check_all_nan_columns

df = build_features_2024(DF_2024).copy()

# Limpeza fora do sklearn
df = filter_age(df, max_age=19)
df = replace_infs(df, NUMERIC_FEATURES)

nan_cols = check_all_nan_columns(df, NUMERIC_FEATURES)

if nan_cols:
    print("Removendo:", nan_cols)
    df = df.drop(columns=nan_cols)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].dropna()
X = X.loc[y.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

pipeline = Pipeline(steps=[
    ("preprocess", make_preprocessor(df)),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        multi_class="multinomial",
        solver="lbfgs"
    )),
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
