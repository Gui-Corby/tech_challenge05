import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

from config import DF_2024, NUMERIC_FEATURES, TARGET_COL
from feature_engineering import build_features_2024
from preprocessing import filter_age, replace_infs, make_preprocessor


