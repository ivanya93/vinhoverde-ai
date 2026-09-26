"""
Feature engineering, as a scikit-learn transformer.

It lives INSIDE the model pipeline on purpose. The registered model then takes the raw lab
measurements and does its own feature engineering, so the API never has to re-implement it
— and can never re-implement it slightly differently (training/serving skew).
How to find good ones: open notebooks/01_eda.ipynb, run the correlation cell, and look at
which raw measurements move together with quality. A good engineered feature usually
combines two raw ones in a way a single raw column can't express (a ratio, a product, a
flag). 

"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import RAW_FEATURES, TYPE_COLUMN

ENGINEERED = [
    "is_red",
    "free_so2_ratio",       # free sulfur dioxide / total sulfur dioxide (share of SO2 that's still protecting the wine)
    "total_acidity",        # fixed acidity + volatile acidity + citric acid
    "alcohol_sugar_ratio",  # alcohol / (residual sugar + 1) (dry&strong vs sweet&light)
    "sulphates_x_alcohol",  # sulphates * alcohol (two strong single predictors)
    "density_adjusted"      # density adjusted for sugar or alcohol (density correlates with both)
]

OUTPUT_FEATURES = RAW_FEATURES + ENGINEERED

class WineFeatures(BaseEstimator, TransformerMixin):
    """Raw measurements + wine_type in, numeric feature matrix out."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        out = df[RAW_FEATURES].astype(float)

        # Worked example: colour as a 0/1 flag the model can use.
        out["is_red"] = (df[TYPE_COLUMN].astype(str).str.lower() == "red").astype(float)
        out["total_acidity"] = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
        out["free_so2_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1)
        out["alcohol_sugar_ratio"] = df["alcohol"] / (df["residual sugar"] + 1)
        out["sulphates_x_alcohol"] = df["sulphates"] * df["alcohol"]
        out["density_adjusted"] = df["density"] / (df["residual sugar"] + 1)

        return out[OUTPUT_FEATURES].fillna(0.0)

    def get_feature_names_out(self, input_features=None):
        return np.array(OUTPUT_FEATURES)
