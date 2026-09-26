"""
One place for names and paths the whole project agrees on.

The training script, the registry script, the API and the tests all import from here, so
a model name or a feature list can never drift between the thing that trains and the
thing that serves.
"""

import os
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# --- UCI Wine Quality (Vinho Verde, red + white) ------------------------------------
UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality"
UCI_FILES = {"red": "winequality-red.csv", "white": "winequality-white.csv"}

# --- Kaggle Wine Reviews (WineEnthusiast) --------------------------------------------
KAGGLE_DATASET = "zynicide/wine-reviews"
KAGGLE_FILE = "winemag-data-130k-v2.csv"
PORTUGAL_CATALOG = PROCESSED_DIR / "portugal_wines.csv"

# --- ML ------------------------------------------------------------------------------
# The eleven physicochemical measurements, exactly as the UCI files name them (spaces and
# all), plus the wine colour. These are what the API accepts.
RAW_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TYPE_COLUMN = "wine_type"  # "red" | "white"
INPUT_COLUMNS = RAW_FEATURES + [TYPE_COLUMN]

# Quality is scored 0-10 by tasters; in practice 3-9. We turn it into three business
# decisions a winery actually makes before bottling.
TARGET = "quality_tier"
TIERS = ["standard", "good", "premium"]  # <=5, 6, >=7


def quality_to_tier(quality: int) -> str:
    if quality <= 5:
        return "standard"
    if quality == 6:
        return "good"
    return "premium"


TIER_ACTIONS = {
    "standard": "Blend or sell as house wine / bulk. Do not bottle under the estate label.",
    "good": "Bottle under the regular label. Good value for retail and restaurants.",
    "premium": "Candidate for the reserve label. Send a sample to the head winemaker.",
}

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "VinhoVerde-Quality")
MODEL_NAME = os.getenv("MODEL_NAME", "VinhoVerde-Quality-Classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# The promotion gate. A candidate must clear the floor AND beat the current champion.
# Set just above logreg_baseline's macro-F1 (0.543, our weakest reasonable config) so a
# model no better than plain logistic regression no longer ships, while rf_balanced
# (0.593) and HistGB (0.563) still clear it. A model that fails this would push more
# premium-quality wines toward misclassification — real bottling/revenue loss per
# config.TIER_ACTIONS.
MIN_MACRO_F1 = float(os.getenv("MIN_MACRO_F1", 0.55))
RANDOM_STATE = 42
TEST_SIZE = 0.2
