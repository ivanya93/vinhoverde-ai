"""
Tiny synthetic stand-ins for the two real datasets, so tests run in seconds, offline, in CI.

They copy the real files' SCHEMA (column names, separators, value ranges) — not their
statistics. Nothing here says anything about real wine.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402


def fake_uci(n: int, colour: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    alcohol = rng.uniform(8.5, 14.0, n)
    sulphates = rng.uniform(0.35, 1.2, n)
    volatile = rng.uniform(0.1, 1.2, n)
    # Quality loosely driven by alcohol, sulphates and volatile acidity, like the real data.
    signal = 0.6 * (alcohol - 10.5) + 2.0 * (sulphates - 0.6) - 2.0 * (volatile - 0.5)
    quality = np.clip(np.round(5.8 + signal + rng.normal(0, 0.6, n)), 3, 9).astype(int)
    total_so2 = rng.uniform(10, 200, n)
    return pd.DataFrame({
        "fixed acidity": rng.uniform(4, 15, n),
        "volatile acidity": volatile,
        "citric acid": rng.uniform(0, 0.8, n),
        "residual sugar": rng.uniform(0.9, 20, n),
        "chlorides": rng.uniform(0.01, 0.2, n),
        "free sulfur dioxide": total_so2 * rng.uniform(0.1, 0.5, n),
        "total sulfur dioxide": total_so2,
        "density": rng.uniform(0.99, 1.004, n),
        "pH": rng.uniform(2.8, 3.9, n),
        "sulphates": sulphates,
        "alcohol": alcohol,
        "quality": quality,
    })


def fake_kaggle() -> pd.DataFrame:
    rows = [
        ("Portugal", "Quinta do Vale 2015 Reserva Red (Douro)", "Quinta do Vale", "Touriga Nacional",
         "Douro", 92, 28.0, "Dense and structured with black plum, violets and firm tannins. Great with roast lamb."),
        ("Portugal", "Casa Alta 2019 Alvarinho (Vinho Verde)", "Casa Alta", "Alvarinho",
         "Vinho Verde", 89, 14.0, "Crisp, citrus and saline minerality. Perfect with grilled sardines and seafood."),
        ("Portugal", "Herdade Sol 2017 Tinto (Alentejo)", "Herdade Sol", "Portuguese Red",
         "Alentejano", 88, 12.0, "Ripe, juicy red fruit, soft tannins, easy drinking with grilled meats."),
        ("Portugal", "Adega Mar 2018 Encruzado (Dão)", "Adega Mar", "Encruzado",
         "Dão", 91, 22.0, "Creamy texture, pear and toasty oak, bright acidity. Try with bacalhau."),
        ("Portugal", "Baga Velha 2014 Bairrada", "Baga Velha", "Baga",
         "Bairrada", 90, 19.0, "High acidity, sour cherry and earthy notes. Classic with leitão (suckling pig)."),
        ("Spain", "Bodega X 2016 Rioja", "Bodega X", "Tempranillo",
         "Northern Spain", 90, 20.0, "Vanilla and cherry."),
    ]
    df = pd.DataFrame(rows, columns=["country", "title", "winery", "variety", "province",
                                     "points", "price", "description"])
    df["region_1"] = df["province"]
    return df


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    fake_uci(300, "red", 1).to_csv(tmp_path / config.UCI_FILES["red"], sep=";", index=False)
    fake_uci(500, "white", 2).to_csv(tmp_path / config.UCI_FILES["white"], sep=";", index=False)
    fake_kaggle().to_csv(tmp_path / config.KAGGLE_FILE)
    return tmp_path
