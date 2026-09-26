import pandas as pd

from src import config
from src.data import build_portugal_catalog, infer_colour, load_uci
from src.features import OUTPUT_FEATURES, WineFeatures


def test_load_uci_combines_colours_and_drops_duplicates(raw_dir):
    red = pd.read_csv(raw_dir / config.UCI_FILES["red"], sep=";")
    pd.concat([red, red.head(10)]).to_csv(raw_dir / config.UCI_FILES["red"], sep=";", index=False)

    wine = load_uci(raw_dir)
    assert set(wine[config.TYPE_COLUMN]) == {"red", "white"}
    assert not wine.duplicated().any()
    assert set(wine[config.TARGET]) <= set(config.TIERS)


def test_quality_tiers():
    assert [config.quality_to_tier(q) for q in (3, 5, 6, 7, 9)] == \
        ["standard", "standard", "good", "premium", "premium"]


def test_portugal_catalog_filters_country_and_adds_colour(raw_dir):
    cat = build_portugal_catalog(raw_dir / config.KAGGLE_FILE)
    assert len(cat) == 5                      # the Spanish wine is gone
    assert {"red", "white"} <= set(cat["colour"])
    assert infer_colour("Alvarinho") == "white"
    assert infer_colour("Touriga Nacional") == "red"


def test_features_are_numeric_and_complete(raw_dir):
    wine = load_uci(raw_dir)
    X = WineFeatures().fit_transform(wine[config.INPUT_COLUMNS])
    assert list(X.columns) == OUTPUT_FEATURES
    assert not X.isna().any().any()
    assert set(X["is_red"]) == {0.0, 1.0}
