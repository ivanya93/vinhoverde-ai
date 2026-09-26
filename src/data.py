"""
Get the two datasets, clean them, and write what the rest of the project reads.

    python -m src.data                 # download UCI red + white, build the ML table
    python -m src.data --kaggle        # also build the Portuguese wine catalog

Two sources, two jobs — they are never joined (they share no key):

    UCI Wine Quality   -> the ML model    (lab measurements -> quality tier)
    Kaggle Wine Reviews -> the sommelier  (Portuguese wines, regions, prices, tasting notes)
"""

import argparse
import hashlib
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

WINE_TABLE = config.PROCESSED_DIR / "wine_quality.csv"


# --------------------------------------------------------------------------------------
# UCI Wine Quality
# --------------------------------------------------------------------------------------
def download_uci(force: bool = False) -> None:
    """Fetch both colour files from UCI into data/raw/. Skips files already there."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for colour, filename in config.UCI_FILES.items():
        target = config.RAW_DIR / filename
        if target.exists() and not force:
            logger.info("  %s already present", filename)
            continue
        url = f"{config.UCI_BASE_URL}/{filename}"
        logger.info("  downloading %s", url)
        # pandas reads straight from the URL; the files use ';' as separator.
        pd.read_csv(url, sep=";").to_csv(target, sep=";", index=False)


def load_uci(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """
    Combine red and white into one table with a `wine_type` column and a target tier.

    Duplicates are dropped BEFORE any split. The UCI files contain ~1,100 exact duplicate
    rows; left in, the same wine lands in train and test and the test score flatters the
    model. This is a leakage bug you can show in the presentation.
    """
    frames = []
    for colour, filename in config.UCI_FILES.items():
        df = pd.read_csv(raw_dir / filename, sep=";")
        df[config.TYPE_COLUMN] = colour
        frames.append(df)

    wine = pd.concat(frames, ignore_index=True)
    before = len(wine)
    wine = wine.drop_duplicates().reset_index(drop=True)
    logger.info("  %d rows, %d exact duplicates removed -> %d", before, before - len(wine), len(wine))

    # Some columns are whole numbers in one file and decimals in the other; make them all
    # float so the model signature does not reject a perfectly valid "11.5" later.
    wine[config.RAW_FEATURES] = wine[config.RAW_FEATURES].astype(float)
    wine[config.TARGET] = wine["quality"].apply(config.quality_to_tier)
    return wine


def file_fingerprint(path: Path) -> str:
    """Short content hash, logged to MLflow so every run says exactly which data it saw."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


# --------------------------------------------------------------------------------------
# Kaggle Wine Reviews -> Portuguese catalog
# --------------------------------------------------------------------------------------
# The Kaggle file has no colour column. For Portuguese wines the variety tells us.
RED_VARIETIES = {
    "portuguese red", "touriga nacional", "tinta roriz", "touriga franca", "baga",
    "castelão", "castelao", "trincadeira", "alicante bouschet", "aragonez", "tinta barroca",
    "syrah", "cabernet sauvignon", "tinto cão", "tinta cão", "jaen", "alfrocheiro",
    "vinhão", "merlot", "pinot noir", "port", "red blend", "bordeaux-style red blend",
}
WHITE_VARIETIES = {
    "portuguese white", "alvarinho", "loureiro", "arinto", "encruzado", "antão vaz",
    "avesso", "fernão pires", "bical", "verdelho", "sercial", "malvasia", "viosinho",
    "rabigato", "chardonnay", "sauvignon blanc", "white blend", "moscatel", "gouveio",
}


def infer_colour(variety: str) -> str:
    v = str(variety).lower()
    if "rosé" in v or "rose" in v:
        return "rosé"
    if v in RED_VARIETIES or "red" in v or "tint" in v:
        return "red"
    if v in WHITE_VARIETIES or "white" in v or "branco" in v:
        return "white"
    return "unknown"


def download_kaggle() -> Path:
    """
    Download the Kaggle dataset. Tries `kagglehub` first, then the `kaggle` CLI.

    Both need a Kaggle account token (~/.kaggle/kaggle.json or KAGGLE_USERNAME /
    KAGGLE_KEY). If neither works, download the CSV by hand from
    https://www.kaggle.com/datasets/zynicide/wine-reviews and drop
    winemag-data-130k-v2.csv into data/raw/.
    """
    target = config.RAW_DIR / config.KAGGLE_FILE
    if target.exists():
        logger.info("  %s already present", config.KAGGLE_FILE)
        return target

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub

        folder = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
        shutil.copy(folder / config.KAGGLE_FILE, target)
        return target
    except Exception as exc:  # noqa: BLE001 — fall through to the manual instructions
        logger.warning("  kagglehub failed: %s", exc)

    sys.exit(
        f"Could not download {config.KAGGLE_FILE}.\n"
        f"Download it manually from https://www.kaggle.com/datasets/{config.KAGGLE_DATASET}\n"
        f"and save it as {target}"
    )


def build_portugal_catalog(raw_file: Path) -> pd.DataFrame:
    """Keep the Portuguese reviews, tidy them, and add a colour column."""
    reviews = pd.read_csv(raw_file, index_col=0)
    pt = reviews[reviews["country"] == "Portugal"].copy()

    keep = ["title", "winery", "variety", "province", "region_1", "points", "price", "description"]
    pt = pt[[c for c in keep if c in pt.columns]]
    pt = pt.dropna(subset=["description", "title"]).drop_duplicates(subset=["title", "description"])
    pt["colour"] = pt["variety"].apply(infer_colour)
    pt = pt.reset_index(drop=True)

    logger.info("  %d Portuguese wines | colours: %s", len(pt), pt["colour"].value_counts().to_dict())
    return pt


# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download and prepare the datasets")
    parser.add_argument("--kaggle", action="store_true", help="also build the Portuguese catalog")
    parser.add_argument("--force", action="store_true", help="re-download UCI files")
    args = parser.parse_args()

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("UCI Wine Quality (red + white Vinho Verde)")
    download_uci(force=args.force)
    wine = load_uci()
    wine.to_csv(WINE_TABLE, index=False)
    logger.info("  tiers: %s", wine[config.TARGET].value_counts().to_dict())
    logger.info("  -> %s (sha %s)", WINE_TABLE, file_fingerprint(WINE_TABLE))

    if args.kaggle:
        logger.info("\nKaggle Wine Reviews -> Portuguese catalog")
        raw = download_kaggle()
        catalog = build_portugal_catalog(raw)
        catalog.to_csv(config.PORTUGAL_CATALOG, index=False)
        logger.info("  -> %s", config.PORTUGAL_CATALOG)


if __name__ == "__main__":
    main()
