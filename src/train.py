"""
Train and compare several model configurations, logging everything to MLflow.

    python -m src.train                        # all configurations
    python -m src.train --only rf_balanced     # just one

Each configuration is one MLflow run with: parameters, metrics, the dataset it saw (with a
content hash), a confusion matrix, feature importances, and the fitted pipeline as a
logged model with a signature. Nothing is registered here — that is src/register.py's job,
so "training" and "deciding what ships" stay two separate, reviewable steps.
"""

import argparse
import logging
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import config  # noqa: E402
from src.data import WINE_TABLE, file_fingerprint  # noqa: E402
from src.features import OUTPUT_FEATURES, WineFeatures  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SRC_DIR = str(config.ROOT / "src")

# Types skops will not load unless we vouch for them (see the note at log_model below).
#   - our own feature transformer
#   - the tree node storage behind RandomForest (raw indices, no bounds checks)
#   - the equivalent for HistGradientBoosting
TRUSTED_TYPES = [
    "src.features.WineFeatures",
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]

# ---------------------------------------------------------------------
CONFIGS = {
    "logreg_baseline": {
        "description": "Scaled logistic regression, balanced classes — the bar to beat",
        "build": lambda: [("scale", StandardScaler()),
                          ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))],
    },
    "rf_balanced": {
        "description": "Random forest, 300 trees, balanced class weights",
        "build": lambda: [("model", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=config.RANDOM_STATE))],
    },
    "HistGB": {
        "description": "HistGradientBoostingClassifier, 200 iterations, balanced class weights",
        "build": lambda: [("model", HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_depth=None, random_state=config.RANDOM_STATE))],
    },
    "rf_unbalanced": {
        "description": "Same random forest as rf_balanced, but WITHOUT class weighting",
        "build": lambda: [("model", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=config.RANDOM_STATE))],
    },
}

def split(df: pd.DataFrame):
    """The one train/test split. register.py re-creates it to score candidate vs champion."""
    X = df[config.INPUT_COLUMNS]
    y = df[config.TARGET]
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y,
                            random_state=config.RANDOM_STATE)

def build_pipeline(config_name: str) -> Pipeline:
    return Pipeline([("features", WineFeatures())] + CONFIGS[config_name]["build"]())

def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "weighted_f1": f1_score(y_test, pred, average="weighted"),
        # The business metric: a premium batch sold as bulk is real money lost.
        "premium_recall": recall_score(y_test, pred, labels=["premium"], average="macro",
                                       zero_division=0),
    }

def log_plots(model, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, labels=config.TIERS, ax=ax,
                                          cmap="Reds", colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)

    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        imp = pd.Series(estimator.feature_importances_, index=OUTPUT_FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(6, 5))
        imp.tail(12).plot.barh(ax=ax, color="#8e1b3a")
        ax.set_title("Top feature importances")
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/feature_importance.png")
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Train wine-quality models")
    parser.add_argument("--data", default=str(WINE_TABLE))
    parser.add_argument("--only", nargs="*", help="subset of configuration names")
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    df = pd.read_csv(args.data)
    X_train, X_test, y_train, y_test = split(df)
    data_sha = file_fingerprint(args.data)

    dataset = mlflow.data.from_pandas(df, source=args.data, name=f"wine_quality_{data_sha}",
                                      targets=config.TARGET)

    names = args.only or list(CONFIGS)
    logger.info("Training %d configurations on %d rows (data sha %s)\n", len(names), len(df), data_sha)

    results = []
    for name in names:
        with mlflow.start_run(run_name=name):
            mlflow.log_input(dataset, context="training")
            mlflow.set_tags({"config": name, "description": CONFIGS[name]["description"],
                             "data_sha": data_sha})

            model = build_pipeline(name)
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            params = {f"model__{k}": v for k, v in model.named_steps["model"].get_params().items()
                      if isinstance(v, (int, float, str, bool)) or v is None}
            mlflow.log_params({"config": name, "n_train": len(X_train), "n_test": len(X_test),
                               "n_features": len(OUTPUT_FEATURES), **params})
            mlflow.log_metrics(metrics)
            log_plots(model, X_test, y_test)

            example = X_train.head(3)
            model_info = mlflow.sklearn.log_model(
                model,
                name="wine_quality_model",
                signature=infer_signature(example, model.predict(example)),
                input_example=example,
                # The pipeline references src.features.WineFeatures; ship that code with
                # the model so it loads anywhere, not only inside this repository.
                code_paths=[SRC_DIR],
                # MLflow 3 saves sklearn models with skops, which refuses types it cannot
                # verify. We created these files ourselves, so we vouch for these types.
                skops_trusted_types=TRUSTED_TYPES,
            )
            # Record the logged model's id so register.py registers exactly this model.
            mlflow.set_tag("logged_model_id", model_info.model_id)

            results.append((name, metrics))
            logger.info("  %-16s macro-F1 %.3f | acc %.3f | premium recall %.3f",
                        name, metrics["macro_f1"], metrics["accuracy"], metrics["premium_recall"])

    best = max(results, key=lambda r: r[1]["macro_f1"])
    logger.info("\nBest: %s (macro-F1 %.3f). Next: python -m src.register", best[0], best[1]["macro_f1"])


if __name__ == "__main__":
    main()
