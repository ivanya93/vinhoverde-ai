"""
Register the best run, validate it, and promote it to @champion only if it earns it.

    python -m src.register              # register best run -> @candidate -> gate -> @champion
    python -m src.register --dry-run    # everything except moving @champion

The gate has two conditions:
    1. macro-F1 on the held-out test set >= MIN_MACRO_F1 (an absolute floor)
    2. macro-F1 >= the current champion's, scored on the SAME test set (no regressions)

The API serves models:/<name>@champion, so moving the alias IS the deployment. After a
promotion:  curl -X POST http://localhost:8080/reload
"""

import argparse
import json
import logging
import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src import config
from src.data import WINE_TABLE
from src.train import evaluate, split

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def best_run(client: MlflowClient):
    experiment = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit(f"No experiment '{config.EXPERIMENT_NAME}'. Run python -m src.train first.")
    runs = client.search_runs([experiment.experiment_id], filter_string="attributes.status = 'FINISHED'",
                              order_by=["metrics.macro_f1 DESC"], max_results=1)
    if not runs:
        raise SystemExit("No runs found. Run python -m src.train first.")
    return runs[0]


def score(uri: str, X_test, y_test) -> dict:
    return evaluate(mlflow.sklearn.load_model(uri), X_test, y_test)


def main():
    parser = argparse.ArgumentParser(description="Register and promote the best model")
    parser.add_argument("--data", default=str(WINE_TABLE))
    parser.add_argument("--dry-run", action="store_true", help="do not move @champion")
    parser.add_argument("--report", default="gate_report.json", help="where to write the gate result")
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient()

    run = best_run(client)
    logger.info("Best run: %s (%s) macro-F1 %.3f", run.info.run_name, run.info.run_id[:8],
                run.data.metrics["macro_f1"])

    # Re-running without new training should not pile up identical versions.
    try:
        current = client.get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
        if current.run_id == run.info.run_id:
            logger.info("That run is already @%s (v%s). Nothing to do.", config.MODEL_ALIAS, current.version)
            with open(args.report, "w") as fh:
                json.dump({"passed": True, "promoted": False, "reason": "already champion"}, fh)
            return
    except Exception:  # noqa: BLE001 — no champion yet
        pass

    # Register the exact logged model by id (the MLflow 3 way).
    version = mlflow.register_model(f"models:/{run.data.tags['logged_model_id']}", config.MODEL_NAME)
    client.set_model_version_tag(config.MODEL_NAME, version.version, "config", run.info.run_name)
    client.set_model_version_tag(config.MODEL_NAME, version.version, "data_sha",
                                 run.data.tags.get("data_sha", "unknown"))
    client.set_registered_model_alias(config.MODEL_NAME, "candidate", version.version)
    logger.info("Registered %s v%s -> @candidate", config.MODEL_NAME, version.version)

    # Score candidate and champion on the same held-out data.
    _, X_test, _, y_test = split(pd.read_csv(args.data))
    cand = score(f"models:/{config.MODEL_NAME}@candidate", X_test, y_test)

    try:
        champ_version = client.get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
        champ = score(f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}", X_test, y_test)
        logger.info("Current champion v%s macro-F1 %.3f", champ_version.version, champ["macro_f1"])
    except Exception:  # noqa: BLE001 — no champion yet is a normal first-run state
        champ_version, champ = None, None
        logger.info("No champion yet.")

    checks = {
        f"macro_f1 >= {config.MIN_MACRO_F1}": cand["macro_f1"] >= config.MIN_MACRO_F1,
        "not worse than champion": champ is None or cand["macro_f1"] >= champ["macro_f1"],
    }
    passed = all(checks.values())
    same_as_champion = champ_version is not None and champ_version.version == version.version

    logger.info("\nCandidate v%s macro-F1 %.3f", version.version, cand["macro_f1"])
    for name, ok in checks.items():
        logger.info("  [%s] %s", "PASS" if ok else "FAIL", name)

    promoted = False
    if passed and not args.dry_run and not same_as_champion:
        client.set_registered_model_alias(config.MODEL_NAME, config.MODEL_ALIAS, version.version)
        promoted = True
        logger.info("\n@%s -> v%s. Tell the API:  curl -X POST http://localhost:8080/reload",
                    config.MODEL_ALIAS, version.version)
    elif not passed:
        logger.info("\nGate failed. The API keeps serving the current champion.")

    report = {
        "candidate_version": version.version,
        "candidate_metrics": cand,
        "champion_version": champ_version.version if champ_version else None,
        "champion_metrics": champ,
        "checks": checks,
        "passed": passed,
        "promoted": promoted,
    }
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
