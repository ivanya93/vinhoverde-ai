"""Train, register and gate against a throwaway local MLflow store."""

import json
import sys

import mlflow
import pytest

from src import config, register, train
from src.data import load_uci


@pytest.fixture()
def tracking(tmp_path, raw_dir, monkeypatch):
    table = tmp_path / "wine.csv"
    load_uci(raw_dir).to_csv(table, index=False)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)  # local artifacts land in tmp
    return uri, table


def test_train_then_gate_promotes_first_model(tracking, tmp_path, monkeypatch):
    uri, table = tracking
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(table), "--only", "logreg_baseline",
                                      "--mlflow-uri", uri])
    train.main()

    monkeypatch.setattr(config, "MIN_MACRO_F1", 0.0)
    report = tmp_path / "gate.json"
    monkeypatch.setattr(sys, "argv", ["register", "--data", str(table), "--mlflow-uri", uri,
                                      "--report", str(report)])
    register.main()

    result = json.loads(report.read_text())
    assert result["passed"] and result["promoted"]
    mlflow.set_tracking_uri(uri)
    model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@champion")
    assert set(model.classes_) <= set(config.TIERS)


def test_gate_blocks_weak_model(tracking, tmp_path, monkeypatch):
    uri, table = tracking
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(table), "--only", "logreg_baseline",
                                      "--mlflow-uri", uri])
    train.main()

    monkeypatch.setattr(config, "MIN_MACRO_F1", 0.999)
    report = tmp_path / "gate.json"
    monkeypatch.setattr(sys, "argv", ["register", "--data", str(table), "--mlflow-uri", uri,
                                      "--report", str(report)])
    register.main()
    assert json.loads(report.read_text())["promoted"] is False
