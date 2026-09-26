#!/bin/bash
# Starts the MLflow tracking server (same setup as the class stacks).
# chmod 777 is a teaching shortcut — never ship it.
mkdir -p /mlflow/db /artifacts
chmod -R 777 /mlflow /artifacts

# --allowed-hosts: MLflow rejects Host headers it does not know (DNS-rebinding
# protection). "mlflow:5000" is how the other containers reach it.
exec mlflow server \
    --backend-store-uri sqlite:////mlflow/db/mlflow.db \
    --artifacts-destination /artifacts \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts "mlflow:5000,mlflow,localhost:5001,localhost:5000,localhost,127.0.0.1:5001,127.0.0.1:5000,127.0.0.1"
