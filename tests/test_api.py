import importlib

from fastapi.testclient import TestClient


def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'empty.db'}")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import api.app as app_module
    importlib.reload(app_module)
    return TestClient(app_module.app)


def test_health_and_ui(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    assert c.get("/health").json()["status"] == "healthy"
    assert "VinhoVerde" in c.get("/").text


def test_predict_without_champion_is_503_not_500(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    wine = c.get("/examples").json()["red"]
    assert c.post("/predict", json=wine).status_code == 503


def test_chat_without_key_explains_itself(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    r = c.post("/chat", json={"message": "hi"})
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["detail"]
